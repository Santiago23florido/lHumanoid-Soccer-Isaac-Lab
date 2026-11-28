"""PACT as a policy: the anchored state encoder and the joint decoder, one model for every body.

The encoder is G-A3's ``pact_kd`` (kinematic anchors and drive loads as inputs, the CoM
velocity, height correction and loads learned); the decoder reads its entity tokens, ``E_b``
and the task's plan, and gives every joint its position target in radians from the joint's
default. Nothing in it is per body: a body enters through its descriptor tokens only.

The plan is what each tracker observes (``ReferenceCommand.observation`` and ``.path``),
decoded out of its generator's normalised units into raw ``z*``
(:func:`plan_from_view`): the trackers' views hold different step counts (8 on the G1, 4 on
the NAO, 9 on H1 and Cassie) over the same span of each body's own time ``sqrt(L/g)``, and
the NAO's generator normalises with its own statistics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from humanoid_transfer.common.skill_state import SKILL_LAYOUT
from humanoid_transfer.transfer.latent.pact.model import (
    BodyBatch,
    EmbodimentEncoder,
    JointDecoder,
    PactDims,
    StateEncoder,
)
from humanoid_transfer.transfer.latent.pact.tokens import split

SKILL = 27
_S = {f.name: SKILL_LAYOUT.slice_of(f.name) for f in SKILL_LAYOUT.fields}
DYNAMIC_Z_INDEX: tuple[int, ...] = tuple(
    i
    for name in (
        "gravity",
        "base_angular_velocity",
        "com_velocity",
        "foot_velocity",
        "foot_load",
        "angular_momentum",
    )
    for i in range(_S[name].start, _S[name].stop)
)
"""20 morphology-invariant dynamic fields of ``z*`` (omits static body-fingerprinting height and
foot_position) fed to the operational-space impedance head."""

PLAN_LIMIT = 50.0
"""The clamp on a tracker's plan view, normalised units (``tracking/command._PLAN_LIMIT``)."""


def view_steps(observation_width: int, body_width: int) -> int:
    """Plan steps in a tracker's view: its policy observation is ``body_width`` numbers of
    the body (IMU, joints, last action), then ``view * 27 + 27`` (plan relative to the
    estimate, then the estimate) and ``view * 3`` (path)."""
    rest = observation_width - body_width - SKILL
    if rest <= 0 or rest % (SKILL + 3):
        raise ValueError(f"{observation_width} - {body_width} is not a tracker's plan view")
    return rest // (SKILL + 3)


def plan_from_view(
    tail: np.ndarray, view: int, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``tail (..., view * 30 + 27)``, the plan part of a tracker's observation, in raw units.

    Returns ``(upcoming (..., view, 27), estimate (..., 27), path (..., view, 3))``; ``mean``
    and ``std`` are the generator's ``skill_norm``. A view clamped at the plan limit stays
    clamped (rare; the same clamp the tracker saw).
    """
    relative = tail[..., : view * SKILL].reshape(*tail.shape[:-1], view, SKILL)
    estimate_norm = tail[..., view * SKILL : view * SKILL + SKILL]
    path = tail[..., view * SKILL + SKILL :].reshape(*tail.shape[:-1], view, 3)
    upcoming = (relative + estimate_norm[..., None, :]) * std + mean
    return upcoming, estimate_norm * std + mean, path


class PactPolicy(nn.Module):
    """``E_b``, the anchored state encoder and the timed-plan joint decoder.

    With ``dims.anchor_inputs`` off (the no-anchor ablation) the encoder does not read the
    analytic anchors (``anchor_in`` is accepted and ignored); trained without the readout
    loss, its entity latent is then anchored by nothing.
    """

    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        if not dims.timed_plan:
            raise ValueError("PactPolicy reads a timed plan")
        self.dims = dims
        self.embodiment = EmbodimentEncoder(dims)
        self.state = StateEncoder(dims)
        self.decoder = JointDecoder(dims)
        if dims.impedance_anchor:
            w, op_w = dims.width, dims.op_width
            # Decoupled mode detaches tau_anchor before the residual decoder; keeping dropout
            # inside op_head would inject irreducible stochastic noise into the residual target.
            op_drop = 0.0 if dims.decoupled_anchor else 0.10
            self.op_step_enc = nn.Sequential(
                nn.Linear(SKILL + 3 + 1, op_w),
                nn.LayerNorm(op_w),
                nn.GELU(),
                nn.Dropout(op_drop),
                nn.Linear(op_w, op_w),
            )
            self.op_step_attn = nn.Linear(op_w, 4)
            self.op_head = nn.Sequential(
                nn.Linear(len(DYNAMIC_Z_INDEX) + 12 + 4 * op_w, op_w),
                nn.LayerNorm(op_w),
                nn.GELU(),
                nn.Dropout(op_drop),
                nn.Linear(op_w, op_w),
                nn.LayerNorm(op_w),
                nn.GELU(),
                nn.Dropout(op_drop),
                nn.Linear(op_w, 24),
            )
            phys_in = 27 if dims.decoupled_anchor else 25
            self.joint_phys = nn.Sequential(
                nn.Linear(phys_in, w), nn.LayerNorm(w), nn.GELU(), nn.Linear(w, w)
            )
            # Tiny near-zero init: starts at exact physical hold while keeping step-0 grad live.
            nn.init.normal_(self.op_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.op_head[-1].bias)
            nn.init.normal_(self.decoder.head[-1].weight, std=1e-3)
            nn.init.zeros_(self.decoder.head[-1].bias)
            self.register_buffer(
                "dyn_idx", torch.as_tensor(DYNAMIC_Z_INDEX, dtype=torch.long), persistent=False
            )
            # Physical friction-cone / ZMP support-polygon wrench bounds (m g force, m g L moment)
            # and operational-space motion cap for state_anchor=True (even under sagittal mirror).
            # Moment bounds (0.50/0.25) accommodate large-sole bodies (NAO) while blocking internal
            # mechanical-stop saturation torques (e.g. Cassie hip_rotation at +-0.98 m g L).
            self.register_buffer(
                "w_cap",
                torch.tensor([1.5, 1.5, 2.0, 0.50, 0.50, 0.25] * 2, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "x_cap",
                torch.full((12,), 1.0, dtype=torch.float32),
                persistent=False,
            )

    def forward(
        self,
        bodies: BodyBatch,
        joint_obs: torch.Tensor,
        imu: torch.Tensor,
        anchor_in: torch.Tensor,
        plan: torch.Tensor,
        plan_relative: torch.Tensor,
        plan_time: torch.Tensor,
        plan_path: torch.Tensor,
        plan_mask: torch.Tensor,
        *,
        phys: dict[str, torch.Tensor] | None = None,
        emb_cached: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """``-> (learned (B, 6), targets_or_tau (B, J))``.

        ``learned``: CoM velocity, height correction, the two loads (as G-A3's encoder,
        in whatever normalisation its targets were given). When ``dims.impedance_anchor`` is
        False, the second output is joint position targets in radians from default; when True,
        it is dimensionless commanded torque in ``m g L`` units (converted analytically to PD
        targets by :meth:`DriveLoads.targets_from_torque`).
        """
        if self.dims.zero_cmd_history:
            joint_obs = joint_obs.clone()
            joint_obs[:, :, :, 2] = 0.0
        if emb_cached is None:
            embodiment, descriptor = self.embodiment(bodies)
        else:
            embodiment, descriptor = emb_cached
        if self.dims.impedance_anchor and self.training:
            if self.dims.emb_drop > 0 and embodiment.shape[1] > 0:
                keep = (
                    torch.rand(embodiment.shape[0], 1, 1, device=embodiment.device)
                    > self.dims.emb_drop
                ).to(embodiment.dtype)
                embodiment = embodiment * keep
            if self.dims.desc_noise > 0:
                descriptor = (
                    descriptor
                    + self.dims.desc_noise
                    * torch.randn_like(descriptor)
                    * bodies.mask[..., None]
                )
        anchors = split(anchor_in) if self.dims.anchor_inputs else None
        out = self.state(bodies, descriptor, embodiment, joint_obs, imu, anchors)
        learned = torch.cat(
            [out["centroid"][:, 1:4], out["centroid"][:, 0:1], out["contacts"][:, :, 6]], dim=1
        )
        value_c, value_k = split(plan)
        delta_c, delta_k = split(plan_relative)
        plan_c = torch.cat([value_c, delta_c], dim=-1)
        plan_k = torch.cat([value_k, delta_k], dim=-1)

        if not self.dims.impedance_anchor:
            targets = self.decoder(
                bodies,
                out["joints"],
                out["latent"],
                embodiment,
                plan_c,
                plan_k,
                plan_time=plan_time,
                plan_path=plan_path,
                plan_mask=plan_mask,
            )
            return (learned, targets, {}) if return_aux else (learned, targets)

        if phys is None:
            raise ValueError("impedance_anchor=True requires phys dict (tau_g, j_eff, j_pinv, ...)")
        tau_g = phys["tau_g"]
        j_eff = phys["j_eff"]
        j_pinv = phys["j_pinv"]
        k_eff = phys["k_eff"]
        tau_eff = phys["tau_eff_dimless"]
        alpha_dt = phys["alpha_dt"]
        w_t = phys["w_t"]

        # 1. Morphology-blind Operational-Space Impedance Anchor (12D force + 12D motion)
        st_dyn = anchor_in.index_select(1, self.dyn_idx)
        sf = torch.cat([plan_relative, plan_path, plan_time[..., None]], dim=-1)
        h_op = self.op_step_enc(sf)
        logits = self.op_step_attn(h_op).masked_fill(~plan_mask[..., None], -1e4)
        pooled = torch.einsum("bhk,bhw->bkw", torch.softmax(logits, dim=1), h_op).reshape(
            h_op.shape[0], -1
        )
        if self.dims.state_anchor:
            zero_w = st_dyn.new_zeros(st_dyn.shape[0], 12)
            op_out = self.op_head(torch.cat([st_dyn, zero_w, pooled], dim=-1))
            if self.dims.decoupled_anchor:
                # Analytical Coulomb friction-cone + ZMP support-polygon wrench bounds derived from
                # each body's dimensionless sole polygon (foot_length/L, foot_width/L in body[:, 4:6]).
                foot_l = bodies.body[:, 4]
                foot_w = bodies.body[:, 5]
                fx = foot_l.new_full((foot_l.shape[0],), 1.5)
                fy = foot_l.new_full((foot_l.shape[0],), 1.5)
                fz = foot_l.new_full((foot_l.shape[0],), 2.0)
                mx = (1.5 * foot_w).clamp(0.05, 0.50)
                my = (1.2 * foot_l).clamp(0.08, 0.55)
                mz = (0.4 * (foot_l + foot_w)).clamp(0.04, 0.25)
                w6 = torch.stack([fx, fy, fz, mx, my, mz], dim=-1)
                w_cap = torch.cat([w6, w6], dim=-1).to(op_out.dtype)
            else:
                w_cap = self.w_cap.to(op_out.dtype)
            x_cap = self.x_cap.to(op_out.dtype)
            w_cmd = w_cap * torch.tanh(op_out[:, :12] / w_cap)
            x_cmd = x_cap * torch.tanh(op_out[:, 12:] / x_cap)
            tau_anchor = (
                -tau_g
                - torch.einsum("njk,nk->nj", j_eff, w_cmd)
                + k_eff * torch.einsum("njk,nk->nj", j_pinv, x_cmd)
            ) * bodies.mask
        else:
            op_out = self.op_head(torch.cat([st_dyn, w_t, pooled], dim=-1))
            w_cmd, x_cmd = op_out[:, :12], op_out[:, 12:]
            cmd_r = joint_obs[:, :, -1, 2]
            tau_t_eff = torch.atanh(cmd_r.clamp(-0.98, 0.98)) * tau_eff
            tau_anchor = (
                tau_t_eff
                - alpha_dt[:, None] * torch.einsum("njk,nk->nj", j_eff, w_cmd)
                + alpha_dt[:, None] * k_eff * torch.einsum("njk,nk->nj", j_pinv, x_cmd)
            ) * bodies.mask

        # 2. Residual joint decoder
        if self.dims.decoupled_anchor in ("radian", "mgl"):
            # Stop-gradient decouples the morphology-blind ZMP prior from the body-specific residual:
            # op_head is trained purely by bc_anc while the residual sees the exact deterministic
            # prior (tau_prior, u_anc) it refines and learns clean seen-body precision.
            tau_prior = tau_anchor.detach()
            u_anc = (tau_prior / tau_eff.clamp_min(1e-6)).clamp(-3.0, 3.0)
            phys_feat = torch.cat(
                [tau_g[..., None], u_anc[..., None], tau_prior[..., None], j_eff, j_pinv], dim=-1
            )
            joints_tok = out["joints"] + self.joint_phys(phys_feat) * bodies.mask[..., None]
            delta_raw = self.decoder(
                bodies,
                joints_tok,
                out["latent"],
                embodiment,
                plan_c,
                plan_k,
                plan_time=plan_time,
                plan_path=plan_path,
                plan_mask=plan_mask,
                bot_mix=False,
            )
            if self.dims.decoupled_anchor == "radian":
                drive_gain = phys["drive_gain"]
                drive_offset = phys["drive_offset"]
                active = drive_gain > 0
                g_safe = drive_gain.clamp_min(1e-6)
                q_anc = torch.where(active, (u_anc - drive_offset) / g_safe, torch.zeros_like(u_anc))
                q_out = (q_anc + delta_raw) * bodies.mask
                if return_aux:
                    return learned, q_out, {
                        "tau_anchor": tau_anchor,
                        "q_anc": q_anc,
                        "delta_q": delta_raw * bodies.mask,
                        "op_out": op_out,
                    }
                return learned, q_out
            # decoupled_anchor == "mgl": dimensionally invariant residual in m g L units
            total_tau = (tau_prior + delta_raw) * bodies.mask
            if return_aux:
                return learned, total_tau, {
                    "tau_anchor": tau_anchor,
                    "delta_mgl": delta_raw * bodies.mask,
                    "op_out": op_out,
                }
            return learned, total_tau
        if self.dims.decoupled_anchor:
            raise ValueError(f"unknown decoupled_anchor mode: {self.dims.decoupled_anchor!r}")

        phys_feat = torch.cat([tau_g[..., None], j_eff, j_pinv], dim=-1)
        joints_tok = out["joints"] + self.joint_phys(phys_feat) * bodies.mask[..., None]

        delta_raw = self.decoder(
            bodies,
            joints_tok,
            out["latent"],
            embodiment,
            plan_c,
            plan_k,
            plan_time=plan_time,
            plan_path=plan_path,
            plan_mask=plan_mask,
            bot_mix=True,
        )
        is_leg = (j_eff.abs().sum(-1) > 1e-4).to(delta_raw.dtype)
        cap = is_leg * self.dims.leg_cap + (1.0 - is_leg) * self.dims.arm_cap
        delta_rel = cap * torch.tanh(delta_raw / cap)
        total_tau = (tau_anchor + delta_rel * tau_eff) * bodies.mask
        if return_aux:
            return learned, total_tau, {
                "tau_anchor": tau_anchor,
                "delta_rel": delta_rel,
                "op_out": op_out,
            }
        return learned, total_tau
