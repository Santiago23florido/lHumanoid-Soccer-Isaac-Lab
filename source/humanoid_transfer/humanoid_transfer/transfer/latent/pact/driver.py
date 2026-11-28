"""PACT driving a body: a trained :class:`~.policy.PactPolicy` behind a tracker's interface.

Each control step it reads what the body senses (joint angles and speeds, the joint targets
in force, projected gravity, base angular velocity) and the tracker's observation, whose
last ``view * 30 + 27`` numbers are the plan (:func:`~.policy.plan_from_view`), and returns
the action in the body's own action units (``q_target = q0 + scale * a`` over the action
term's joints). It keeps the encoder's history; :meth:`PactDriver.reset` refills it for
environments that start an episode.

The anchor inputs are G-A3's ``pact_kd`` ones, computed here in torch, step by step:
the kinematic anchors, the drive loads, and the odometry built from them
(:func:`anchor_inputs`; ``tests/test_pact_driver.py`` holds it to the offline gates' NumPy
versions).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from humanoid_transfer.common.skill_state import SKILL_LAYOUT
from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
from humanoid_transfer.transfer.latent.pact.descriptors import GRAVITY, BodyDescriptor
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies
from humanoid_transfer.transfer.latent.pact.policy import SKILL, PactPolicy, plan_from_view

S = {f.name: SKILL_LAYOUT.slice_of(f.name) for f in SKILL_LAYOUT.fields}


def anchor_inputs(
    anchors: KinematicAnchors,
    drive: DriveLoads,
    q: torch.Tensor,
    qd: torch.Tensor,
    gravity: torch.Tensor,
    omega: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """``(N, 27)``: the analytic ``z*`` with the odometry and drive loads in place (G-A3's
    ``with_physics(analytic_z, physics_bases)``), from raw joint readings (rad, rad/s)."""
    a = anchors(q, qd, gravity, omega)
    z = q.new_zeros(q.shape[0], SKILL_LAYOUT.size)
    z[:, S["height"]] = a["height"]
    z[:, S["gravity"]] = a["gravity"]
    z[:, S["base_angular_velocity"]] = a["base_angular_velocity"]
    z[:, S["foot_position"]] = a["foot_position"].reshape(-1, 6)
    relative = a["foot_velocity_relative"].reshape(-1, 2, 3)
    z[:, S["angular_momentum"]] = a["angular_momentum"]
    loads = drive(q, qd, gravity, target=target)["loads"]
    weights = loads.clamp_min(0.0)
    total = weights.sum(-1, keepdim=True)
    weights = torch.where(total > 0.05, weights / total.clamp_min(1e-9), 0.5)
    odometry = -(weights[..., None] * relative).sum(1)
    z[:, S["com_velocity"]] = odometry
    z[:, S["foot_load"]] = loads
    z[:, S["foot_velocity"]] = (relative + odometry[:, None]).reshape(-1, 6)
    return z


class PactDriver:
    """A trained PACT policy driving ``num_envs`` copies of one body (or the URMA baseline,
    ``latent/pact/urma.py``, which takes the same inputs; a checkpoint's ``kind`` says which)."""

    def __init__(
        self,
        checkpoint: dict,
        descriptor: BodyDescriptor,
        contacts: list[int],
        anchors: KinematicAnchors,
        drive: DriveLoads,
        default_joint_pos: torch.Tensor,
        action_joints: list[int] | slice,
        action_scale: float | torch.Tensor,
        view: int,
        skill_mean: torch.Tensor,
        skill_std: torch.Tensor,
        control_dt: float,
        num_envs: int,
        device: torch.device | str,
        mirror=None,
    ) -> None:
        kind = checkpoint.get("kind", "pact")
        self.routing: dict = {}
        if kind == "urma":  # the end-to-end baseline: same interface, anchors ignored
            from humanoid_transfer.transfer.latent.pact.urma import UrmaDims, UrmaPolicy

            dims = UrmaDims(**checkpoint["dims"])
            self.model = UrmaPolicy(dims).to(device).eval()
        elif kind == "hpt":  # the per-body-stem baseline: this body's stem and head
            from humanoid_transfer.transfer.latent.pact.hpt import HptDims, HptPolicy

            raw = checkpoint["dims"]
            dims = HptDims(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})
            if descriptor.name not in dims.bodies:
                raise ValueError(f"this HPT model has no stem for {descriptor.name}")
            self.model = HptPolicy(dims).to(device).eval()
            body_id = dims.bodies.index(descriptor.name)
            self.routing = {"body_ids": torch.full((num_envs,), body_id, device=device)}
        else:
            dims = PactDims(**checkpoint["dims"])
            self.model = PactPolicy(dims).to(device).eval()
        self.model.load_state_dict(checkpoint["state"])
        self.device = torch.device(device)
        as_t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
        self.z_mean, self.z_std = as_t(checkpoint["z_mean"]), as_t(checkpoint["z_std"])
        self.a_mean, self.a_std = as_t(checkpoint["a_mean"]), as_t(checkpoint["a_std"])
        self.table = stack_bodies([descriptor], [contacts]).to(device)
        residual = checkpoint.get("residual")
        if residual is not None and kind != "pact":
            raise ValueError("a body residual (G-N2) belongs to a PACT model")
        if residual is not None:  # a body added by PPO (G-N2): its E_b and descriptor residuals
            from humanoid_transfer.transfer.latent.pact.actor import BodyResidual

            shifted = BodyResidual(self.model.embodiment, self.table, dims.body_tokens, dims.width)
            with torch.no_grad():
                shifted.delta_e.copy_(torch.as_tensor(residual["delta_e"]))
                shifted.delta_d.copy_(torch.as_tensor(residual["delta_d"]))
            self.model.embodiment = shifted.to(device).eval()
        self.kind = kind
        self.impedance_anchor = bool(kind == "pact" and getattr(dims, "impedance_anchor", False))
        self.decoupled_anchor = (
            str(getattr(dims, "decoupled_anchor", "")) if self.impedance_anchor else ""
        )
        self.anchors, self.drive = anchors.to(device), drive.to(device)
        self.q0 = default_joint_pos.to(device)
        self.action_joints = action_joints
        self.action_scale = action_scale
        self.view = view
        self.skill_mean, self.skill_std = skill_mean.to(device), skill_std.to(device)
        self.time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
        self.alpha_dt = float(1.0 - math.exp(-(control_dt / self.time_unit) / 0.08))
        self.times = (torch.arange(1, view + 1, device=device) * control_dt / self.time_unit)[None]
        if kind == "pact":
            with torch.no_grad():
                self._emb_one = self.model.embodiment(self.table)
        else:
            self._emb_one = None
        joints, history = len(descriptor.joint_names), dims.history
        self.joint_history = torch.zeros(num_envs, joints, history, 3, device=device)
        self.imu_history = torch.zeros(num_envs, history, 6, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.symmetrize = mirror is not None
        if mirror is not None:
            self._m_joints = as_t(mirror.joints)
            self._m_offset = as_t(mirror.offset)
            self._m_drives = as_t(mirror.drives)
            self._m_drive_offset = as_t(mirror.drive_offset)
            self._m_imu = as_t(mirror.imu_matrix)
            self._m_z = as_t(mirror.z)
            self._m_path = as_t(mirror.path(np.eye(3, dtype=np.float32)))
            self._m_wrench = as_t(mirror.wrench_matrix)

    def reset(self, ids: torch.Tensor) -> None:
        """Environments ``ids`` start an episode: their history refills on the next step."""
        self.fresh[ids] = True

    @torch.no_grad()
    def __call__(
        self,
        policy_obs: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        target: torch.Tensor,
        gravity: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """``-> (N, action_dim)`` in the action term's units."""
        n = q.shape[0]
        # The target history as the model was trained on it: the torque it commands (drive_ratio).
        commanded = self.drive.drive_ratio(q, qd, target)
        joint_now = torch.stack(
            [q - self.q0, qd * self.time_unit, commanded], dim=-1
        )  # (N, J, 3)
        imu_now = torch.cat([gravity, omega * self.time_unit], dim=-1)
        self.joint_history = torch.roll(self.joint_history, -1, dims=2)
        self.joint_history[:, :, -1] = joint_now
        self.imu_history = torch.roll(self.imu_history, -1, dims=1)
        self.imu_history[:, -1] = imu_now
        if self.fresh.any():
            f = self.fresh
            self.joint_history[f] = joint_now[f][:, :, None].expand_as(self.joint_history[f])
            self.imu_history[f] = imu_now[f][:, None].expand_as(self.imu_history[f])
            self.fresh[:] = False

        anchor = anchor_inputs(self.anchors, self.drive, q, qd, gravity, omega, target)
        tail = policy_obs[:, policy_obs.shape[-1] - (self.view * (SKILL + 3) + SKILL) :]
        upcoming, estimate, path = plan_from_view(tail, self.view, self.skill_mean, self.skill_std)
        if not self.symmetrize:
            mask = torch.ones(n, self.view, dtype=torch.bool, device=self.device)
            extra = dict(self.routing)
            if self._emb_one is not None:
                extra["emb_cached"] = (
                    self._emb_one[0].expand(n, -1, -1),
                    self._emb_one[1].expand(n, -1, -1),
                )
            if self.impedance_anchor:
                phys = self.drive.operational_anchor_from_ratio(q, gravity, commanded)
                phys["alpha_dt"] = q.new_full((n,), self.alpha_dt)
                if self.decoupled_anchor == "radian":
                    d_gain, d_offset = self.drive.drive_gain_offset(q, qd, default_pos=self.q0)
                    phys["drive_gain"] = d_gain.expand(n, -1)
                    phys["drive_offset"] = d_offset
                extra["phys"] = phys
            _, out = self.model(
                self.table.take(torch.zeros(n, dtype=torch.long, device=self.device)),
                self.joint_history,
                self.imu_history,
                (anchor - self.a_mean) / self.a_std,
                (upcoming - self.z_mean) / self.z_std,
                (upcoming - estimate[:, None]) / self.z_std,
                self.times.expand(n, -1),
                path,
                mask,
                **extra,
            )
            if self.impedance_anchor and self.decoupled_anchor != "radian":
                out = self.drive.targets_from_torque(q, qd, out, default_pos=self.q0)
            return out[:, self.action_joints] / self.action_scale

        # Exact Reynolds projection onto sagittally equivariant policies: 0.5 * (pi(x) + M pi(M x)),
        # batched in a single 2N forward pass (matches the symmetrised teachers, symmetrize_actor.py).
        jh = self.joint_history.permute(0, 2, 3, 1)  # (N, H, 3, J)
        jh_m = torch.stack(
            [
                jh[:, :, 0] @ self._m_joints.T + self._m_offset,
                jh[:, :, 1] @ self._m_joints.T,
                jh[:, :, 2] @ self._m_drives.T,
            ],
            dim=2,
        ).permute(0, 3, 1, 2)
        imu_m = self.imu_history @ self._m_imu.T
        anchor_m = anchor @ self._m_z.T
        up_m = upcoming @ self._m_z.T
        est_m = estimate @ self._m_z.T
        path_m = path @ self._m_path.T
        n2 = 2 * n
        mask = torch.ones(n2, self.view, dtype=torch.bool, device=self.device)
        extra = {k: v.repeat(2) for k, v in self.routing.items()}
        if self._emb_one is not None:
            extra["emb_cached"] = (
                self._emb_one[0].expand(n2, -1, -1),
                self._emb_one[1].expand(n2, -1, -1),
            )
        if self.impedance_anchor:
            phys = self.drive.operational_anchor_from_ratio(q, gravity, commanded)
            phys_2 = {
                "tau_g": torch.cat([phys["tau_g"], phys["tau_g"] @ self._m_joints.T], dim=0),
                "j_eff": torch.cat(
                    [
                        phys["j_eff"],
                        torch.einsum("ij,njk->nik", self._m_joints, phys["j_eff"])
                        @ self._m_wrench.T,
                    ],
                    dim=0,
                ),
                "j_pinv": torch.cat(
                    [
                        phys["j_pinv"],
                        torch.einsum("ij,njk->nik", self._m_joints, phys["j_pinv"])
                        @ self._m_wrench.T,
                    ],
                    dim=0,
                ),
                "w_t": torch.cat([phys["w_t"], phys["w_t"] @ self._m_wrench.T], dim=0),
                "k_eff": phys["k_eff"],
                "tau_eff_dimless": phys["tau_eff_dimless"],
                "alpha_dt": q.new_full((n2,), self.alpha_dt),
            }
            if self.decoupled_anchor == "radian":
                d_gain, d_offset = self.drive.drive_gain_offset(q, qd, default_pos=self.q0)
                d_offset_m = d_offset @ self._m_drives.T - d_gain * self._m_drive_offset
                phys_2["drive_gain"] = d_gain.expand(n2, -1)
                phys_2["drive_offset"] = torch.cat([d_offset, d_offset_m], dim=0)
            extra["phys"] = phys_2
        _, out_2 = self.model(
            self.table.take(torch.zeros(n2, dtype=torch.long, device=self.device)),
            torch.cat([self.joint_history, jh_m], dim=0),
            torch.cat([self.imu_history, imu_m], dim=0),
            (torch.cat([anchor, anchor_m], dim=0) - self.a_mean) / self.a_std,
            (torch.cat([upcoming, up_m], dim=0) - self.z_mean) / self.z_std,
            torch.cat([upcoming - estimate[:, None], up_m - est_m[:, None]], dim=0) / self.z_std,
            self.times.expand(n2, -1),
            torch.cat([path, path_m], dim=0),
            mask,
            **extra,
        )
        if self.impedance_anchor and self.decoupled_anchor != "radian":
            tau_sym = 0.5 * (out_2[:n] + out_2[n:] @ self._m_drives.T)
            out = self.drive.targets_from_torque(q, qd, tau_sym, default_pos=self.q0)
        else:
            out = 0.5 * (out_2[:n] + (out_2[n:] @ self._m_drives.T + self._m_drive_offset))
        return out[:, self.action_joints] / self.action_scale
