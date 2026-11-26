"""Gate G-D0: one decoder for every body -- the four trackers cloned offline into PACT.

Registered in ``docs/report/research_notes/icml/preregistration.md`` before it ran. On the
co-execution map captures (``pact_coexecute.sh``: each body's tracker on the 71 gait
probes), one :class:`~humanoid_transfer.transfer.latent.pact.policy.PactPolicy` -- G-A3's
``pact_kd`` encoder and the joint decoder -- learns every body's joint targets from what
the body senses and the plan its tracker was shown:

- per joint and step (history 4): ``q - q0`` (rad), ``q̇ sqrt(L/g)``, and the target in
  force, ``q_target - q0`` (rad; the tracker's last action, in radians);
- the IMU, and G-A3's ``pact_kd`` anchor inputs (kinematic anchors, odometry, drive loads);
- the plan: the tracker's view decoded into raw ``z*`` with its generator's normaliser
  (``plan_from_view``; the decoded estimate is checked against the capture's), each step
  as its value and its difference to the estimate, with its path and its time ahead in
  units of ``sqrt(L/g)``.

The label is the target the tracker set at that step, ``q_target(t + 1) - q0``, on the
joints whose target moves in the capture. Arms:

- ``pact_d``: one model for every body, BC loss (MSE over one pooled scale) plus G-A3's
  anchor loss on the encoder's six learned fields;
- ``pact_na``: ``pact_d`` without the anchor (the ablation): no analytic anchor inputs and
  no readout loss, everything else equal;
- ``pact_nb``: ``pact_d`` without ``E_b`` (the ablation): no body tokens, so no body summary
  is attended; the per-joint descriptor tokens, which identify the joints, are kept;
- ``urma_d``: one URMA policy for every body (``latent/pact/urma.py``): the same inputs but
  the anchor inputs, the same BC loss, no anchor loss (the end-to-end baseline);
- ``hpt_d``: HPT (``latent/pact/hpt.py``): a stem and a head per body around a shared trunk,
  no descriptors, no anchor (the per-body-stem baseline; not read on a held-out body);
- ``mlp_d``: one MLP per body on the same inputs, flattened (the in-body reference);
- ``hold``: the target in force kept (no learning; what persistence alone explains).

Test set: probes with index % 4 == 0 (as G-A3). Loss and metric are on the torque each target
commands, as a fraction of the joint's limit and softly saturated (``drive_ratio``, prereg
"Labels", 2026-11-22): RMSE over the controlled joints and ``R² = 1 - MSE / Var(label)``, with
the radian RMSE and R² beside (``rmse_rad``, ``r2_rad``); bootstrap over test environments. The
target history enters the models the same way.
The trained ``pact_d`` and its normalisers are saved for G-D1 (``logs/pact/models/``).

Usage::

    python transfer/scripts/pact_gate_d.py
    python transfer/scripts/pact_gate_d.py --hold-out nao --output <json>

DAgger (G-D1): each round's captures (``capture_contract.py --pact-model``) join the map
captures, and the previous round's model is the start::

    python transfer/scripts/pact_gate_d.py --dagger "logs/pact/dagger/<tag>/r*/{body}.npz"
        --init logs/pact/models/<previous>.pt --steps 8000 --save logs/pact/models/<next>.pt

(one command; ``pact_dagger.sh`` runs the rounds).

Adding a body (G-N3): a model trained without a body (``--hold-out``) is adapted to it
on its first ``--probes`` training probes of the map capture (one seeded permutation, so
budgets are nested and every arm gets the same probes), for ``--steps`` steps, and read on
that body's test probes. Each model adapts as its authors add a body: PACT by ``etune``
(a residual on the body's ``E_b`` and descriptor tokens), ``bitfit`` (also every bias and
LayerNorm parameter) or ``full``; HPT by ``stem`` (the body's own stem and head, trunk
frozen) or ``full``; ``zero`` reads the model unadapted (not HPT: it has no stem for an
unseen body). ``mlp_d`` beside it is the same budget from scratch::

    python transfer/scripts/pact_gate_d.py --hold-out nao --init <model without nao>
        --arms hold,pact_d,mlp_d --adapt etune --probes 4 --steps 2000 --output <json>

Windows: a DAgger capture holds many episode starts (PACT falls where the tracker did not),
and the driver acts there with its history refilled by the first reading. Windows that
cross the start of an episode begun inside the capture therefore repeat that reading
(:func:`window_rows`) and are trained on; the offline metric keeps G-D0's complete windows.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import torch  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.pact.actor import (  # noqa: E402
    BodyResidual,
    bias_and_norm,
)
from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    EFFORT_LIMIT_CLIP,
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.hpt import HptDims, HptPolicy  # noqa: E402
from humanoid_transfer.transfer.latent.pact.mirror import SampleMirror  # noqa: E402
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import (  # noqa: E402
    PLAN_LIMIT,
    SKILL,
    PactPolicy,
    plan_from_view,
)
from humanoid_transfer.transfer.latent.pact.urma import UrmaDims, UrmaPolicy  # noqa: E402
from pact_check_anchors import sole_override  # noqa: E402
from pact_gate_a import bootstrap_rmse  # noqa: E402
from pact_gate_a2 import analytic_z  # noqa: E402
from pact_gate_a3 import (  # noqa: E402
    DRIVES,
    PROBES,
    physics_bases,
    residual_targets,
    with_physics,
)
from torch import nn  # noqa: E402

_TEACHERS = {
    body: entry
    for body, entry in json.loads((REPO / "transfer/bodies/teachers.json").read_text("utf-8")).items()
    if not body.startswith("$")
}
CONTRACTS = ",".join(
    f"{body}={Path(entry['conditions']['HUMANOID_CONTRACT']).parent.name}"
    for body, entry in _TEACHERS.items()
)
"""The walk contract each body's trackers run under (``transfer/bodies/teachers.json``)."""
TEACHER_VARIANTS = {body: entry["variant"] for body, entry in _TEACHERS.items()}
"""Each body's final teacher in the audit's names: ``{variant}`` in --captures and --dagger."""
ADAPT_LR = {"zero": 0.0, "etune": 1e-3, "bitfit": 1e-3, "full": 1e-4, "stem": 3e-4}
"""G-N3's learning rates (G-N1's: residuals and biases 1e-3, every weight 1e-4; a body's
new stem and head, trained from their initialisation, at the trainer's own 3e-4)."""
PREFIX = {"pact_d": "d0", "pact_na": "na0", "pact_nb": "nb0", "urma_d": "u0", "hpt_d": "h0"}
"""Saved-model name prefix of each shared arm (``logs/pact/models/``)."""
ADAPT_ARMS = {
    "pact": ("zero", "etune", "bitfit", "full"),
    "urma": ("zero", "bitfit", "full"),
    "hpt": ("stem", "full"),
}


def _effective_effort(tau: np.ndarray, effort_cap: float | None) -> np.ndarray:
    """Cap unlimited drive effort limits (e.g. Digit's FLT_MAX) at ``EFFORT_LIMIT_CLIP * m g L``."""
    return tau if effort_cap is None else np.minimum(tau, effort_cap)


def drive_gain(drives: tuple, effort_cap: float | None = None) -> np.ndarray:
    """``(J,)`` kp / tau_eff: the commanded torque, in units of the joint's effective limit, per
    radian of target (0 for a joint whose drive has no stiffness or no torque: not controlled)."""
    kp, _, tau = (np.asarray(x, dtype=np.float64) for x in drives)
    tau = _effective_effort(tau, effort_cap)
    active = (kp > 0) & (tau > 0)
    return np.where(active, kp / np.where(active, tau, 1.0), 0.0).astype(np.float32)


def drive_offset(
    q: np.ndarray, qd: np.ndarray, q0: np.ndarray, drives: tuple, effort_cap: float | None = None
) -> np.ndarray:
    """``(..., J)`` the commanded torque ratio of the target ``q0`` at state ``(q, qd)``, so that
    ``drive_gain * (T - q0) + drive_offset`` is ``(kp (T - q) - kd qd) / tau_eff``."""
    kp, kd, tau = (np.asarray(x, dtype=np.float64) for x in drives)
    tau = _effective_effort(tau, effort_cap)
    active = (kp > 0) & (tau > 0)
    safe = np.where(active, tau, 1.0)
    return np.where(active, (kp * (q0 - q) - kd * qd) / safe, 0.0).astype(np.float32)


def drive_ratio(
    target: np.ndarray, q: np.ndarray, qd: np.ndarray, drives: tuple, effort_cap: float | None = None
) -> np.ndarray:
    """The torque a target commands, as a fraction of the joint's effective limit and softly
    saturated: ``tanh((kp (T - q) - kd qd) / tau_eff)`` (prereg, "Labels", 2026-11-22). Every
    target past the drive's saturation reads about +-1, whatever its size; a joint without a
    drive reads 0. Unlimited drives are capped at ``effort_cap = EFFORT_LIMIT_CLIP * m g L``."""
    return np.tanh(
        drive_gain(drives, effort_cap) * (target - q) + drive_offset(q, qd, q, drives, effort_cap)
    ).astype(np.float32)


def drive_loss(
    out: torch.Tensor, label: torch.Tensor, gain: torch.Tensor, offset: torch.Tensor, moved: torch.Tensor
) -> torch.Tensor:
    """Behaviour cloning on the torques the targets command, as fractions of each joint's limit
    and softly saturated (``drive_ratio``): a target error costs what it changes in the drive's
    torque, and two targets both past saturation cost nothing (prereg, "Labels", 2026-11-22)."""
    commanded = torch.tanh(gain * out + offset)
    taught = torch.tanh(gain * label + offset)
    return (commanded - taught).square()[moved].mean()


def unclamped(estimate: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(...)`` bool: rows whose estimate the tracker's view holds unclamped. A body far out
    of the plan's range (fallen) is seen clamped at ``PLAN_LIMIT`` normalised units; the
    clamped view is what the tracker, and so its label, saw: those rows are kept."""
    return (np.abs((estimate - mean) / std) < PLAN_LIMIT).all(-1)


def window_rows(
    done: np.ndarray, history: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(t, env, rows (N, history), start (N,))``: every step with a label in its episode
    (``t + 1`` in the capture, no reset at ``t``) whose window either lies inside one
    episode (no reset in ``[t - history, t]``, ``t >= history``: G-D0's windows) or crosses
    the start of an episode that began inside the capture (``start``). The rows before the
    start repeat its first reading: the driver refills its history at a reset, as the
    observation manager's history buffers do, so a policy acts there and must learn there.
    An episode that began before the capture has no known start and keeps G-D0's rule.
    Ordered by step, then environment."""
    steps, envs = done.shape
    first = np.full((steps, envs), -1, dtype=np.int64)
    current = np.full(envs, -1, dtype=np.int64)
    for s in range(steps):
        first[s] = current
        current = np.where(done[s], s + 1, current)
    ended = np.zeros((steps + 1, envs), dtype=np.int64)
    ended[1:] = np.cumsum(done, axis=0)
    tt = np.arange(steps)
    resets = ended[tt + 1] - ended[np.maximum(tt - history, 0)]  # resets in [t - h, t]
    full = (tt[:, None] >= history) & (resets == 0)
    starting = (first >= 0) & (tt[:, None] - first < history) & ~done.astype(bool)
    valid = (full | starting) & (tt[:, None] + 1 < steps)
    t, env = np.nonzero(valid)
    base = np.maximum(first[t, env], 0)
    rows = np.maximum(t[:, None] + np.arange(-history + 1, 1)[None], base[:, None])
    return t, env, rows, starting[t, env]


class BodyData:
    """One body's samples, from its map capture and any DAgger captures: inputs and labels.

    In a map capture the tracker drove, and the label is the target it set (read at the
    next step); in a DAgger capture (``capture_contract.py --pact-model``) PACT drove and
    the label is the tracker's own action on that state, ``q0 + scale * a`` on the action's
    joints. The offline metric reads the map capture's test probes only (``evaluate``).
    """

    def __init__(self, body, args, registry, device: torch.device | str = "cpu"):
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        unit = entry.get("leg_length_unit")
        self.descriptor = describe(card, leg_length=unit)
        self.contacts = contact_joints(card)
        self.device = torch.device(device)
        self.anchors = KinematicAnchors(card, unit, sole_override(body)).to(self.device)
        self.time_unit = float(np.sqrt(self.descriptor.leg_length / GRAVITY))
        self.card, self.name = card, body
        contract = dict(c.split("=") for c in args.contracts.split(","))[body]
        norm = torch.load(
            REPO / f"logs/contract/{contract}/generator.pt", map_location="cpu", weights_only=False
        )["generator"]
        self.skill = norm["skill_norm.mean"].numpy(), norm["skill_norm.std"].numpy()
        fields = {"body": body, "variant": TEACHER_VARIANTS.get(body, "")}
        paths = [REPO / args.captures.format(**fields)]
        for pattern in filter(None, args.dagger.split(";")):
            paths += sorted(REPO.glob(pattern.format(**fields)))
        parts = [
            self._samples(path, args, args.stride if i == 0 else args.dagger_stride)
            for i, path in enumerate(paths)
        ]
        for i, part in enumerate(parts):
            part["group"] = i * 1000 + part["group"]
            part["evaluate"] = part["test"] & (i == 0) & ~part["start"]
        static_keys = (
            "view",
            "times",
            "controlled",
            "drive_gain",
            "leg_mask",
            "tau_eff_dimless",
            "k_eff",
            "alpha_dt",
        )
        for key in parts[0]:
            if key not in static_keys:
                setattr(self, key, np.concatenate([part[key] for part in parts]))
        for key in static_keys:
            setattr(self, key, parts[0][key])
        self.captures = [str(path.relative_to(REPO)) for path in paths]
        self.capture_samples = [int(len(part["label"])) for part in parts]
        q0 = np.asarray(np.load(paths[0])["default_joint_pos"], dtype=np.float32)
        self.mirror = SampleMirror.for_body(REPO, body, self.descriptor.joint_names, q0)

    def _samples(self, path: Path, args, stride: int) -> dict:
        h = args.history
        data = np.load(path)
        names = tuple(str(n) for n in data["joint_names"])
        if names != tuple(self.descriptor.joint_names):
            raise ValueError(f"{path}: joints {names[:4]}... differ from the card's")
        done = data["done"]
        t, env, rows, start = window_rows(done, h)
        keep = np.arange(len(t))[::stride]
        if args.min_duty > 0 and "probes" in data.files:
            # The paper's task family is walking: rows of probes asking flight phases are dropped.
            duty = np.array([np.mean(p["duty"]) for p in json.loads(str(data["probes"]))])
            keep = keep[duty[data["probe_index"][env[keep]]] >= args.min_duty]
        t, env, rows, start = t[keep], env[keep], rows[keep], start[keep]

        q0 = data["default_joint_pos"].astype(np.float32)
        q, target = data["joint_pos"], data["joint_pos_target"]
        vel = data["joint_vel"]
        qd = vel * self.time_unit
        gains = tuple(np.asarray(data[k]) for k in DRIVES)
        effort_cap = float(EFFORT_LIMIT_CLIP * self.descriptor.mass * GRAVITY * self.descriptor.leg_length)
        cols = env[:, None]
        q_w, qd_w = q[rows, cols], qd[rows, cols]  # (N, h, J)
        # The target history as the torque it commands (drive_ratio), bounded and saturation-free.
        command_w = drive_ratio(target[rows, cols], q_w, vel[rows, cols], gains, effort_cap)
        out: dict = {}
        out["joint_obs"] = (
            np.stack([q_w - q0, qd_w, command_w], axis=-1)
            .transpose(0, 2, 1, 3)
            .astype(np.float16)
        )  # (N, J, h, 3)
        imu = np.concatenate(
            [data["projected_gravity"], data["base_ang_vel"] * self.time_unit], axis=-1
        )
        out["imu"] = imu[rows, cols].astype(np.float16)
        z_true = data["skill_state"][t, env].astype(np.float32)

        # Compute kinematic anchors, drive loads, and operational-space impedance anchors on device.
        drive = DriveLoads(self.card, self.anchors).to(self.device)
        drive.set_drives(*(np.asarray(data[k]) for k in DRIVES))
        s_map = {f.name: SKILL_LAYOUT.slice_of(f.name) for f in SKILL_LAYOUT.fields}
        q_cur = q[t, env].astype(np.float32)
        vel_cur = vel[t, env].astype(np.float32)
        g_cur = data["projected_gravity"][t, env].astype(np.float32)
        w_cur = data["base_ang_vel"][t, env].astype(np.float32)
        tgt_cur = target[t, env].astype(np.float32)
        cmd_cur = command_w[:, -1, :].astype(np.float32)
        n_rows, n_joints = len(t), len(q0)
        analytic = np.empty((n_rows, SKILL), np.float32)
        loads = np.empty((n_rows, 2), np.float32)
        need_op = not getattr(args, "no_impedance_anchor", False)
        tau_g_arr = np.empty((n_rows, n_joints), np.float16) if need_op else None
        j_eff_arr = np.empty((n_rows, n_joints, 12), np.float16) if need_op else None
        j_pinv_arr = np.empty((n_rows, n_joints, 12), np.float16) if need_op else None
        w_t_arr = np.empty((n_rows, 12), np.float16) if need_op else None
        chunk = 8192 if self.device.type == "cuda" else 4096
        with torch.no_grad():
            for s0 in range(0, n_rows, chunk):
                s1 = min(n_rows, s0 + chunk)
                qt = torch.as_tensor(q_cur[s0:s1], device=self.device)
                qdt = torch.as_tensor(vel_cur[s0:s1], device=self.device)
                gt = torch.as_tensor(g_cur[s0:s1], device=self.device)
                wt = torch.as_tensor(w_cur[s0:s1], device=self.device)
                tt = torch.as_tensor(tgt_cur[s0:s1], device=self.device)
                a = self.anchors(qt, qdt, gt, wt)
                z_a = qt.new_zeros(s1 - s0, SKILL)
                z_a[:, s_map["height"]] = a["height"]
                z_a[:, s_map["gravity"]] = a["gravity"]
                z_a[:, s_map["base_angular_velocity"]] = a["base_angular_velocity"]
                z_a[:, s_map["foot_position"]] = a["foot_position"].reshape(-1, 6)
                z_a[:, s_map["foot_velocity"]] = a["foot_velocity_relative"].reshape(-1, 6)
                z_a[:, s_map["angular_momentum"]] = a["angular_momentum"]
                analytic[s0:s1] = z_a.cpu().numpy()
                loads[s0:s1] = drive(qt, qdt, gt, target=tt)["loads"].cpu().numpy()
                if need_op:
                    cr = torch.as_tensor(cmd_cur[s0:s1], device=self.device)
                    op = drive.operational_anchor_from_ratio(qt, gt, cr)
                    tau_g_arr[s0:s1] = op["tau_g"].cpu().numpy().astype(np.float16)
                    j_eff_arr[s0:s1] = op["j_eff"].cpu().numpy().astype(np.float16)
                    j_pinv_arr[s0:s1] = op["j_pinv"].cpu().numpy().astype(np.float16)
                    w_t_arr[s0:s1] = op["w_t"].cpu().numpy().astype(np.float16)
        weights = np.clip(loads, 0.0, None)
        tot = weights.sum(-1, keepdims=True)
        weights = np.where(tot > 0.05, weights / np.maximum(tot, 1e-9), 0.5)
        rel_v = analytic[:, s_map["foot_velocity"]].reshape(-1, 2, 3)
        odometry = -(weights[..., None] * rel_v).sum(1)
        physics = np.concatenate([odometry, loads], axis=1).astype(np.float32)
        tau_eff_t, k_eff_t, leg_t = drive.operational_params()
        if need_op:
            out["tau_g"] = tau_g_arr
            out["j_eff"] = j_eff_arr
            out["j_pinv"] = j_pinv_arr
            out["w_t"] = w_t_arr

        relabelled = None
        if args.labels:  # another teacher's actions on the same states (pact_relabel.py)
            relabelled = path.with_name(f"{path.stem}.labels_{args.labels}.npz")
        if relabelled is not None:
            if not relabelled.exists():
                raise FileNotFoundError(f"{relabelled}: run pact_relabel.py --name {args.labels}")
            with np.load(relabelled) as labels:
                joints = labels["action_joints"]
                label = np.zeros((len(t), len(q0)), np.float32)
                label[:, joints] = labels["action_scale"] * labels["action_teacher"][t, env]
            controlled = np.zeros(len(q0), bool)
            controlled[joints] = True
        elif "action_joints" in data.files:
            joints = data["action_joints"]
            label = np.zeros((len(t), len(q0)), np.float32)
            label[:, joints] = data["action_scale"] * data["action_teacher"][t, env]
            controlled = np.zeros(len(q0), bool)
            controlled[joints] = True
        else:
            label = target[t + 1, env] - q0
            controlled = target.reshape(-1, target.shape[-1]).std(0) > 1e-6
        controlled = controlled & (drive_gain(gains, effort_cap) > 0)
        out["label"] = label.astype(np.float32)
        out["hold"] = (target[t, env] - q0).astype(np.float32)
        out["controlled"] = controlled
        out["leg_mask"] = leg_t.cpu().numpy() & controlled
        out["tau_eff_dimless"] = tau_eff_t.cpu().numpy().astype(np.float32)
        out["k_eff"] = k_eff_t.cpu().numpy().astype(np.float32)
        # Each sample's drive: the loss compares the torques targets command (drive_loss).
        out["drive_offset"] = drive_offset(q_cur, vel_cur, q0, gains, effort_cap)
        out["drive_gain"] = drive_gain(gains, effort_cap)
        out["anchor_in"] = with_physics(analytic, physics).astype(np.float16)
        out["learned"] = residual_targets(z_true, analytic, physics)
        out["z"] = z_true
        out["start"] = start
        mean, std = self.skill
        obs = data["policy_obs"]
        estimate = data["estimate"]
        view = self._view(obs, estimate, mean, std)
        tail = obs[t, env, obs.shape[-1] - (view * (SKILL + 3) + SKILL) :]
        upcoming, est, path_ahead = plan_from_view(tail, view, mean, std)
        inside = unclamped(estimate[t, env], mean, std)
        error = float(np.abs(est - estimate[t, env])[inside].max()) if inside.any() else 0.0
        if error > 1e-3:
            raise ValueError(f"{path}: decoded estimate differs from the capture's by {error}")
        out["clamped"] = ~inside
        out["plan"] = upcoming.astype(np.float16)
        out["relative"] = (upcoming - est[:, None]).astype(np.float16)
        out["path"] = path_ahead.astype(np.float16)
        out["view"] = view
        dt_star = float(data["control_dt"]) / self.time_unit
        out["times"] = (np.arange(1, view + 1) * dt_star).astype(np.float32)
        out["alpha_dt"] = float(1.0 - math.exp(-dt_star / 0.08))
        out["group"] = env.astype(np.int64)
        if "probe_index" in data.files:
            probe = data["probe_index"][env]
            if not np.array_equal(probe, env % PROBES):
                raise ValueError(f"{path}: environments are not laid out probe by probe")
            out["test"] = probe % 4 == 0
        else:  # a capture in the training task's own plans (pact_dagger.sh, PACT_DAGGER_ENV=train)
            out["test"] = np.zeros(len(env), bool)
        return out

    @staticmethod
    def _view(obs, estimate, mean, std) -> int:
        """The view length whose decoded estimate matches the capture's (rows 20-200, where
        the view is not clamped)."""
        rows = slice(20, 200)
        inside = unclamped(estimate[rows], mean, std)
        for view in range(1, 20):
            width = view * (SKILL + 3) + SKILL
            if width > obs.shape[-1]:
                break
            start = obs.shape[-1] - width + view * SKILL
            zhat = obs[rows, :, start : start + SKILL] * std + mean
            if inside.any() and np.abs(zhat - estimate[rows])[inside].max() < 1e-3:
                return view
        raise ValueError("no plan view decodes to the capture's estimate")


def budget_rows(b: BodyData, probes: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """``b``'s map-capture training rows in its first ``probes`` training probes of one
    seeded permutation (budgets nested; every training probe at 0), and those probes."""
    first = (b.group < 1000) & ~b.test
    probe = b.group % PROBES
    ids = np.unique(probe[first])
    if probes:
        if probes > len(ids):
            raise ValueError(f"{b.name} has {len(ids)} training probes, not {probes}")
        ids = np.random.default_rng(seed).permutation(ids)[:probes]
    return np.nonzero(first & np.isin(probe, ids))[0], sorted(int(i) for i in ids)


def adapt_parameters(
    model: nn.Module, kind: str, arm: str, dims, table, added: int
) -> list[nn.Parameter]:
    """Freeze what ``arm`` does not tune and return what it tunes (G-N3).

    ``table`` is the added body's row of the padded body table. PACT's residual arms wrap
    its embodiment encoder in :class:`~.actor.BodyResidual` (zero at the start).
    """
    if arm not in ADAPT_ARMS[kind]:
        raise ValueError(f"a {kind} model adapts by {ADAPT_ARMS[kind]}, not {arm!r}")
    chosen: list[nn.Parameter] = []
    if kind == "pact" and arm != "zero":
        device = next(model.parameters()).device
        residual = BodyResidual(model.embodiment, table, dims.body_tokens, dims.width)
        model.embodiment = residual.to(device)
        chosen += [model.embodiment.delta_e, model.embodiment.delta_d]
    if arm == "bitfit":
        chosen += bias_and_norm(model)
    if arm == "stem":
        chosen += [*model.proprio[added].parameters(), *model.heads[added].parameters()]
    if arm == "full":
        chosen += list(model.parameters())
    ids = {id(p) for p in chosen}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    return [p for p in model.parameters() if p.requires_grad]


def driver_state(model: nn.Module, joints: int) -> tuple[dict, dict]:
    """``model``'s state as :class:`~.driver.PactDriver` loads it, and the checkpoint's extra
    fields: with a body residual (G-N3's PACT arms), the policy unwrapped and the residual
    beside it on the body's own ``joints`` (G-N2's layout; its padded rows are masked)."""
    state = model.state_dict()
    residual = getattr(model, "embodiment", None)
    if not isinstance(residual, BodyResidual):
        return state, {}
    state = {
        k.replace("embodiment.inner.", "embodiment.", 1): v
        for k, v in state.items()
        if not k.startswith("embodiment.delta_")
    }
    return state, {
        "residual": {
            "delta_e": residual.delta_e.detach().cpu(),
            "delta_d": residual.delta_d.detach().cpu()[:joints],
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default="")
    parser.add_argument("--captures", default="logs/pact/coexec/{body}_map.npz")
    parser.add_argument(
        "--dagger",
        default="",
        help="globs of a body's DAgger captures, ';'-separated, e.g. logs/pact/dagger/r*/{body}.npz "
        "({variant}: the body's final teacher, transfer/bodies/teachers.json)",
    )
    parser.add_argument("--dagger-stride", type=int, default=2)
    parser.add_argument(
        "--min-duty",
        type=float,
        default=0.0,
        help="drop the rows of gait probes asking a mean duty below this (0.55: walking only)",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="start pact_d from this checkpoint and its normalisers",
    )
    parser.add_argument("--contracts", default=CONTRACTS)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th window")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None, help="3e-4 (1.5e-4 on --init); with --adapt, ADAPT_LR")
    parser.add_argument(
        "--adapt",
        default="",
        choices=("", "zero", "etune", "bitfit", "full", "stem"),
        help="G-N3: add the --hold-out body to the --init model, tuning this arm",
    )
    parser.add_argument(
        "--probes", type=int, default=0, help="with --adapt: the body's training probes (0: all)"
    )
    parser.add_argument("--anchor-weight", type=float, default=None)
    parser.add_argument(
        "--no-impedance-anchor",
        action="store_true",
        help="legacy v1 radian decoder without the operational-space impedance anchor",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="train on another teacher's labels: <capture>.labels_<name>.npz (pact_relabel.py)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="train pact_d on each sample or its mirror image, at random (pact/mirror.py)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument(
        "--arms",
        default="hold,pact_d,mlp_d",
        help="of hold,pact_d,pact_na,pact_nb,urma_d,hpt_d,mlp_d",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_gate_d0.json")
    parser.add_argument("--save", type=Path, default=None, help="where to save pact_d")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if init is not None and init.get("kind", "pact") == "pact":
        # Follow the --init checkpoint's architecture (v1 vs v2) automatically.
        if not init.get("dims", {}).get("impedance_anchor", False):
            args.no_impedance_anchor = True
    arms = args.arms.split(",")
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    names = [b for b in args.bodies.split(",") if b]
    bodies = []
    for body in names:
        start = time.time()
        bodies.append(BodyData(body, args, registry, device=device))
        b = bodies[-1]
        print(
            f"{body}: {len(b.label)} samples, view {b.view}, controlled "
            f"{int(b.controlled.sum())}/{len(b.controlled)} ({time.time() - start:.0f} s)",
            flush=True,
        )
    train_index = [i for i, b in enumerate(names) if b != args.hold_out]
    train_rows = {i: np.nonzero(~b.test)[0] for i, b in enumerate(bodies)}

    def pooled(key):
        return np.concatenate([getattr(bodies[i], key)[train_rows[i]].astype(np.float32) for i in train_index])

    added, probes_used = None, None
    if args.adapt:
        if init is None or not args.hold_out or init.get("hold_out") != args.hold_out:
            raise ValueError(
                "--adapt adds the --hold-out body to an --init model trained without it"
            )
        added = names.index(args.hold_out)
        train_index = [added]
        train_rows[added], probes_used = budget_rows(bodies[added], args.probes, args.seed)
        print(f"adding {args.hold_out}: {len(train_rows[added])} samples, probes {probes_used}")
    lr = (
        args.lr
        if args.lr is not None
        else ADAPT_LR[args.adapt]
        if args.adapt
        else (1.5e-4 if init is not None else 3e-4)
    )
    if init is not None:
        z_mean, z_std, a_mean, a_std, t_mean, t_std = (
            np.asarray(init[k]) for k in ("z_mean", "z_std", "a_mean", "a_std", "t_mean", "t_std")
        )
    else:
        z = pooled("z")
        z_mean, z_std = z.mean(0), np.maximum(z.std(0), 1e-3)
        a = pooled("anchor_in")
        a_mean, a_std = a.mean(0), np.maximum(a.std(0), 1e-3)
        t_all = pooled("learned")
        t_mean, t_std = t_all.mean(0), np.maximum(t_all.std(0), 1e-3)
    label_scale = float(
        np.sqrt(
            np.mean([np.mean(bodies[i].label[:, bodies[i].controlled] ** 2) for i in train_index])
        )
    )
    if init is not None:
        label_scale = float(init["label_scale"])
    as_t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)  # noqa: E731
    table = stack_bodies([b.descriptor for b in bodies], [b.contacts for b in bodies]).to(device)
    count = table.mask.shape[1]
    steps_max = max(b.view for b in bodies)
    controlled = torch.zeros(len(bodies), count, dtype=torch.bool, device=device)
    leg_table = torch.zeros(len(bodies), count, dtype=torch.bool, device=device)
    gain_table = torch.zeros(len(bodies), count, device=device)
    teff_table = torch.ones(len(bodies), count, device=device)
    inv_teff_table = torch.zeros(len(bodies), count, device=device)
    keff_table = torch.zeros(len(bodies), count, device=device)
    alpha_table = torch.tensor([b.alpha_dt for b in bodies], dtype=torch.float32, device=device)
    jweight_table = torch.zeros(len(bodies), count, device=device)
    anc_weight_table = torch.zeros(len(bodies), count, device=device)
    for i, b in enumerate(bodies):
        j = len(b.controlled)
        controlled[i, :j] = torch.as_tensor(b.controlled, device=device)
        leg_table[i, :j] = torch.as_tensor(b.leg_mask, device=device)
        gain_table[i, :j] = torch.as_tensor(b.drive_gain, device=device)
        teff_table[i, :j] = torch.as_tensor(b.tau_eff_dimless, device=device)
        inv_teff_table[i, :j] = torch.as_tensor(
            np.where(b.controlled, 1.0 / np.maximum(b.tau_eff_dimless, 1e-6), 0.0), device=device
        )
        keff_table[i, :j] = torch.as_tensor(b.k_eff, device=device)
        tr_i = train_rows[i]
        ctrl = b.controlled
        taught_tr = np.tanh(b.drive_gain * b.label[tr_i] + b.drive_offset[tr_i])
        jvar = taught_tr.var(0)
        jmean = np.abs(taught_tr.mean(0))
        mv = max(float(jvar[ctrl].mean()), 1e-4)
        # Joints permanently saturated against a mechanical joint stop (e.g. Cassie hip_rotation)
        # carry internal stop preload rather than ground wrench: keep at 1x in full BC and exclude
        # from the morphology-blind operational-space anchor supervision bc_anc.
        pinned = jmean >= 0.90
        jw = np.where(ctrl, np.where(pinned, 1.0 / mv, 1.0 / (jvar + 0.1 * mv)), 0.0)
        aw = np.where(b.leg_mask & (~pinned), jw, 0.0)
        jweight_table[i, :j] = torch.as_tensor(jw, dtype=torch.float32, device=device)
        anc_weight_table[i, :j] = torch.as_tensor(aw, dtype=torch.float32, device=device)
    arm_table = controlled & (~leg_table)

    def batch(
        i: int, rows: np.ndarray, augment: bool = False, include_phys: bool = False
    ) -> dict[str, torch.Tensor]:
        b = bodies[i]
        n, j, v = len(rows), b.joint_obs.shape[1], b.view
        raw = {
            "joint_obs": b.joint_obs[rows].astype(np.float32),
            "imu": b.imu[rows].astype(np.float32),
            "anchor_in": b.anchor_in[rows].astype(np.float32),
            "plan": b.plan[rows].astype(np.float32),
            "relative": b.relative[rows].astype(np.float32),
            "path": b.path[rows].astype(np.float32),
            "label": b.label[rows],
            "drive_offset": b.drive_offset[rows],
            "learned": b.learned[rows],
        }
        if include_phys:
            raw["tau_g"] = b.tau_g[rows].astype(np.float32)
            raw["j_eff"] = b.j_eff[rows].astype(np.float32)
            raw["j_pinv"] = b.j_pinv[rows].astype(np.float32)
            raw["w_t"] = b.w_t[rows].astype(np.float32)
        if augment and args.mirror:  # each sample or its mirror image, at random
            flip = rng.random(n) < 0.5
            m = b.mirror
            ops = [
                ("joint_obs", m.joint_obs),
                ("imu", m.imu),
                ("anchor_in", m.skill),
                ("plan", m.skill),
                ("relative", m.skill),
                ("path", m.path),
                ("label", m.drive_positions),
                ("drive_offset", lambda x, m=m, g=b.drive_gain: m.drive_speeds(x) - g * m.drive_offset),
                ("learned", m.learned),
            ]
            if include_phys:
                ops += [
                    ("tau_g", m.speeds),
                    ("j_eff", m.jacobian),
                    ("j_pinv", m.jacobian),
                    ("w_t", m.wrench),
                ]
            for key, reflect in ops:
                raw[key] = raw[key].copy()
                raw[key][flip] = reflect(raw[key][flip])
        obs = np.zeros((n, count, args.history, 3), np.float32)
        obs[:, :j] = raw["joint_obs"]
        plan = np.zeros((n, steps_max, SKILL), np.float32)
        plan[:, :v] = (raw["plan"] - z_mean) / z_std
        rel = np.zeros((n, steps_max, SKILL), np.float32)
        rel[:, :v] = raw["relative"] / z_std
        path = np.zeros((n, steps_max, 3), np.float32)
        path[:, :v] = raw["path"]
        times = np.zeros((n, steps_max), np.float32)
        times[:, :v] = b.times
        mask = np.zeros((n, steps_max), bool)
        mask[:, :v] = True
        label = np.zeros((n, count), np.float32)
        label[:, :j] = raw["label"]
        offset = np.zeros((n, count), np.float32)
        offset[:, :j] = raw["drive_offset"]
        out_d = {
            "joint_obs": as_t(obs),
            "imu": as_t(raw["imu"]),
            "anchor_in": as_t((raw["anchor_in"] - a_mean) / a_std),
            "plan": as_t(plan),
            "plan_relative": as_t(rel),
            "plan_time": as_t(times),
            "plan_path": as_t(path),
            "plan_mask": torch.as_tensor(mask, device=device),
            "label": as_t(label),
            "drive_offset": as_t(offset),
            "learned": as_t((raw["learned"] - t_mean) / t_std),
        }
        if include_phys:
            tg = np.zeros((n, count), np.float32)
            tg[:, :j] = raw["tau_g"]
            je = np.zeros((n, count, 12), np.float32)
            je[:, :j] = raw["j_eff"]
            jp = np.zeros((n, count, 12), np.float32)
            jp[:, :j] = raw["j_pinv"]
            out_d["tau_g"] = as_t(tg)
            out_d["j_eff"] = as_t(je)
            out_d["j_pinv"] = as_t(jp)
            out_d["w_t"] = as_t(raw["w_t"])
        return out_d

    def fit(model: nn.Module, step_fn, label: str, tuned: list | None = None) -> float:
        tuned = list(model.parameters()) if tuned is None else tuned
        if not tuned or args.steps == 0:
            return 0.0
        opt = torch.optim.AdamW(tuned, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        start = time.time()
        for step in range(args.steps):
            loss = step_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tuned, 1.0)
            opt.step()
            sched.step()
            if step % 2000 == 0 or step == args.steps - 1:
                print(
                    f"{label} step {step:5d} loss {loss.item():.5f} ({time.time() - start:.0f} s)"
                )
        return time.time() - start

    def chunks(rows):
        return np.array_split(rows, max(1, len(rows) // 2048))

    predictions: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in arms}
    params: dict[str, int] = {}
    seconds: dict[str, float] = {}
    if "hold" in arms:
        for i, b in enumerate(bodies):
            predictions["hold"][i] = b.hold[b.evaluate]

    shared = [arm for arm in ("pact_d", "pact_na", "pact_nb", "urma_d", "hpt_d") if arm in arms]
    if init is not None and len(shared) != 1:
        raise ValueError(
            "--init continues one model: give one of pact_d, pact_na, pact_nb, urma_d, hpt_d"
        )
    amp_ctx = (
        (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16))
        if device.type == "cuda"
        else contextlib.nullcontext
    )
    for arm in shared:
        use_imp = False
        if arm in ("pact_d", "pact_na", "pact_nb"):
            kind = "pact"
            anchored = arm != "pact_na"
            use_imp = anchored and not args.no_impedance_anchor
            if init is not None and init.get("kind", "pact") == "pact":
                idims = {k: tuple(v) if isinstance(v, list) else v for k, v in init["dims"].items()}
                dims = PactDims(**idims)
                use_imp = bool(dims.impedance_anchor)
            else:
                heads = 8 if args.width >= 256 else 4
                op_w = 384 if args.width >= 256 else 256
                imp_extra = (
                    {"state_anchor": True, "leg_cap": 2.0, "arm_cap": 3.0} if use_imp else {}
                )
                dims = PactDims(
                    observation=3,
                    history=args.history,
                    width=args.width,
                    heads=heads,
                    op_width=op_w,
                    emb_drop=0.10,
                    desc_noise=0.02,
                    anchor_inputs=anchored,
                    timed_plan=True,
                    impedance_anchor=use_imp,
                    **imp_extra,
                    **({"body_tokens": 0} if arm == "pact_nb" else {}),
                )
            model = PactPolicy(dims).to(device)
            default_aw = 0.1 if use_imp else 1.0
            anchor_weight = (args.anchor_weight if args.anchor_weight is not None else default_aw) if anchored else 0.0
        elif arm == "urma_d":
            kind = "urma"
            if init is not None and init.get("kind") == "urma":
                dims = UrmaDims(**init["dims"])
            elif args.width >= 256:
                dims = UrmaDims(
                    history=args.history,
                    plan_steps=steps_max,
                    description=256,
                    joint=512,
                    core=2048,
                    action=512,
                    decoder=1024,
                )
            else:
                dims = UrmaDims(history=args.history, plan_steps=steps_max)
            model = UrmaPolicy(dims).to(device)
            anchor_weight = 0.0  # no anchored fields
        else:
            kind = "hpt"
            if init is not None and init.get("kind") == "hpt":
                idims = {k: tuple(v) if isinstance(v, list) else v for k, v in init["dims"].items()}
                dims = HptDims(**idims)
            elif args.width >= 256:
                dims = HptDims(
                    bodies=tuple(names),
                    joints=tuple(len(b.descriptor.joint_names) for b in bodies),
                    history=args.history,
                    width=384,
                    heads=8,
                    head_hidden=512,
                )
            else:
                dims = HptDims(
                    bodies=tuple(names),
                    joints=tuple(len(b.descriptor.joint_names) for b in bodies),
                    history=args.history,
                )
            model = HptPolicy(dims).to(device)
            anchor_weight = 0.0
        if init is not None:
            if init.get("kind", "pact") != kind:
                raise ValueError(f"--init is a {init.get('kind', 'pact')} model, not {kind}")
            saved = {k: tuple(v) if isinstance(v, list) else v for k, v in init["dims"].items()}
            if type(dims)(**saved) != dims:
                raise ValueError("--init was trained with other dimensions")
            model.load_state_dict(init["state"])
        tuned = None
        if args.adapt:
            one = table.take(torch.tensor([added], device=device))
            tuned = adapt_parameters(model, kind, args.adapt, dims, one, added)
            params[f"{arm}/tuned"] = sum(p.numel() for p in tuned)
        per = max(1, args.batch // len(train_index))

        def routed(which: torch.Tensor, kind: str = kind) -> dict:
            """HPT routes each row to its body's stem and head."""
            return {"body_ids": which} if kind == "hpt" else {}

        def step_shared(
            model=model,
            anchor_weight=anchor_weight,
            routed=routed,
            per=per,
            kind=kind,
            use_imp=use_imp,
        ):
            parts = [
                batch(i, rng.choice(train_rows[i], per), augment=True, include_phys=use_imp)
                for i in train_index
            ]
            x = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            which = torch.cat(
                [torch.full((per,), i, device=device, dtype=torch.long) for i in train_index]
            )
            label, learned, offset = x.pop("label"), x.pop("learned"), x.pop("drive_offset")
            b_batch = table.take(which)
            moved = controlled[which]
            jw_batch = jweight_table[which]
            aw_batch = anc_weight_table[which]
            extra = routed(which)
            if not args.adapt:
                # Copycat-breaking action-history regularization applied uniformly across every
                # shared arm: 50% channel dropout + episode-correlated bias (0.25) + jitter (0.10).
                jo = x["joint_obs"].clone()
                drop = torch.rand(jo.shape[0], 1, 1, device=device) < 0.5
                bias = 0.25 * torch.randn(jo.shape[0], jo.shape[1], 1, device=device)
                jitter = 0.10 * torch.randn_like(jo[:, :, :, 2])
                jo[:, :, :, 2] = (
                    torch.where(
                        drop,
                        torch.zeros_like(jo[:, :, :, 2]),
                        (jo[:, :, :, 2] + bias + jitter).clamp(-0.98, 0.98),
                    )
                    * b_batch.mask[..., None]
                )
                x["joint_obs"] = jo
            if use_imp:
                phys = {
                    "tau_g": x.pop("tau_g"),
                    "j_eff": x.pop("j_eff"),
                    "j_pinv": x.pop("j_pinv"),
                    "w_t": x.pop("w_t"),
                    "k_eff": keff_table[which],
                    "tau_eff_dimless": teff_table[which],
                    "alpha_dt": alpha_table[which],
                }
                extra["phys"] = phys
                extra["return_aux"] = True
            with amp_ctx():
                if kind == "pact" and not isinstance(model.embodiment, BodyResidual):
                    e_all, d_all = model.embodiment(table)
                    extra["emb_cached"] = (e_all[which], d_all[which])
                res = model(b_batch, **x, **extra)
                if use_imp:
                    out_learned, out_tau, aux = res
                    taught_r = torch.tanh(gain_table[which] * label + offset)
                    pred_r = torch.tanh(out_tau * inv_teff_table[which])
                    anc_r = torch.tanh(aux["tau_anchor"] * inv_teff_table[which])
                    bc = (
                        ((pred_r - taught_r).square() * jw_batch).sum(1)
                        / moved.sum(1).clamp_min(1)
                    ).mean() * 0.05
                    leg_m = leg_table[which]
                    arm_m = arm_table[which]
                    bc_anc = (
                        ((anc_r - taught_r).square() * aw_batch).sum(1)
                        / (aw_batch > 0).sum(1).clamp_min(1)
                    ).mean() * 0.05
                    reg_leg = aux["delta_rel"].square()[leg_m].mean() if leg_m.any() else 0.0
                    reg_arm = aux["delta_rel"].square()[arm_m].mean() if arm_m.any() else 0.0
                    reg_op = aux["op_out"].square().mean()
                    loss = bc + 0.5 * bc_anc + 0.001 * reg_leg + 0.001 * reg_arm + 2e-4 * reg_op
                    if anchor_weight > 0.0:
                        loss = loss + anchor_weight * (out_learned - learned).square().mean()
                    return loss
                out_learned, out = res
                taught_r = torch.tanh(gain_table[which] * label + offset)
                pred_r = torch.tanh(gain_table[which] * out + offset)
                bc = (
                    ((pred_r - taught_r).square() * jw_batch).sum(1)
                    / moved.sum(1).clamp_min(1)
                ).mean() * 0.05
                if anchor_weight == 0.0:
                    return bc
                return bc + anchor_weight * (out_learned - learned).square().mean()

        seconds[arm] = fit(model, step_shared, arm, tuned)
        model.eval()
        params[arm] = sum(p.numel() for p in model.parameters())
        for i, b in enumerate(bodies):
            if args.adapt and i != added:
                continue  # G-N3 reads the added body
            if kind == "hpt" and i not in train_index:
                continue  # a body HPT was not trained on has an untrained stem and head
            out = []
            with torch.no_grad():
                e_all, d_all = (
                    model.embodiment(table)
                    if kind == "pact" and not isinstance(model.embodiment, BodyResidual)
                    else (None, None)
                )
                for rows in chunks(np.nonzero(b.evaluate)[0]):
                    x = batch(i, rows, include_phys=use_imp)
                    x.pop("label"), x.pop("learned")
                    off = x.pop("drive_offset")
                    which = torch.full((len(rows),), i, device=device, dtype=torch.long)
                    extra = routed(which)
                    if e_all is not None:
                        extra["emb_cached"] = (e_all[which], d_all[which])
                    if use_imp:
                        extra["phys"] = {
                            "tau_g": x.pop("tau_g"),
                            "j_eff": x.pop("j_eff"),
                            "j_pinv": x.pop("j_pinv"),
                            "w_t": x.pop("w_t"),
                            "k_eff": keff_table[which],
                            "tau_eff_dimless": teff_table[which],
                            "alpha_dt": alpha_table[which],
                        }
                    with amp_ctx():
                        pred = model(table.take(which), **x, **extra)[1]
                    if use_imp:
                        # Exact analytic PD target inversion (same as DriveLoads.targets_from_torque)
                        u_cmd = (pred.float() * inv_teff_table[which]).clamp(-3.0, 3.0)
                        g_safe = gain_table[which].clamp_min(1e-6)
                        targets = torch.where(controlled[which], (u_cmd - off) / g_safe, 0.0)
                    else:
                        targets = pred.float()
                    out.append(targets[:, : len(b.controlled)].cpu().numpy())
            predictions[arm][i] = np.concatenate(out)
        prefix = PREFIX[arm]
        name = f"{prefix}_{args.hold_out or 'all'}"
        if args.adapt:
            name = f"{prefix}_add_{args.hold_out}_{args.adapt}_k{args.probes or 'all'}"
        save = (args.save if len(shared) == 1 else None) or REPO / (
            f"logs/pact/models/{name}_s{args.seed}_p{args.steps}.pt"
        )
        joints = len(bodies[added].descriptor.joint_names) if args.adapt else 0
        state, extra = driver_state(model, joints)
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": kind,
                "state": state,
                **extra,
                "dims": vars(dims),
                "bodies": names,
                "hold_out": args.hold_out,
                "z_mean": z_mean,
                "z_std": z_std,
                "a_mean": a_mean,
                "a_std": a_std,
                "t_mean": t_mean,
                "t_std": t_std,
                "label_scale": label_scale,
                "views": {b.name: b.view for b in bodies},
                "added": {"body": args.hold_out, "arm": args.adapt, "probes": probes_used}
                if args.adapt
                else None,
            },
            save,
        )
        print(f"saved {save}")

    if "mlp_d" in arms:
        for i, b in enumerate(bodies):
            if args.adapt and i != added:
                continue
            j, v = b.joint_obs.shape[1], b.view
            inputs = j * args.history * 3 + args.history * 6 + SKILL + v * (2 * SKILL + 3)
            moved = np.nonzero(b.controlled)[0]
            model = nn.Sequential(
                nn.Linear(inputs, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(),
                nn.Linear(512, len(moved)),
            ).to(device)  # fmt: skip

            def features(rows, b=b):
                return torch.cat(
                    [
                        as_t(b.joint_obs[rows]).flatten(1),
                        as_t(b.imu[rows]).flatten(1),
                        as_t((b.anchor_in[rows] - a_mean) / a_std),
                        as_t((b.plan[rows] - z_mean) / z_std).flatten(1),
                        as_t(b.relative[rows] / z_std).flatten(1),
                        as_t(b.path[rows]).flatten(1),
                    ],
                    dim=1,
                )

            def step_mlp(i=i, b=b, model=model, features=features, moved=moved):
                rows = rng.choice(train_rows[i], max(1, args.batch // len(train_index)))
                target = as_t(b.label[rows][:, moved])
                gain, offset = as_t(b.drive_gain[moved])[None], as_t(b.drive_offset[rows][:, moved])
                out = model(features(rows))
                return drive_loss(out, target, gain, offset, torch.ones_like(target, dtype=torch.bool))

            seconds[f"mlp_d/{b.name}"] = fit(model, step_mlp, f"mlp_d/{b.name}")
            model.eval()
            params[f"mlp_d/{b.name}"] = sum(p.numel() for p in model.parameters())
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(b.evaluate)[0]):
                    full = np.zeros((len(rows), len(b.controlled)), np.float32)
                    full[:, moved] = model(features(rows)).cpu().numpy()
                    out.append(full)
            predictions["mlp_d"][i] = np.concatenate(out)

    report = {
        "gate": "G-N3" if args.adapt else "G-D0",
        "added": {
            "body": args.hold_out,
            "arm": args.adapt,
            "probes": probes_used,
            "train_samples": int(len(train_rows[added])),
        }
        if args.adapt
        else None,
        "config": {k: str(v) for k, v in vars(args).items()},
        "parameters": params,
        "train_seconds": {k: round(v, 1) for k, v in seconds.items()},
        "label_scale_rad": label_scale,
        "bodies": {},
    }
    for i, b in enumerate(bodies):
        if args.adapt and i != added:
            continue
        moved = b.controlled
        truth = b.label[b.evaluate][:, moved]
        groups = b.group[b.evaluate]
        variance = float(truth.var(0).mean())
        # The metric is the loss's: commanded torque as a fraction of each joint's limit, softly
        # saturated (drive_ratio). Radians are reported beside it.
        gain, offset = b.drive_gain[moved], b.drive_offset[b.evaluate][:, moved]
        taught = np.tanh(gain * truth + offset)
        variance_torque = float(taught.var(0).mean())
        if variance_torque <= 1e-12:
            raise ValueError(
                f"{b.name}: zero label torque variance ({variance_torque}); check effort cap and controlled joints"
            )
        entry: dict = {
            "held_out": b.name == args.hold_out,
            "view": b.view,
            "controlled_joints": int(moved.sum()),
            "test_samples": int(b.evaluate.sum()),
            "captures": dict(zip(b.captures, b.capture_samples, strict=True)),
            "label_variance_rad2": variance,
            "label_variance_torque": variance_torque,
        }
        sq = {
            arm: ((np.tanh(gain * p[i][:, moved] + offset) - taught) ** 2).mean(1)
            for arm, p in predictions.items()
            if i in p
        }
        sq_rad = {
            arm: ((p[i][:, moved] - truth) ** 2).mean(1) for arm, p in predictions.items() if i in p
        }
        for arm, err in sq.items():
            rmse, se, _, _ = bootstrap_rmse(err, None, groups, args.bootstrap, args.seed)
            rmse_rad, se_rad, _, _ = bootstrap_rmse(sq_rad[arm], None, groups, args.bootstrap, args.seed)
            entry[arm] = {
                "rmse": rmse, "se": se, "r2": 1.0 - float(err.mean()) / variance_torque,
                "rmse_rad": rmse_rad, "se_rad": se_rad,
                "r2_rad": 1.0 - float(sq_rad[arm].mean()) / variance,
            }  # fmt: skip
        for a_arm, b_arm in (
            ("pact_d", "mlp_d"),
            ("pact_d", "hold"),
            ("mlp_d", "hold"),
            ("urma_d", "pact_d"),
            ("pact_na", "pact_d"),
            ("pact_nb", "pact_d"),
            ("urma_d", "pact_na"),
            ("hpt_d", "pact_d"),
            ("hpt_d", "mlp_d"),
            ("urma_d", "mlp_d"),
            ("urma_d", "hold"),
        ):
            if a_arm in sq and b_arm in sq:
                _, _, diff, se = bootstrap_rmse(
                    sq[a_arm], sq[b_arm], groups, args.bootstrap, args.seed
                )
                entry[f"{a_arm}_minus_{b_arm}"] = diff
                entry[f"{a_arm}_minus_{b_arm}_se"] = se
        report["bodies"][b.name] = entry
        print(
            b.name,
            "(held out)" if entry["held_out"] else "",
            {arm: (round(entry[arm]["rmse"], 4), round(entry[arm]["r2"], 3)) for arm in sq},
            flush=True,
        )
    if "pact_d" in arms and not args.adapt:
        r2 = {name: report["bodies"][name]["pact_d"]["r2"] for name in names}
        report["proceed"] = bool(min(r2.values()) >= 0.9)
        print(f"G-D0: pact_d R² {r2}; proceed to G-D1: {report['proceed']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
