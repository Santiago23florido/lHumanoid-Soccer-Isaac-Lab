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
    ) -> None:
        kind = checkpoint.get("kind", "pact")
        if kind == "urma":  # the end-to-end baseline: same interface, anchors ignored
            from humanoid_transfer.transfer.latent.pact.urma import UrmaDims, UrmaPolicy

            dims = UrmaDims(**checkpoint["dims"])
            self.model = UrmaPolicy(dims).to(device).eval()
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
        self.anchors, self.drive = anchors.to(device), drive.to(device)
        self.q0 = default_joint_pos.to(device)
        self.action_joints = action_joints
        self.action_scale = action_scale
        self.view = view
        self.skill_mean, self.skill_std = skill_mean.to(device), skill_std.to(device)
        self.time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
        self.times = (torch.arange(1, view + 1, device=device) * control_dt / self.time_unit)[None]
        joints, history = len(descriptor.joint_names), dims.history
        self.joint_history = torch.zeros(num_envs, joints, history, 3, device=device)
        self.imu_history = torch.zeros(num_envs, history, 6, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

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
        joint_now = torch.stack(
            [q - self.q0, qd * self.time_unit, target - self.q0], dim=-1
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
        mask = torch.ones(n, self.view, dtype=torch.bool, device=self.device)
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
        )
        return out[:, self.action_joints] / self.action_scale
