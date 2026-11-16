"""The robot's estimator in torch, for thousands of simulated robots at once.

A transcription of :mod:`humanoid_transfer.nao.deploy.estimator` (and the
kinematics and ``z*`` it uses) operation for operation: the robot runs the NumPy
version, training runs this one on the GPU. ``tests/test_nao_estimator_torch.py``
holds the two to the same numbers on random sequences, resets included.
"""

from __future__ import annotations

import torch

from ..deploy.kinematics import BodyModel
from ..deploy.skill import LOADED


class TorchBody:
    """:class:`~...deploy.kinematics.BodyModel`'s arrays as tensors."""

    def __init__(self, model: BodyModel, device, dtype=torch.float32) -> None:
        def t(x):
            return torch.as_tensor(x, device=device, dtype=dtype)

        self.model = model
        self.count = len(model.names)
        self.parent = model.parent.tolist()
        self.source = model.source.tolist()
        self.order = list(model.order)
        self.base = model.base
        self.feet = list(model.feet)
        self.origin_rot = t(model.origin_rot)
        self.origin_pos = t(model.origin_pos)
        axis = t(model.axis)
        zero = torch.zeros_like(axis[:, 0])
        k = torch.stack(
            [
                torch.stack([zero, -axis[:, 2], axis[:, 1]], -1),
                torch.stack([axis[:, 2], zero, -axis[:, 0]], -1),
                torch.stack([-axis[:, 1], axis[:, 0], zero], -1),
            ],
            -2,
        )
        self.k, self.k2 = k, k @ k
        self.mass = t(model.mass)
        self.total_mass = float(model.mass.sum())
        self.com = t(model.com)
        self.inertia = t(model.inertia)
        self.sole_from_com = t(model.sole_from_com)
        self.corners = t(model.corners)
        self.scales = t(model.scales)
        self.force_scale = model.force_scale
        self.ground_height = model.ground_height
        self.fields = list(model.fields)
        self.eye = torch.eye(3, device=device, dtype=dtype)

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = q.shape[0]
        rot = [None] * self.count
        pos = [None] * self.count
        rot[self.base] = self.eye.expand(n, 3, 3)
        pos[self.base] = torch.zeros(n, 3, device=q.device, dtype=q.dtype)
        for b in self.order:
            p = self.parent[b]
            local = self.origin_rot[b].expand(n, 3, 3)
            if self.source[b] >= 0:
                angle = q[:, self.source[b]][:, None, None]
                turn = self.eye + torch.sin(angle) * self.k[b] + (1 - torch.cos(angle)) * self.k2[b]
                local = local @ turn
            rot[b] = rot[p] @ local
            pos[b] = pos[p] + torch.einsum("nij,j->ni", rot[p], self.origin_pos[b])
        return torch.stack(rot, 1), torch.stack(pos, 1)

    def centres(self, rot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return pos + torch.einsum("nbij,bj->nbi", rot, self.com)


def _heading(base_rot: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(base_rot[..., 1, 0], base_rot[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, -s, zero], -1),
            torch.stack([s, c, zero], -1),
            torch.stack([zero, zero, one], -1),
        ],
        -2,
    )


def _rotate_inverse(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ji,...j->...i", rotation, vectors)


def sole_points(body: TorchBody, rot: torch.Tensor, com: torch.Tensor) -> torch.Tensor:
    f = body.feet
    return com[:, f] + torch.einsum("nfij,fj->nfi", rot[:, f], body.sole_from_com)


def skill_state(body: TorchBody, rot, com, com_vel, ang_vel, foot_force) -> torch.Tensor:
    masses = body.mass
    c = torch.einsum("b,nbi->ni", masses, com) / body.total_mass
    c_dot = torch.einsum("b,nbi->ni", masses, com_vel) / body.total_mass
    relative_pos = com - c[:, None]
    relative_momentum = masses[:, None] * (com_vel - c_dot[:, None])
    orbital = torch.linalg.cross(relative_pos, relative_momentum, dim=-1).sum(dim=1)
    inertia = rot @ body.inertia @ rot.transpose(-1, -2)
    spin = torch.einsum("nbij,nbj->ni", inertia, ang_vel)
    momentum = orbital + spin
    base_rot = rot[:, body.base]
    heading = _heading(base_rot)
    down = torch.zeros_like(c)
    down[:, 2] = -1.0
    f = body.feet
    offset = torch.einsum("nfij,fj->nfi", rot[:, f], body.sole_from_com)
    sole_pos = com[:, f] + offset
    sole_vel = com_vel[:, f] + torch.linalg.cross(ang_vel[:, f], offset, dim=-1)
    n = c.shape[0]
    values = {
        "height": (c[:, 2] - body.ground_height)[:, None],
        "com_velocity": _rotate_inverse(heading, c_dot),
        "gravity": _rotate_inverse(base_rot, down),
        "base_angular_velocity": _rotate_inverse(base_rot, ang_vel[:, body.base]),
        "foot_position": _rotate_inverse(heading[:, None], sole_pos - c[:, None]).reshape(n, 6),
        "foot_velocity": _rotate_inverse(heading[:, None], sole_vel).reshape(n, 6),
        "foot_load": foot_force[..., 2],
        "angular_momentum": _rotate_inverse(heading, momentum),
    }
    si = torch.cat([values[name].reshape(n, size) for name, size in body.fields], dim=-1)
    return si / body.scales


def stance_velocity(
    body: TorchBody, rot, com, com_vel, ang_vel, foot_load, pivot: str = "sole", previous_load=None
):
    f = body.feet
    if pivot == "sole":
        offset = sole_points(body, rot, com) - com[:, f]
        point_vel = com_vel[:, f] + torch.linalg.cross(ang_vel[:, f], offset, dim=-1)
    elif pivot == "corner":
        points = com[:, f, None] + torch.einsum("nfij,fkj->nfki", rot[:, f], body.corners)
        arm = points - com[:, f, None]
        corner_vel = com_vel[:, f, None] + torch.linalg.cross(
            ang_vel[:, f, None].expand_as(arm), arm, dim=-1
        )
        from ..deploy.skill import CORNER_SOFTNESS

        height = points[..., 2]
        weight = torch.exp(-(height - height.min(dim=-1, keepdim=True).values) / CORNER_SOFTNESS)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        point_vel = (weight[..., None] * corner_vel).sum(dim=-2)
    else:
        raise ValueError(f"unknown pivot {pivot!r}")
    if previous_load is None:
        stance = foot_load.argmax(dim=-1)
    else:
        loaded = (foot_load > LOADED) & (previous_load > LOADED)
        speed = torch.where(loaded, point_vel.norm(dim=-1), torch.inf)
        stance = torch.where(loaded.any(dim=-1), speed.argmin(dim=-1), previous_load.argmax(dim=-1))
    return point_vel[torch.arange(len(stance), device=stance.device), stance]


def proprioceptive_skill_state(
    body: TorchBody, rot, com, com_vel, ang_vel, foot_load, pivot="sole", previous_load=None
):
    f = body.feet
    points = com[:, f, None] + torch.einsum("nfij,fkj->nfki", rot[:, f], body.corners)
    ground = points[..., 2].flatten(1).min(dim=1).values
    lifted = com.clone()
    lifted[..., 2] = com[..., 2] - ground[:, None] + body.ground_height
    still = stance_velocity(body, rot, com, com_vel, ang_vel, foot_load, pivot, previous_load)
    velocity = com_vel - still[:, None]
    force = torch.zeros(*foot_load.shape, 3, device=foot_load.device, dtype=foot_load.dtype)
    force[..., 2] = foot_load * body.force_scale
    return skill_state(body, rot, lifted, velocity, ang_vel, force)


def tilt_from_gravity(gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g = gravity / gravity.norm(dim=-1, keepdim=True)
    pitch = torch.asin(g[..., 0].clamp(-1.0, 1.0))
    roll = torch.atan2(-g[..., 1], -g[..., 2])
    return roll, pitch


def rpy_matrix(roll, pitch, yaw) -> torch.Tensor:
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
            torch.stack([-sp, cp * sr, cp * cr], -1),
        ],
        -2,
    )


def rotation_rate(now: torch.Tensor, before: torch.Tensor, dt: float) -> torch.Tensor:
    w = now @ before.transpose(-1, -2)
    skew = 0.5 * (w - w.transpose(-1, -2))
    return torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1) / dt


class TorchEstimator:
    """:class:`~...deploy.estimator.Estimator` for ``count`` robots on ``device``."""

    def __init__(self, model: BodyModel, count: int, dt: float, device, pivot: str = "corner",
                 dtype=torch.float32) -> None:  # fmt: skip
        self.body = TorchBody(model, device, dtype)
        self.count, self.dt, self.pivot = count, float(dt), pivot
        bodies = self.body.count
        self.yaw = torch.zeros(count, device=device, dtype=dtype)
        self.rot = self.body.eye.expand(count, bodies, 3, 3).clone()
        self.com = torch.zeros(count, bodies, 3, device=device, dtype=dtype)
        self.load = torch.zeros(count, 2, device=device, dtype=dtype)
        self.fresh = torch.ones(count, dtype=torch.bool, device=device)
        self.position = torch.zeros(count, 2, device=device, dtype=dtype)

    def reset(self, which: torch.Tensor | None = None) -> None:
        if which is None:
            self.fresh[:] = True
        else:
            self.fresh |= which

    @torch.no_grad()
    def update(self, q, gravity, gyro, foot_load) -> torch.Tensor:
        body, fresh = self.body, self.fresh
        roll, pitch = tilt_from_gravity(gravity)
        rate = (torch.sin(roll) * gyro[:, 1] + torch.cos(roll) * gyro[:, 2]) / torch.cos(pitch)
        yaw = torch.where(fresh, 0.0, self.yaw + self.dt * rate)
        base = rpy_matrix(roll, pitch, yaw)
        rot_b, pos_b = body.forward(q)
        rot = base[:, None] @ rot_b
        com = torch.einsum("nij,nbj->nbi", base, body.centres(rot_b, pos_b))
        velocity = torch.where(fresh[:, None, None], 0.0, (com - self.com) / self.dt)
        spin = torch.where(fresh[:, None, None], 0.0, rotation_rate(rot, self.rot, self.dt))
        spin[:, body.base] = torch.einsum("nij,nj->ni", base, gyro)
        previous = torch.where(fresh[:, None], foot_load, self.load)
        z = proprioceptive_skill_state(
            body, rot, com, velocity, spin, foot_load, self.pivot, previous
        )
        still = stance_velocity(body, rot, com, velocity, spin, foot_load, self.pivot, previous)
        centre = torch.einsum("b,nbi->ni", body.mass, com) / body.total_mass
        moving = torch.einsum("b,nbi->ni", body.mass, velocity) / body.total_mass - still
        self.position = torch.where(
            fresh[:, None], centre[:, :2], self.position + self.dt * moving[:, :2]
        )
        self.yaw, self.rot, self.com, self.load = yaw, rot, com, foot_load.clone()
        self.fresh = torch.zeros_like(fresh)
        return z


__all__ = ["TorchBody", "TorchEstimator"]
