"""The skill state ``z*`` in torch, so a simulator can compute it on the GPU every step.

:mod:`~humanoid_transfer.common.skill_state` is the definition: NumPy,
certified against the simulator, and the only place the fields are declared.
This module is a transcription of :func:`~...skill_state.skill_values_si`
and :meth:`~...skill_state.ZLayout.pack` into torch, for the one consumer
that cannot leave the GPU: an environment that rewards a body for the state
it reaches. ``tests/test_skill_torch.py`` holds it to the NumPy encoder on
random multibody states and on recorded simulator states, so the two cannot
drift apart silently.

The body is described once, by :class:`BodyConstants` -- masses, inertias,
which link is the base, which are the feet and where their soles are -- and
the per-step state is what Isaac Lab reports: link orientations, and each
body's centre of mass, its velocity and its angular velocity, all in the
world frame, plus the net contact force on each foot.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...common.similarity import UnitSystem
from ...common.skill_state import SKILL_LAYOUT, ZLayout


@dataclass
class BodyConstants:
    """What does not change while a body moves."""

    masses: torch.Tensor
    """``(B,)``, kg."""

    inertia_local: torch.Tensor
    """``(B, 3, 3)``, about each body's CoM, in its link frame."""

    base_index: int
    foot_index: tuple[int, int]
    sole_from_com: torch.Tensor
    """``(2, 3)``, from each foot's CoM to its sole point, in the foot link frame."""

    units: UnitSystem
    ground_height: float = 0.0

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> BodyConstants:
        return BodyConstants(
            masses=self.masses.to(device=device, dtype=dtype),
            inertia_local=self.inertia_local.to(device=device, dtype=dtype),
            base_index=self.base_index,
            foot_index=self.foot_index,
            sole_from_com=self.sole_from_com.to(device=device, dtype=dtype),
            units=self.units,
            ground_height=self.ground_height,
        )

    def scales(self, layout: ZLayout = SKILL_LAYOUT) -> torch.Tensor:
        """``(size,)``: the unit each entry of ``z*`` is divided by."""
        values = []
        for field in layout.fields:
            values += [self.units.scale(field.dimension)] * field.size
        return torch.tensor(values, dtype=self.masses.dtype, device=self.masses.device)


def quat_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """``(..., 4) -> (..., 3, 3)``, w-x-y-z, normalised first, as in :mod:`~...common.frames`."""
    q = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _heading(base_rot: torch.Tensor) -> torch.Tensor:
    """Rotation about the vertical by the base's yaw, ``(..., 3, 3)``."""
    yaw = torch.atan2(base_rot[..., 1, 0], base_rot[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, -s, zero], dim=-1),
            torch.stack([s, c, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )


def _rotate_inverse(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Transpose of ``rotation`` applied to ``vectors``: world vectors into the rotated frame."""
    return torch.einsum("...ji,...j->...i", rotation, vectors)


def skill_state(
    body: BodyConstants,
    link_quat: torch.Tensor,
    com_pos: torch.Tensor,
    com_lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    foot_force: torch.Tensor,
    layout: ZLayout = SKILL_LAYOUT,
) -> torch.Tensor:
    """``z*`` in the body's own units, ``(N, size)``.

    Args:
        link_quat: ``(N, B, 4)`` world-from-link orientations, w-x-y-z
            (Isaac Lab ``body_link_quat_w``).
        com_pos: ``(N, B, 3)`` each body's centre of mass, relative to the env origin.
        com_lin_vel: ``(N, B, 3)`` velocity at each body's centre of mass.
        ang_vel: ``(N, B, 3)`` each body's angular velocity.
        foot_force: ``(N, 2, 3)`` net contact force on the left and right foot.
    """
    masses = body.masses
    total = masses.sum()
    link_rot = quat_to_matrix(link_quat)
    c = torch.einsum("b,nbi->ni", masses, com_pos) / total
    c_dot = torch.einsum("b,nbi->ni", masses, com_lin_vel) / total

    relative_pos = com_pos - c[:, None, :]
    relative_momentum = masses[None, :, None] * (com_lin_vel - c_dot[:, None, :])
    orbital = torch.linalg.cross(relative_pos, relative_momentum, dim=-1).sum(dim=1)
    inertia = link_rot @ body.inertia_local @ link_rot.transpose(-1, -2)
    spin = torch.einsum("nbij,nbj->ni", inertia, ang_vel)
    momentum = orbital + spin

    base_rot = link_rot[:, body.base_index]
    heading = _heading(base_rot)
    down = torch.zeros_like(c)
    down[:, 2] = -1.0

    feet = list(body.foot_index)
    foot_rot = link_rot[:, feet]
    offset = torch.einsum("nfij,fj->nfi", foot_rot, body.sole_from_com)
    sole_pos = com_pos[:, feet] + offset
    sole_vel = com_lin_vel[:, feet] + torch.linalg.cross(ang_vel[:, feet], offset, dim=-1)

    values = {
        "height": (c[:, 2] - body.ground_height)[:, None],
        "com_velocity": _rotate_inverse(heading, c_dot),
        "gravity": _rotate_inverse(base_rot, down),
        "base_angular_velocity": _rotate_inverse(base_rot, ang_vel[:, body.base_index]),
        "foot_position": _rotate_inverse(heading[:, None], sole_pos - c[:, None, :]).flatten(1),
        "foot_velocity": _rotate_inverse(heading[:, None], sole_vel).flatten(1),
        "foot_load": foot_force[..., 2],
        "angular_momentum": _rotate_inverse(heading, momentum),
    }
    si = torch.cat([values[field.name] for field in layout.fields], dim=-1)
    return si / body.scales(layout)


def sole_points(
    body: BodyConstants, link_quat: torch.Tensor, com_pos: torch.Tensor
) -> torch.Tensor:
    """``(N, 2, 3)`` each foot's sole point, in the frame ``com_pos`` is given in."""
    feet = list(body.foot_index)
    foot_rot = quat_to_matrix(link_quat[:, feet])
    return com_pos[:, feet] + torch.einsum("nfij,fj->nfi", foot_rot, body.sole_from_com)


def proprioceptive_skill_state(
    body: BodyConstants,
    link_quat: torch.Tensor,
    com_pos: torch.Tensor,
    com_lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    foot_load: torch.Tensor,
    base_velocity: bool = True,
    layout: ZLayout = SKILL_LAYOUT,
) -> torch.Tensor:
    """``z*`` as a body can compute it from its own sensors and kinematic model.

    Everything in ``z*`` but the feet's loads is kinematics: the links' poses and
    velocities relative to the base follow from the joint encoders through the
    body's model, the base's tilt and turn rate from its IMU, and ``z*`` never
    depends on where the body is or which way it faces. What a body cannot sense
    is replaced by what it can infer:

    - the ground is under the lower sole, so the height is the CoM's above it;
    - without a base velocity (``base_velocity=False``, as on the NAO) the more
      loaded foot is taken to stand still, and every velocity is measured from it
      (leg odometry) -- the angular momentum, about the CoM, does not change;
    - the loads, ``(N, 2)`` in units of ``mg``, come from elsewhere (a learned
      estimate).

    The arguments are those of :func:`skill_state`, in any world frame.
    """
    soles = sole_points(body, link_quat, com_pos)
    ground = soles[..., 2].min(dim=1).values
    lifted = com_pos.clone()
    lifted[..., 2] = com_pos[..., 2] - ground[:, None] + body.ground_height
    velocity = com_lin_vel
    if not base_velocity:
        feet = list(body.foot_index)
        offset = soles - com_pos[:, feet]
        sole_vel = com_lin_vel[:, feet] + torch.linalg.cross(ang_vel[:, feet], offset, dim=-1)
        stance = foot_load.argmax(dim=-1)
        velocity = com_lin_vel - sole_vel[torch.arange(len(stance)), stance][:, None, :]
    force = torch.zeros(*foot_load.shape, 3, dtype=com_pos.dtype, device=com_pos.device)
    force[..., 2] = foot_load * body.units.scale(_FORCE)
    return skill_state(body, link_quat, lifted, velocity, ang_vel, force, layout)


_FORCE = next(field.dimension for field in SKILL_LAYOUT.fields if field.name == "foot_load")

__all__ = [
    "BodyConstants",
    "proprioceptive_skill_state",
    "quat_to_matrix",
    "skill_state",
    "sole_points",
]
