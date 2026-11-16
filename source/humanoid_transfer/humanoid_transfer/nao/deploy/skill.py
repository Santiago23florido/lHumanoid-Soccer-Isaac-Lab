"""``z*`` computed from the robot's own sensors and kinematic model, in NumPy.

A transcription of :func:`humanoid_transfer.transfer.latent.skill_torch.skill_state`
and :func:`~humanoid_transfer.transfer.latent.skill_torch.proprioceptive_skill_state`
for the robot, which has neither torch nor the extension.
``tests/test_nao_deploy.py`` holds the two to the same numbers.

The fields and their order come with the body model (:attr:`BodyModel.fields`), so a change of
layout reaches the robot with the model and cannot be missed here.
"""

from __future__ import annotations

import numpy as np

from .kinematics import BodyModel


def _heading(base_rot: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(base_rot[..., 1, 0], base_rot[..., 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.stack(
        [np.stack([c, -s, zero], -1), np.stack([s, c, zero], -1), np.stack([zero, zero, one], -1)],
        axis=-2,
    )


def _rotate_inverse(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.einsum("...ji,...j->...i", rotation, vectors)


def sole_points(model: BodyModel, rot: np.ndarray, com: np.ndarray) -> np.ndarray:
    """``(..., 2, 3)`` each foot's sole point, in the frame ``rot`` and ``com`` are in."""
    feet = list(model.feet)
    return com[..., feet, :] + np.einsum(
        "...fij,fj->...fi", rot[..., feet, :, :], model.sole_from_com
    )


def skill_state(
    model: BodyModel,
    rot: np.ndarray,
    com: np.ndarray,
    com_vel: np.ndarray,
    ang_vel: np.ndarray,
    foot_force: np.ndarray,
) -> np.ndarray:
    """``z*`` in the body's own units ``(..., size)``.

    ``rot`` ``(..., B, 3, 3)`` world-from-link rotations, ``com``, ``com_vel`` and ``ang_vel``
    ``(..., B, 3)`` each body's centre of mass, its velocity and angular velocity, and
    ``foot_force`` ``(..., 2, 3)`` the net force on each foot, all in one world frame.
    """
    masses = model.mass
    total = masses.sum()
    c = np.einsum("b,...bi->...i", masses, com) / total
    c_dot = np.einsum("b,...bi->...i", masses, com_vel) / total
    relative_pos = com - c[..., None, :]
    relative_momentum = masses[:, None] * (com_vel - c_dot[..., None, :])
    orbital = np.cross(relative_pos, relative_momentum).sum(axis=-2)
    inertia = rot @ model.inertia @ np.swapaxes(rot, -1, -2)
    spin = np.einsum("...bij,...bj->...i", inertia, ang_vel)
    momentum = orbital + spin
    base_rot = rot[..., model.base, :, :]
    heading = _heading(base_rot)
    down = np.zeros_like(c)
    down[..., 2] = -1.0
    feet = list(model.feet)
    offset = np.einsum("...fij,fj->...fi", rot[..., feet, :, :], model.sole_from_com)
    sole_pos = com[..., feet, :] + offset
    sole_vel = com_vel[..., feet, :] + np.cross(ang_vel[..., feet, :], offset)
    lead = c.shape[:-1]
    values = {
        "height": (c[..., 2] - model.ground_height)[..., None],
        "com_velocity": _rotate_inverse(heading, c_dot),
        "gravity": _rotate_inverse(base_rot, down),
        "base_angular_velocity": _rotate_inverse(base_rot, ang_vel[..., model.base, :]),
        "foot_position": _rotate_inverse(
            heading[..., None, :, :], sole_pos - c[..., None, :]
        ).reshape(*lead, 6),
        "foot_velocity": _rotate_inverse(heading[..., None, :, :], sole_vel).reshape(*lead, 6),
        "foot_load": foot_force[..., 2],
        "angular_momentum": _rotate_inverse(heading, momentum),
    }
    si = np.concatenate([values[name].reshape(*lead, size) for name, size in model.fields], axis=-1)
    return si / model.scales


PIVOTS = ("sole", "corner")
CORNER_SOFTNESS = 0.002
"""How much higher than the lowest corner another still counts as touching, metres."""
LOADED = 0.25
"""A foot carrying more than this share of the weight is on the ground."""


def _corners(model: BodyModel, rot: np.ndarray, com: np.ndarray) -> np.ndarray:
    """``(..., 2, K, 3)`` each sole's corners, in the frame ``rot`` and ``com`` are in."""
    feet = list(model.feet)
    return com[..., feet, None, :] + np.einsum(
        "...fij,fkj->...fki", rot[..., feet, :, :], model.corners
    )


def stance_velocity(
    model: BodyModel,
    rot: np.ndarray,
    com: np.ndarray,
    com_vel: np.ndarray,
    ang_vel: np.ndarray,
    foot_load: np.ndarray,
    pivot: str = "sole",
    previous_load: np.ndarray | None = None,
) -> np.ndarray:
    """``(..., 3)`` the velocity of the point of a stance foot taken to stand still.

    Which foot: without ``previous_load``, the more loaded one, as
    :func:`~...skill_torch.proprioceptive_skill_state`. With the loads at the start of
    the step the velocities were differenced over, one loaded (:data:`LOADED`) at both
    ends of the step, the slower of two -- a foot that landed during the step was still
    swinging for part of it, and taking it to stand still loses that motion at every
    change of stance; without one, the foot most loaded at the step's start.

    Which point: ``sole``, its sole point.
    ``corner``: its lowest corners, weighted by how close each is to the lowest -- the
    edge a foot rolls about when its heel or toe leaves the ground, still while the sole
    point under the ankle moves.
    """
    feet = list(model.feet)
    if pivot == "sole":
        offset = sole_points(model, rot, com) - com[..., feet, :]
        point_vel = com_vel[..., feet, :] + np.cross(ang_vel[..., feet, :], offset)
    elif pivot == "corner":
        points = _corners(model, rot, com)
        arm = points - com[..., feet, None, :]
        corner_vel = com_vel[..., feet, None, :] + np.cross(ang_vel[..., feet, None, :], arm)
        height = points[..., 2]
        weight = np.exp(-(height - height.min(axis=-1, keepdims=True)) / CORNER_SOFTNESS)
        weight = weight / weight.sum(axis=-1, keepdims=True)
        point_vel = (weight[..., None] * corner_vel).sum(axis=-2)
    else:
        raise ValueError(f"pivot must be one of {PIVOTS}, not {pivot!r}")
    if previous_load is None:
        stance = np.argmax(foot_load, axis=-1)
    else:
        loaded = (foot_load > LOADED) & (previous_load > LOADED)
        speed = np.where(loaded, np.linalg.norm(point_vel, axis=-1), np.inf)
        stance = np.where(loaded.any(axis=-1), speed.argmin(axis=-1), previous_load.argmax(axis=-1))
    return np.take_along_axis(point_vel, stance[..., None, None], axis=-2)[..., 0, :]


def proprioceptive_skill_state(
    model: BodyModel,
    rot: np.ndarray,
    com: np.ndarray,
    com_vel: np.ndarray,
    ang_vel: np.ndarray,
    foot_load: np.ndarray,
    pivot: str = "sole",
    previous_load: np.ndarray | None = None,
) -> np.ndarray:
    """``z*`` from what a body senses: its kinematics, and loads ``(..., 2)`` in units of mg.

    As :func:`~...skill_torch.proprioceptive_skill_state` with ``base_velocity=False`` and
    sole corners: the ground is under the lowest corner of either sole, and every velocity
    is measured from a stance foot taken to stand still (leg odometry): the more loaded
    one, or with ``previous_load`` the one :func:`stance_velocity` picks, at its ``pivot``.
    """
    points = _corners(model, rot, com)
    ground = points[..., 2].reshape(*points.shape[:-3], -1).min(axis=-1)
    lifted = com.copy()
    lifted[..., 2] = com[..., 2] - ground[..., None] + model.ground_height
    still = stance_velocity(model, rot, com, com_vel, ang_vel, foot_load, pivot, previous_load)
    velocity = com_vel - still[..., None, :]
    force = np.zeros((*foot_load.shape, 3))
    force[..., 2] = foot_load * model.force_scale
    return skill_state(model, rot, lifted, velocity, ang_vel, force)


__all__ = [
    "CORNER_SOFTNESS",
    "LOADED",
    "PIVOTS",
    "proprioceptive_skill_state",
    "skill_state",
    "sole_points",
    "stance_velocity",
]
