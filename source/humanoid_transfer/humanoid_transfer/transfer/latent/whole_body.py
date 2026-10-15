"""Whole-body kinematics and statics in torch: the physics a tracker's prior is built from.

A tracker that maps a plan of ``z*`` to joint targets has to learn inverse
kinematics and the forces that hold the body up, and a network fitted on the
plans it practised extrapolates neither (DTC, Jenelten et al. 2024: "learning IK
appears difficult"). Both are known in closed form from the body's model, so the
prior of :mod:`~...tasks.tracking.physics_prior` computes them and leaves the
network the correction:

* **Where the joints should be.** One damped least-squares step
  (:func:`damped_least_squares`) puts each sole where the plan's
  ``foot_position`` asks, relative to the CoM, with the foot flat.
* **What holds the body up.** The plan's loads, as forces along the legs through
  the CoM (:func:`leg_forces`, the variable-height inverted pendulum the
  procedural family is built on), mapped to joint torques through the contact
  Jacobians, with gravity compensation: ``tau = g(q) - sum_f J_f^T F_f``.

Pure torch, batched over environments; the Jacobians come from the simulator
(or any rigid-body library) in the world frame, a floating base's six columns
first.
"""

from __future__ import annotations

import torch


def skew(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3) -> (..., 3, 3)``: the matrix of ``v x``."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def point_jacobian(
    jacobian: torch.Tensor, origin: torch.Tensor, point: torch.Tensor
) -> torch.Tensor:
    """Linear Jacobian of a point fixed to a link: ``J_v - [p - o]x J_w``.

    Args:
        jacobian: ``(..., 6, D)`` the link frame's Jacobian, linear rows first.
        origin: ``(..., 3)`` the link frame's origin, where ``jacobian`` is taken.
        point: ``(..., 3)`` the point, in the same frame.
    """
    lever = skew(point - origin)
    return jacobian[..., :3, :] - lever @ jacobian[..., 3:, :]


def com_jacobian(
    jacobian: torch.Tensor, origin: torch.Tensor, com: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    """``(N, 3, D)`` Jacobian of the whole-body CoM from every link's.

    Args:
        jacobian: ``(N, B, 6, D)``; origin, com: ``(N, B, 3)``; masses: ``(B,)``.
    """
    each = point_jacobian(jacobian, origin, com)
    return torch.einsum("b,nbid->nid", masses, each) / masses.sum()


def damped_least_squares(
    jacobian: torch.Tensor, error: torch.Tensor, damping: float
) -> torch.Tensor:
    """``J^T (J J^T + d^2 I)^-1 e``, ``(N, R, C), (N, R) -> (N, C)``.

    The minimum-norm step towards ``e`` that stays bounded near singularities
    (Wampler 1986; Nakamura and Hanafusa 1986). It commutes with any reflection of
    the task and joint spaces, so a mirror-symmetric body gets a mirror-symmetric
    step.
    """
    rows = jacobian.shape[-2]
    eye = torch.eye(rows, dtype=jacobian.dtype, device=jacobian.device)
    system = jacobian @ jacobian.transpose(-1, -2) + damping**2 * eye
    solved = torch.linalg.solve(system, error.unsqueeze(-1))
    return (jacobian.transpose(-1, -2) @ solved).squeeze(-1)


def yaw_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)``: the rotation about the vertical by ``rotation``'s yaw."""
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
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


def orientation_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    """``(..., 3)``: the small rotation taking ``current`` to ``desired``, world frame.

    ``1/2 sum_i c_i x d_i`` over the frames' axes (Luh, Walker and Paul 1980); it is
    the rotation vector to first order and zero exactly when the frames agree.
    """
    return 0.5 * torch.linalg.cross(current, desired, dim=-2).sum(dim=-1)


def leg_forces(load: torch.Tensor, foot_position: torch.Tensor, weight: float) -> torch.Tensor:
    """``(N, 2, 3)``: the ground forces on the feet, along the legs through the CoM.

    The variable-height inverted pendulum: each foot pushes along the line from its
    sole to the CoM with vertical component ``load * weight``. The forces sum to the
    plan's total and exert no moment about the CoM.

    Args:
        load: ``(N, 2)`` each foot's vertical load, in units of the body's weight.
        foot_position: ``(N, 2, 3)`` each sole relative to the CoM (``p_f - c``), any
            gravity-aligned frame; the forces come out in the same frame.
        weight: ``m g``, newtons.
    """
    below = (-foot_position[..., 2]).clamp(min=1e-3)
    vertical = load * weight
    horizontal = vertical[..., None] * (-foot_position[..., :2]) / below[..., None]
    return torch.cat([horizontal, vertical[..., None]], dim=-1)


def ground_reaction(
    joint_force: torch.Tensor,
    joint_moment: torch.Tensor,
    joint_pos: torch.Tensor,
    foot_com: torch.Tensor,
    foot_mass: torch.Tensor,
    gravity: float = 9.81,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(force, moment about the joint)`` the ground exerts on each foot, world frame.

    A simulator reports the wrench the leg transmits to the foot through its joint;
    under the quasi-static assumption the ground balances it and the foot's weight
    (as ``nao/scripts/run_baseline.py`` reconstructs the NAO's)::

        f = -f_joint - m g,     m_joint = -m_joint - (c_foot - a) x (m g)

    Args:
        joint_force, joint_moment: ``(N, 2, 3)`` the incoming joint wrench, world frame.
        joint_pos: ``(N, 2, 3)`` where the joint is (the foot link's origin).
        foot_com: ``(N, 2, 3)``; foot_mass: ``(2,)`` or ``(N, 2)``.
    """
    weight = torch.zeros_like(joint_force)
    weight[..., 2] = -foot_mass * gravity
    force = -joint_force - weight
    moment = -joint_moment - torch.linalg.cross(foot_com - joint_pos, weight, dim=-1)
    return force, moment


def centre_of_pressure(
    force: torch.Tensor, moment: torch.Tensor, about: torch.Tensor, ground: torch.Tensor
) -> torch.Tensor:
    """``(N, 2, 2)``: where on the ground each foot's reaction has no horizontal moment.

    Slides the wrench, given about ``about`` ``(N, 2, 3)``, down to the plane at height
    ``ground`` ``(N,)``. Meaningless where the normal force vanishes; callers mask it.
    """
    normal = force[..., 2]
    safe = torch.where(normal.abs() > 1e-6, normal, torch.ones_like(normal))
    height = about[..., 2] - ground[:, None]
    x = about[..., 0] - (moment[..., 1] + height * force[..., 0]) / safe
    y = about[..., 1] + (moment[..., 0] - height * force[..., 1]) / safe
    return torch.stack([x, y], dim=-1)


__all__ = [
    "centre_of_pressure",
    "ground_reaction",
    "com_jacobian",
    "damped_least_squares",
    "leg_forces",
    "orientation_error",
    "point_jacobian",
    "skew",
    "yaw_rotation",
]
