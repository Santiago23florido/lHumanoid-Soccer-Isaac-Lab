"""The whole-body prior's physics: Jacobians of points, the least-squares step, leg forces."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.whole_body import (  # noqa: E402
    com_jacobian,
    damped_least_squares,
    leg_forces,
    orientation_error,
    point_jacobian,
    yaw_rotation,
)


@pytest.fixture(autouse=True)
def _double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / axis.norm()
    k = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return torch.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def test_a_point_on_a_spinning_link_moves_as_omega_cross_r() -> None:
    gen = torch.Generator().manual_seed(0)
    jacobian = torch.randn(4, 6, 5, generator=gen)
    origin, point = torch.randn(4, 3, generator=gen), torch.randn(4, 3, generator=gen)
    qdot = torch.randn(4, 5, generator=gen)
    twist = jacobian @ qdot[..., None]
    v, w = twist[:, :3, 0], twist[:, 3:, 0]
    expected = v + torch.linalg.cross(w, point - origin, dim=-1)
    measured = (point_jacobian(jacobian, origin, point) @ qdot[..., None])[..., 0]
    torch.testing.assert_close(measured, expected)


def test_the_com_jacobian_is_the_mass_weighted_mean_of_the_links() -> None:
    gen = torch.Generator().manual_seed(1)
    jacobian = torch.randn(3, 4, 6, 7, generator=gen)
    origin, com = torch.randn(3, 4, 3, generator=gen), torch.randn(3, 4, 3, generator=gen)
    masses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    qdot = torch.randn(3, 7, generator=gen)
    each = torch.stack(
        [(point_jacobian(jacobian[:, b], origin[:, b], com[:, b]) @ qdot[..., None])[..., 0]
         for b in range(4)],
        dim=1,
    )  # fmt: skip
    expected = (masses[None, :, None] * each).sum(1) / masses.sum()
    measured = (com_jacobian(jacobian, origin, com, masses) @ qdot[..., None])[..., 0]
    torch.testing.assert_close(measured, expected)


def test_the_least_squares_step_solves_a_square_system_and_commutes_with_a_mirror() -> None:
    gen = torch.Generator().manual_seed(2)
    jacobian = torch.randn(5, 6, 6, generator=gen)
    error = torch.randn(5, 6, generator=gen)
    step = damped_least_squares(jacobian, error, 1e-6)
    torch.testing.assert_close((jacobian @ step[..., None])[..., 0], error, atol=1e-6, rtol=1e-6)
    # A reflection of the task (M) and of the joints (P): the step reflects with them.
    task = torch.diag(torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]))
    joints = torch.zeros(6, 6)
    for i, j, sign in ((0, 3, 1.0), (1, 4, -1.0), (2, 5, 1.0)):
        joints[i, j] = joints[j, i] = sign
    mirrored = damped_least_squares(task @ jacobian @ joints, error @ task.T, 0.3)
    torch.testing.assert_close(mirrored, damped_least_squares(jacobian, error, 0.3) @ joints.T)


def test_the_orientation_error_is_the_small_rotation_between_frames() -> None:
    current = _rotation(torch.tensor([0.3, -0.2, 0.9]), 0.7)
    axis = torch.tensor([0.1, 0.5, -0.2])
    small = 1e-4
    desired = _rotation(axis, small) @ current
    error = orientation_error(current, desired)
    torch.testing.assert_close(error / small, axis / axis.norm(), atol=1e-4, rtol=1e-4)
    assert orientation_error(current, current).abs().max() < 1e-12


def test_the_yaw_rotation_keeps_only_the_heading() -> None:
    # Yaw, then pitch, then roll (Z-Y-X): the last two leave the heading alone.
    rotation = (
        _rotation(torch.tensor([0.0, 0.0, 1.0]), 0.8)
        @ _rotation(torch.tensor([0.0, 1.0, 0.0]), 0.3)
        @ _rotation(torch.tensor([1.0, 0.0, 0.0]), -0.2)
    )
    torch.testing.assert_close(
        yaw_rotation(rotation), _rotation(torch.tensor([0.0, 0.0, 1.0]), 0.8)
    )


def test_leg_forces_carry_the_load_through_the_com() -> None:
    gen = torch.Generator().manual_seed(3)
    load = torch.rand(6, 2, generator=gen)
    feet = torch.randn(6, 2, 3, generator=gen) * 0.2
    feet[..., 2] = -0.9 - torch.rand(6, 2, generator=gen) * 0.1
    forces = leg_forces(load, feet, weight=300.0)
    torch.testing.assert_close(forces[..., 2], 300.0 * load)
    # Along the leg: no moment about the CoM.
    moment = torch.linalg.cross(feet, forces, dim=-1)
    assert moment.abs().max() < 1e-9


def test_the_centre_of_pressure_is_where_a_point_force_acts() -> None:
    from humanoid_transfer.transfer.latent.whole_body import centre_of_pressure

    gen = torch.Generator().manual_seed(4)
    point = torch.randn(5, 2, 3, generator=gen)
    point[..., 2] = 0.02  # the ground
    force = torch.randn(5, 2, 3, generator=gen)
    force[..., 2] = 50.0 + 10.0 * torch.rand(5, 2, generator=gen)
    about = point + torch.randn(5, 2, 3, generator=gen) * 0.1
    about[..., 2] = 0.08
    moment = torch.linalg.cross(point - about, force, dim=-1)
    cop = centre_of_pressure(force, moment, about, torch.full((5,), 0.02))
    torch.testing.assert_close(cop, point[..., :2])


def test_the_ground_reaction_balances_the_joint_and_the_foot_weight() -> None:
    from humanoid_transfer.transfer.latent.whole_body import ground_reaction

    gen = torch.Generator().manual_seed(5)
    ground_force = torch.randn(3, 2, 3, generator=gen)
    ground_moment = torch.randn(3, 2, 3, generator=gen)
    joint = torch.randn(3, 2, 3, generator=gen)
    com = joint + 0.05 * torch.randn(3, 2, 3, generator=gen)
    mass = torch.tensor([0.6, 0.6])
    weight = torch.zeros(3, 2, 3)
    weight[..., 2] = -mass * 9.81
    # Static foot: joint wrench + ground wrench + weight = 0 (moments about the joint).
    joint_force = -ground_force - weight
    joint_moment = -ground_moment - torch.linalg.cross(com - joint, weight, dim=-1)
    force, moment = ground_reaction(joint_force, joint_moment, joint, com, mass)
    torch.testing.assert_close(force, ground_force)
    torch.testing.assert_close(moment, ground_moment)


def test_the_tilt_is_recovered_from_gravity_in_the_base() -> None:
    from humanoid_transfer.transfer.latent.whole_body import tilt_from_gravity

    pitch, roll = 0.25, -0.4
    tilt = _rotation(torch.tensor([0.0, 1.0, 0.0]), pitch) @ _rotation(
        torch.tensor([1.0, 0.0, 0.0]), roll
    )
    gravity = tilt.T @ torch.tensor([0.0, 0.0, -1.0])
    torch.testing.assert_close(tilt_from_gravity(gravity), tilt)
    # Its yaw is zero, and a vector without length reads as upright.
    torch.testing.assert_close(yaw_rotation(tilt_from_gravity(gravity)), torch.eye(3))
    torch.testing.assert_close(tilt_from_gravity(torch.zeros(3)), torch.eye(3))
