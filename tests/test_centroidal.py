"""Checks on the centroidal aggregation every encoder in the transfer study rests on.

Each test is a case where the right answer is known in closed form, so a
wrong term cannot hide behind plausible numbers -- which is how the G1 capture
came to document a spin term it never computed.
"""

from __future__ import annotations

import numpy as np
import pytest
from humanoid_transfer.common.centroidal import (
    angular_momentum,
    center_of_mass,
    com_acceleration,
    com_velocity,
    composite_inertia,
    world_inertia,
)
from humanoid_transfer.common.frames import axis_angle_to_matrix, quat_to_matrix, yaw_of

BODIES = 7


def _random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    return quat_to_matrix(rng.normal(size=(n, 4)))


def _random_inertias(rng: np.random.Generator, n: int) -> np.ndarray:
    """Symmetric positive-definite, deliberately anisotropic and non-diagonal."""
    a = rng.normal(size=(n, 3, 3))
    return a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3)


def _random_system(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "masses": rng.uniform(0.2, 5.0, size=BODIES),
        "inertia_local": _random_inertias(rng, BODIES),
        "link_rot": _random_rotations(rng, BODIES),
        "com_pos": rng.normal(size=(BODIES, 3)),
    }


# --- frames -----------------------------------------------------------------


def test_quaternions_are_read_in_wxyz_order() -> None:
    """A 90 degree yaw is (cos 45, 0, 0, sin 45) in w-x-y-z and something else in x-y-z-w."""
    half = np.sqrt(0.5)
    rotation = quat_to_matrix(np.array([half, 0.0, 0.0, half]))
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert yaw_of(rotation) == pytest.approx(np.pi / 2)


def test_rodrigues_agrees_with_the_quaternion_for_the_same_rotation() -> None:
    rng = np.random.default_rng(0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = 1.1
    quat = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    assert axis_angle_to_matrix(axis, angle) == pytest.approx(quat_to_matrix(quat))


# --- centre of mass ---------------------------------------------------------


def test_translating_every_body_translates_the_centre_of_mass() -> None:
    rng = np.random.default_rng(1)
    system = _random_system(rng)
    shift = np.array([0.3, -1.2, 0.05])
    moved = center_of_mass(system["masses"], system["com_pos"] + shift)
    assert moved == pytest.approx(center_of_mass(system["masses"], system["com_pos"]) + shift)


def test_centre_of_mass_acceleration_is_the_derivative_of_its_velocity() -> None:
    """Analytic trajectories r_i(t) = a_i + b_i t + c_i t^2 / 2, compared by central difference."""
    rng = np.random.default_rng(2)
    masses = rng.uniform(0.2, 5.0, size=BODIES)
    b, c = rng.normal(size=(2, BODIES, 3))
    dt = 1e-4

    def velocity(t: float) -> np.ndarray:
        return com_velocity(masses, b + c * t)

    numeric = (velocity(0.5 + dt) - velocity(0.5 - dt)) / (2 * dt)
    assert com_acceleration(masses, c) == pytest.approx(numeric, abs=1e-8)


def test_a_massless_system_is_refused() -> None:
    with pytest.raises(ValueError):
        center_of_mass(np.zeros(3), np.zeros((3, 3)))


# --- angular momentum ---------------------------------------------------------


def test_a_rigidly_rotating_posture_has_momentum_equal_to_its_composite_inertia() -> None:
    """The defining property: a body moving as one has ``k_G = I_c w``.

    Every body gets ``v_i = V + w x (r_i - c)`` and angular velocity ``w``. Both
    the orbital and the spin term are needed to reach ``I_c w``; drop either
    and this fails, which is the test the original capture would not have
    passed.
    """
    rng = np.random.default_rng(3)
    system = _random_system(rng)
    omega = np.array([0.4, -1.3, 2.1])
    drift = np.array([0.7, 0.1, -0.2])
    c = center_of_mass(system["masses"], system["com_pos"])
    velocities = drift + np.cross(omega, system["com_pos"] - c)
    ang_vel = np.broadcast_to(omega, (BODIES, 3))

    momentum = angular_momentum(
        system["masses"],
        system["inertia_local"],
        system["link_rot"],
        system["com_pos"],
        velocities,
        ang_vel,
    )
    expected = (
        composite_inertia(
            system["masses"], system["inertia_local"], system["link_rot"], system["com_pos"]
        )
        @ omega
    )

    assert momentum.total == pytest.approx(expected)
    assert com_velocity(system["masses"], velocities) == pytest.approx(drift)
    assert np.linalg.norm(momentum.spin) > 0.1 * np.linalg.norm(momentum.total)


def test_a_body_spinning_in_place_has_only_spin_momentum() -> None:
    rng = np.random.default_rng(4)
    inertia = _random_inertias(rng, 1)
    rotation = _random_rotations(rng, 1)
    omega = np.array([[1.0, 2.0, -0.5]])

    momentum = angular_momentum(
        np.array([2.0]), inertia, rotation, np.zeros((1, 3)), np.zeros((1, 3)), omega
    )

    assert momentum.orbital == pytest.approx(np.zeros(3))
    assert momentum.spin == pytest.approx(rotation[0] @ inertia[0] @ rotation[0].T @ omega[0])


def test_the_inertia_is_rotated_by_the_link_frame_and_not_left_in_it() -> None:
    """Negative control on the convention: ``I w`` without the rotation is a different number.

    If the inertia were already in the world frame -- or rotated by the wrong
    quaternion -- the two would agree only by accident.
    """
    rng = np.random.default_rng(5)
    inertia = _random_inertias(rng, 1)
    rotation = axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), 0.9)[None]
    omega = np.array([[0.3, 0.0, 1.0]])

    momentum = angular_momentum(
        np.array([1.0]), inertia, rotation, np.zeros((1, 3)), np.zeros((1, 3)), omega
    )

    assert not np.allclose(momentum.spin, inertia[0] @ omega[0])
    assert world_inertia(inertia, rotation)[0] @ omega[0] == pytest.approx(momentum.spin)


def test_two_point_masses_on_a_turntable_obey_the_parallel_axis_theorem() -> None:
    """``k_z = 2 m d^2 w`` for two point masses at distance ``d`` from the axis."""
    m, d, w = 1.5, 0.4, 3.0
    positions = np.array([[d, 0.0, 0.0], [-d, 0.0, 0.0]])
    velocities = np.array([[0.0, w * d, 0.0], [0.0, -w * d, 0.0]])

    momentum = angular_momentum(
        np.array([m, m]),
        np.zeros((2, 3, 3)),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        positions,
        velocities,
        np.zeros((2, 3)),
    )

    assert momentum.total == pytest.approx([0.0, 0.0, 2 * m * d * d * w])


def test_moving_the_observer_does_not_change_the_centroidal_momentum() -> None:
    """Galilean invariance: a common velocity added to every body leaves ``k_G`` alone."""
    rng = np.random.default_rng(6)
    system = _random_system(rng)
    velocities = rng.normal(size=(BODIES, 3))
    ang_vel = rng.normal(size=(BODIES, 3))
    args = (system["masses"], system["inertia_local"], system["link_rot"], system["com_pos"])

    still = angular_momentum(*args, velocities, ang_vel).total
    moving = angular_momentum(*args, velocities + np.array([5.0, -3.0, 1.0]), ang_vel).total

    assert moving == pytest.approx(still)


def test_aggregation_is_batched_over_leading_dimensions() -> None:
    """(time, env, body, ...) in, (time, env, ...) out, each slice equal to the unbatched result."""
    rng = np.random.default_rng(7)
    shape = (4, 3)
    masses = rng.uniform(0.2, 5.0, size=(*shape, BODIES))
    inertia = _random_inertias(rng, int(np.prod(shape)) * BODIES).reshape(*shape, BODIES, 3, 3)
    rotation = _random_rotations(rng, int(np.prod(shape)) * BODIES).reshape(*shape, BODIES, 3, 3)
    positions, velocities, ang_vel = rng.normal(size=(3, *shape, BODIES, 3))

    batched = angular_momentum(masses, inertia, rotation, positions, velocities, ang_vel).total
    one = (2, 1)
    single = angular_momentum(
        masses[one], inertia[one], rotation[one], positions[one], velocities[one], ang_vel[one]
    ).total

    assert batched.shape == (*shape, 3)
    assert batched[one] == pytest.approx(single)
