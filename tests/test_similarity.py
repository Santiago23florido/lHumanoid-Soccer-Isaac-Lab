"""Checks on the unit system a body defines.

The general rule has to reproduce the two special cases the project already
relies on -- Froude-matched speed and pendulum-time scaling -- or it is a
different rule wearing the same name.
"""

from __future__ import annotations

import math

import pytest
from humanoid_transfer.common.capturability import lipm_omega
from humanoid_transfer.common.scaling import (
    froude_matched_speed,
    froude_matched_time,
    froude_number,
)
from humanoid_transfer.common.similarity import (
    ANGULAR_MOMENTUM,
    ANGULAR_VELOCITY,
    DIMENSIONLESS,
    FORCE,
    INERTIA,
    LENGTH,
    MASS,
    ROTATIONAL_DAMPING,
    TIME,
    TORQUE,
    VELOCITY,
    Dimension,
    UnitSystem,
)

TEACHER = UnitSystem(length=0.70, mass=32.0)
STUDENT = UnitSystem(length=0.25, mass=5.3)


def test_derived_units_are_the_ones_the_centroidal_equations_need() -> None:
    u = TEACHER
    assert u.time * u.velocity == pytest.approx(u.length)
    assert u.scale(FORCE) == pytest.approx(u.mass * u.gravity)
    assert u.scale(TORQUE) == pytest.approx(u.mass * u.gravity * u.length)
    assert u.scale(ANGULAR_MOMENTUM) == pytest.approx(u.mass * u.length * u.velocity)
    assert u.scale(INERTIA) == pytest.approx(u.mass * u.length**2)
    assert u.scale(DIMENSIONLESS) == 1.0


def test_the_time_unit_is_the_pendulum_time_at_the_characteristic_length() -> None:
    assert TEACHER.time == pytest.approx(1.0 / lipm_omega(TEACHER.length))


def test_a_dimensionless_speed_squared_is_the_froude_number() -> None:
    speed = 0.25
    assert TEACHER.to_dimensionless(speed, VELOCITY) ** 2 == pytest.approx(
        froude_number(speed, TEACHER.length)
    )


def test_converting_a_speed_is_froude_matching() -> None:
    assert TEACHER.convert_to(STUDENT, 0.25, VELOCITY) == pytest.approx(
        froude_matched_speed(0.25, TEACHER.length, STUDENT.length)
    )


def test_converting_a_duration_is_pendulum_time_scaling() -> None:
    assert TEACHER.convert_to(STUDENT, 0.8, TIME) == pytest.approx(
        froude_matched_time(0.8, TEACHER.length, STUDENT.length)
    )


@pytest.mark.parametrize(
    "dimension",
    [LENGTH, MASS, TIME, VELOCITY, ANGULAR_VELOCITY, FORCE, TORQUE, ANGULAR_MOMENTUM,
     INERTIA, ROTATIONAL_DAMPING, Dimension(mass=0.5, length=-1.5, time=2.0)],
)  # fmt: skip
def test_dimensionless_values_are_invariant_under_the_similarity_group(
    dimension: Dimension,
) -> None:
    """The theorem itself: scale length by l and mass by u at fixed gravity.

    Time then scales by sqrt(l), so a quantity of dimension M^a L^b T^c scales
    by u^a l^b l^(c/2) -- and so does its unit, leaving the number unchanged.
    """
    lam, mu = 0.37, 0.18
    value = 1.234
    scaled_body = UnitSystem(TEACHER.length * lam, TEACHER.mass * mu)
    physical_factor = mu**dimension.mass * lam**dimension.length * math.sqrt(lam) ** dimension.time

    original = TEACHER.to_dimensionless(value, dimension)
    rescaled = scaled_body.to_dimensionless(value * physical_factor, dimension)
    assert rescaled == pytest.approx(original)


def test_round_trip_through_the_dimensionless_form_is_exact() -> None:
    assert TEACHER.to_si(TEACHER.to_dimensionless(3.0, TORQUE), TORQUE) == pytest.approx(3.0)


@pytest.mark.parametrize("field", ["length", "mass", "gravity"])
def test_a_non_positive_unit_is_refused(field: str) -> None:
    values = {"length": 0.7, "mass": 32.0, "gravity": 9.81, field: 0.0}
    with pytest.raises(ValueError):
        UnitSystem(**values)
