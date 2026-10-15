"""The system of units a body defines, and the dimensionless numbers it yields.

:mod:`scaling` answers one question with dimensional analysis: which speed on
one robot corresponds to a speed on another. This module answers all of them
at once, by doing what the Buckingham pi theorem says to do -- pick a body's
own length, mass and gravity as the units, and express every quantity in them.

With leg length ``L``, total mass ``m`` and gravity ``g`` the derived units are

=====================  =======================  ==========================
quantity               unit                     for the NAO / the G1
=====================  =======================  ==========================
time                   ``T = sqrt(L / g)``      the pendulum time
velocity               ``V = sqrt(g L)``        ``v / V`` is ``sqrt(Fr)``
force                  ``m g``                  body weight
angular momentum       ``m L V``
torque                 ``m g L``
=====================  =======================  ==========================

The point is what happens to the centroidal equations of motion in these
units. With ``f* = f / (m g)``, positions over ``L`` and time over ``T``,

.. math::

    \\frac{d v^*}{d t^*} = \\sum_i f_i^* - \\hat z, \\qquad
    \\frac{d k^*}{d t^*} = \\sum_i (r_i^* - c^*) \\times f_i^* + \\tau_i^*

-- no mass, no length, no joint count. **Every humanoid obeys the same
dimensionless centroidal dynamics.** What differs between two bodies is which
contact wrenches and which states they can reach, and that is set by further
dimensionless groups (foot length over ``L``, torque over ``m g L``, ...),
which :mod:`transfer.embodiment` computes per robot.

Gravity is a unit rather than a constant on purpose: it keeps the test of
invariance honest, and it is the third base dimension the theorem needs.

References
----------
Buckingham (1914), the pi theorem. Alexander and Jayes (1983), dynamic
similarity in legged locomotion. Orin, Goswami and Lee (2013), centroidal
dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scaling import GRAVITY


@dataclass(frozen=True)
class Dimension:
    """Exponents of mass, length and time: ``M^mass L^length T^time``."""

    mass: float = 0.0
    length: float = 0.0
    time: float = 0.0

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(self.mass + other.mass, self.length + other.length, self.time + other.time)

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(self.mass - other.mass, self.length - other.length, self.time - other.time)

    @property
    def is_dimensionless(self) -> bool:
        return self.mass == 0.0 and self.length == 0.0 and self.time == 0.0


DIMENSIONLESS = Dimension()
MASS = Dimension(mass=1)
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
ANGULAR_VELOCITY = DIMENSIONLESS / TIME
FORCE = MASS * ACCELERATION
TORQUE = FORCE * LENGTH
MOMENTUM = MASS * VELOCITY
ANGULAR_MOMENTUM = MOMENTUM * LENGTH
INERTIA = MASS * LENGTH * LENGTH
ROTATIONAL_STIFFNESS = TORQUE
"""Torque per radian. A radian is dimensionless, so this is a torque."""
ROTATIONAL_DAMPING = TORQUE * TIME


@dataclass(frozen=True)
class UnitSystem:
    """The units a body defines: its leg length, its mass and the gravity it walks in."""

    length: float
    """Characteristic length ``L``, in metres."""

    mass: float
    """Total mass ``m``, in kilograms."""

    gravity: float = GRAVITY
    """Gravitational acceleration ``g``, in m/s^2."""

    def __post_init__(self) -> None:
        for name in ("length", "mass", "gravity"):
            value = getattr(self, name)
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def time(self) -> float:
        """``T = sqrt(L / g)``, in seconds. ``1 / T`` is the LIPM frequency at height ``L``."""
        return math.sqrt(self.length / self.gravity)

    @property
    def velocity(self) -> float:
        """``V = sqrt(g L)``, in m/s."""
        return math.sqrt(self.gravity * self.length)

    def scale(self, dimension: Dimension) -> float:
        """The unit of ``dimension`` in this system: ``m^a L^b T^c``."""
        return self.mass**dimension.mass * self.length**dimension.length * self.time**dimension.time

    def to_dimensionless(self, value: np.ndarray | float, dimension: Dimension) -> np.ndarray:
        """Express ``value`` (in SI) in this body's units."""
        return np.asarray(value, dtype=float) / self.scale(dimension)

    def to_si(self, value: np.ndarray | float, dimension: Dimension) -> np.ndarray:
        """Inverse of :meth:`to_dimensionless`."""
        return np.asarray(value, dtype=float) * self.scale(dimension)

    def convert_to(
        self, other: UnitSystem, value: np.ndarray | float, dimension: Dimension
    ) -> np.ndarray:
        """The value on ``other`` that is dynamically similar to ``value`` on this body.

        For a velocity this is Froude matching, for a duration it is the
        pendulum-time scaling in :mod:`scaling` -- the same rule applied to
        every dimension instead of to the two that were needed first.
        """
        return other.to_si(self.to_dimensionless(value, dimension), dimension)


__all__ = [
    "ACCELERATION",
    "ANGULAR_MOMENTUM",
    "ANGULAR_VELOCITY",
    "DIMENSIONLESS",
    "Dimension",
    "FORCE",
    "INERTIA",
    "LENGTH",
    "MASS",
    "MOMENTUM",
    "ROTATIONAL_DAMPING",
    "ROTATIONAL_STIFFNESS",
    "TIME",
    "TORQUE",
    "UnitSystem",
    "VELOCITY",
]
