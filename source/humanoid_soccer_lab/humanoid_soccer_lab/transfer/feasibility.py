"""What the student can physically do, so the teacher can be ignored where it cannot.

The premise of the transfer is that the teacher knows a skill the student should
acquire. The premise is only partly true. Some of what the teacher does is the
skill, and some of it is available to the teacher because it is a 35 kg machine
with 0.74 m legs. Imitating the second part is not learning slowly, it is
learning something false.

This module makes the distinction computable. It asks, for a given teacher
state, whether the corresponding student state would be recoverable, and returns
a mask the student objective can weight by.

The criterion is zero-step capturability, taken from :mod:`common.capturability`
and evaluated with the student's geometry. It is a necessary condition, not a
sufficient one: a state inside the capturable envelope may still be unreachable
because the student's actuators cannot produce the torque. Torque feasibility is
a separate question and is not addressed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..common.capturability import capturable_velocity_in_direction, lipm_omega


@dataclass(frozen=True)
class StudentEnvelope:
    """The student's zero-step capturable region, as a function of direction.

    Defaults are the NAO in its nominal standing posture, derived in
    ``nao.assets.nao_kinematics`` and reproduced here so the transfer package
    can state a feasibility criterion without importing the student's URDF.
    """

    com_height: float = 0.26890
    com_xy: tuple[float, float] = (0.01263, 0.0)
    polygon_x: tuple[float, float] = (-0.0607, 0.1033)
    polygon_y: tuple[float, float] = (-0.1026, 0.1026)

    @property
    def omega(self) -> float:
        """Inverted-pendulum frequency, in rad/s."""
        return lipm_omega(self.com_height)

    def bound(self, angle: float) -> float:
        """Capturable centre-of-mass speed along ``angle``, in m/s."""
        return capturable_velocity_in_direction(
            angle,
            com_xy=self.com_xy,
            com_height=self.com_height,
            polygon_x=self.polygon_x,
            polygon_y=self.polygon_y,
        )

    def extremes(self) -> tuple[float, float]:
        """Weakest and strongest direction, in m/s. Computed exactly.

        For a rectangle seen from an interior point the extremes are analytic,
        so there is no reason to sample for them:

        * the **minimum** is the perpendicular distance to the nearest edge,
        * the **maximum** is the distance to the furthest corner.

        For the NAO this gives (0.443, 0.827), a ratio of 1.87.

        Sampling gets this wrong, and quietly. A sweep on a 15-degree grid
        returns 0.775 for the maximum because the furthest corner sits at
        48.5 degrees and no grid point lands on it; 5-degree steps give 0.809
        and 1-degree steps 0.821, converging from below. Every one of those
        numbers looks plausible, which is why the summary statistic was wrong
        in this repository's documentation for some time.

        Only the reported range was affected. The curriculum and the
        feasibility mask both evaluate :meth:`bound` at the heading they
        actually need, which was exact all along.
        """
        edges = (
            self.com_xy[0] - self.polygon_x[0],
            self.polygon_x[1] - self.com_xy[0],
            self.com_xy[1] - self.polygon_y[0],
            self.polygon_y[1] - self.com_xy[1],
        )
        corners = (
            math.hypot(cx - self.com_xy[0], cy - self.com_xy[1])
            for cx in self.polygon_x
            for cy in self.polygon_y
        )
        return self.omega * min(edges), self.omega * max(corners)


def feasible_fraction(
    com_vel_xy: np.ndarray, envelope: StudentEnvelope | None = None
) -> np.ndarray:
    """How much of the capturable bound each velocity uses, per sample.

    Args:
        com_vel_xy: Centre-of-mass velocities, shape ``(n, 2)``, in m/s. These
            are the *student's* velocities implied by a teacher state, after
            whatever scaling the retargeting applies -- not the teacher's own.
        envelope: Student envelope. Defaults to the NAO.

    Returns:
        Array of shape ``(n,)``. Values at or below 1.0 are recoverable without
        stepping; above 1.0 the student would have to step, widen its stance or
        change its centroidal angular momentum to survive the state the teacher
        is asking for.
    """
    envelope = envelope or StudentEnvelope()
    velocities = np.atleast_2d(np.asarray(com_vel_xy, dtype=float))
    speeds = np.linalg.norm(velocities, axis=1)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    bounds = np.array([envelope.bound(float(a)) for a in angles])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bounds > 0.0, speeds / bounds, np.inf)


def imitation_mask(
    com_vel_xy: np.ndarray,
    envelope: StudentEnvelope | None = None,
    margin: float = 0.9,
) -> np.ndarray:
    """Per-sample weight in [0, 1] for an imitation loss.

    Full weight where the teacher asks for something comfortably recoverable,
    decaying to zero where it asks for something outside the student's envelope.
    The decay is linear in the fraction returned by :func:`feasible_fraction`
    and reaches zero at 1.0.

    ``margin`` is where the weight starts falling. 0.9 leaves a tenth of the
    envelope as a buffer, on the argument that a walking gait has to stay
    recoverable *between* steps rather than only at the boundary.

    A hard cut would be simpler and is the obvious alternative to measure
    against: the zero-step work in this repository found that soft penalties
    price a behaviour while only constraints remove it, and whether that carries
    over from a reward term to a loss weight is an open question rather than a
    settled one.
    """
    fraction = feasible_fraction(com_vel_xy, envelope)
    if not 0.0 < margin < 1.0:
        raise ValueError(f"margin must lie in (0, 1), got {margin}")
    weight = (1.0 - fraction) / (1.0 - margin)
    return np.clip(weight, 0.0, 1.0)


__all__ = [
    "StudentEnvelope",
    "feasible_fraction",
    "imitation_mask",
]
