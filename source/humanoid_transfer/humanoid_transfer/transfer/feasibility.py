"""What the student physically cannot do, so the teacher can be ignored there.

The premise of the transfer is that the teacher knows a skill the student should
acquire. The premise is only partly true. Some of what the teacher does is the
skill, and some of it is available to the teacher because it is a 35 kg machine
with 0.74 m legs. Imitating the second part is not learning slowly, it is
learning something false.

This module makes that distinction computable. The question it answers is
deliberately narrow:

    Is the state the teacher is asking for one the student's **body** can
    occupy at all?

That is a question about the embodiment. It is not a question about strategy,
and the difference is the whole of what this module got wrong before.

A criterion that is about the body
----------------------------------
The criterion is **joint-range reachability**: mapped through
:mod:`correspondence`, does the teacher's configuration land inside the
student's joint limits? A limit is a property of the robot. It holds whatever
the policy decides to do, so weighting by it removes states the student cannot
reach rather than behaviours the designer did not expect.

For this embodiment pair it binds, hard. The NAO's elbow cannot straighten:
``LElbowRoll`` is bounded to :math:`[-1.545, -0.035]` and ``RElbowRoll`` to
:math:`[+0.035, +1.545]`, so zero is not in either range. A teacher pose with a
straight arm is unreachable, full stop, and no amount of training buys it.

A criterion that was about strategy
-----------------------------------
What this module used to do was weight by **zero-step capturability**: full
weight where the implied student state could be brought to rest without
stepping, decaying to zero at the boundary of that envelope.

That is the right criterion for the standing task, where "recover without
stepping" *is* the task being posed and the envelope is its definition. See
:mod:`nao.tasks.nao_stand`, which keeps it, correctly.

It is the wrong criterion for a walking skill, for a reason that is structural
rather than a matter of degree. Walking *is* the divergence of the DCM from the
current support polygon: :math:`\\dot\\xi = \\omega_0 (\\xi - p)`, and a gait
continues because the next footstep moves :math:`p`. A state that needs a step
is not an infeasible state, it is a *walking* state. Weighting by zero-step
capturability therefore does not measure what the student can reach; it
measures how close to standing still the teacher is, and it discounts the
teacher exactly where the teacher is demonstrating the skill.

Measured on the captured teacher gait, 12 800 samples at the Froude-matched
speed, the mask it produced was inert:

====================================  ==========
quantity                              value
====================================  ==========
capturable fraction, median           0.213
capturable fraction, maximum          0.564
samples down-weighted at all          **0.00 %**
====================================  ==========

So it was simultaneously wrong in principle and doing nothing in practice --
the margin starts biting at 0.493 m/s, 3.3x the speed the experiment runs at.
It was also computed on the **double-stance** polygon while a walking robot is
in single support for most of a cycle, which makes it 2.58x too generous
laterally; the single-foot lateral bound is 0.240 m/s against the 0.620 m/s it
assumed.

Capturability has not been deleted. It is instrumentation now:
:func:`standing_margin` reports the margin a gait *chose* to keep, which is a
result worth having, and is not permitted to change the objective that produced
it.

Not addressed
-------------
Torque feasibility. A configuration inside the joint limits may still be
unreachable because the actuators cannot hold it against gravity --
``LAnklePitch`` has 3.023 N m against a 12.98 N m/rad destabilising gravitational
stiffness. That is a real limit of the body and belongs here; it is not
implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..common.capturability import capturable_velocity_in_direction, lipm_omega
from .correspondence import signed_channels

# The commanded student joints and their URDF limits, in radians. Reproduced
# from ``nao.assets.nao_kinematics`` with the same reasoning as the joint names
# in ``correspondence``: the transfer package should be able to state a
# feasibility criterion without parsing the student's URDF. A test re-derives
# this table from the URDF so it cannot drift.
STUDENT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "LHipYawPitch": (-1.14529, 0.740718),
    "LHipRoll": (-0.379435, 0.79046),
    "LHipPitch": (-1.53589, 0.48398),
    "LKneePitch": (-0.0923279, 2.11255),
    "LAnklePitch": (-1.18944, 0.922581),
    "LAnkleRoll": (-0.397761, 0.768992),
    "RHipRoll": (-0.79046, 0.379435),
    "RHipPitch": (-1.53589, 0.48398),
    "RKneePitch": (-0.0923279, 2.11255),
    "RAnklePitch": (-1.1863, 0.932006),
    "RAnkleRoll": (-0.768992, 0.397761),
    "LShoulderPitch": (-2.08567, 2.08567),
    "LShoulderRoll": (-0.314159, 1.32645),
    "LElbowYaw": (-2.08567, 2.08567),
    "LElbowRoll": (-1.54462, -0.0349066),
    "RShoulderPitch": (-2.08567, 2.08567),
    "RShoulderRoll": (-1.32645, 0.314159),
    "RElbowYaw": (-2.08567, 2.08567),
    "RElbowRoll": (0.0349066, 1.54462),
}

# Joints whose range is symmetric about zero. A reachability check cannot detect
# a sign error on these -- either convention stays inside the limits -- so a
# clean report about them is weaker evidence than a clean report elsewhere, and
# :func:`reachability_report` says so rather than letting the reader assume.
_SIGN_BLIND = tuple(
    name
    for name, (lower, upper) in STUDENT_JOINT_LIMITS.items()
    if math.isclose(lower, -upper, abs_tol=1e-9)
)


@dataclass(frozen=True)
class JointReachability:
    """Per-sample reachability of a teacher trajectory, joint by joint."""

    joint_names: tuple[str, ...]
    """Student joints, in the order of the columns below."""

    outside: np.ndarray
    """Boolean, shape ``(n, len(joint_names))``. True where the limit is violated."""

    excess: np.ndarray
    """Signed overshoot in radians, shape ``(n, len(joint_names))``.

    Zero inside the limits, otherwise how far past the nearer bound the mapped
    angle lies. The magnitude matters: an elbow 0.9 rad outside its range is a
    different problem from one 0.01 rad outside, which is a rounding question.
    """

    sign_blind: tuple[str, ...] = field(default_factory=tuple)
    """Joints whose symmetric range makes a sign error undetectable here."""

    @property
    def any_outside(self) -> np.ndarray:
        """Per-sample, whether any joint is unreachable. Shape ``(n,)``."""
        return self.outside.any(axis=1)

    @property
    def fraction_outside(self) -> dict[str, float]:
        """Per joint, the fraction of samples outside its range."""
        return {name: float(self.outside[:, i].mean()) for i, name in enumerate(self.joint_names)}


def joint_reachability(
    teacher_joint_pos: np.ndarray,
    teacher_joint_names: list[str] | tuple[str, ...],
) -> JointReachability:
    """Check a teacher trajectory against the student's joint limits.

    Args:
        teacher_joint_pos: Teacher joint positions, shape ``(n, m)`` in radians.
        teacher_joint_names: The ``m`` column names, as the articulation reports
            them.

    Returns:
        A :class:`JointReachability` over the mapped channels only. Teacher
        joints with no student counterpart are not checked here -- they are
        reported by :func:`correspondence.unmapped_teacher_joints`, which is a
        different question with a different answer.

    The mapping applies :data:`correspondence.JOINT_SIGN`, so this catches a
    convention mismatch as well as a genuine range violation. It cannot tell
    the two apart, and should not try: both mean the state as mapped is
    unreachable, and the fix differs only after a human looks.
    """
    positions = np.atleast_2d(np.asarray(teacher_joint_pos, dtype=float))
    index = {name: i for i, name in enumerate(teacher_joint_names)}

    names: list[str] = []
    columns: list[np.ndarray] = []
    for teacher_joint, student_joint, sign in signed_channels():
        if teacher_joint not in index:
            continue
        names.append(student_joint)
        columns.append(sign * positions[:, index[teacher_joint]])

    if not names:
        raise ValueError(
            "no mapped joint found in teacher_joint_names; the names given do "
            "not look like this teacher's articulation"
        )

    mapped = np.stack(columns, axis=1)
    lower = np.array([STUDENT_JOINT_LIMITS[n][0] for n in names])
    upper = np.array([STUDENT_JOINT_LIMITS[n][1] for n in names])

    below = np.minimum(mapped - lower, 0.0)
    above = np.maximum(mapped - upper, 0.0)
    return JointReachability(
        joint_names=tuple(names),
        outside=(mapped < lower) | (mapped > upper),
        excess=below + above,
        sign_blind=tuple(n for n in names if n in _SIGN_BLIND),
    )


def reachability_mask(
    teacher_joint_pos: np.ndarray,
    teacher_joint_names: list[str] | tuple[str, ...],
    tolerance: float = 0.05,
) -> np.ndarray:
    """Per-sample weight in ``[0, 1]`` for an imitation loss.

    Full weight where every mapped joint is inside the student's range, decaying
    to zero as the worst joint goes ``tolerance`` radians past its bound.

    ``tolerance`` is a smoothing width, not a permission. The bound itself is
    hard -- the joint stops there -- so the only reason not to use a step
    function is that a gradient through a cliff is worse behaved than a gradient
    through a ramp. 0.05 rad is about 3 degrees, chosen as small relative to the
    narrowest range in the map (``LElbowRoll``, 1.510 rad) and large relative to
    the retargeting error anyone should expect.

    Note what this does *not* do: it says nothing about stepping, contact,
    velocity or momentum. A state where the student would fall over gets full
    weight, because falling over is something the student can physically do and
    therefore something its own objective has to handle. That is the division
    this module now keeps: the body's limits here, the behaviour in the reward.
    """
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    reach = joint_reachability(teacher_joint_pos, teacher_joint_names)
    worst = np.abs(reach.excess).max(axis=1)
    return np.clip(1.0 - worst / tolerance, 0.0, 1.0)


def reachability_report(
    teacher_joint_pos: np.ndarray,
    teacher_joint_names: list[str] | tuple[str, ...],
) -> str:
    """A human-readable table of where the teacher leaves the student's range."""
    reach = joint_reachability(teacher_joint_pos, teacher_joint_names)
    fractions = reach.fraction_outside

    lines = [
        f"{'student joint':<18}{'limit':>22}{'outside':>10}{'worst':>10}",
        "-" * 60,
    ]
    for i, name in enumerate(reach.joint_names):
        lower, upper = STUDENT_JOINT_LIMITS[name]
        worst = float(np.abs(reach.excess[:, i]).max())
        flag = "  <--" if fractions[name] > 0.0 else ""
        lines.append(
            f"{name:<18}[{lower:>+8.3f},{upper:>+8.3f}]"
            f"{100 * fractions[name]:>9.2f}%{worst:>10.3f}{flag}"
        )

    lines.append("-" * 60)
    lines.append(f"samples with an unreachable joint: {100 * reach.any_outside.mean():.2f}%")
    if reach.sign_blind:
        lines.append(
            f"note: {len(reach.sign_blind)} joints have ranges symmetric about "
            f"zero ({', '.join(reach.sign_blind)}),"
        )
        lines.append("      so a sign error on them cannot show up in this table.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Instrumentation: capturability, reported but never imposed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandingEnvelope:
    """The student's zero-step capturable region, as a function of direction.

    This is the **standing** task's criterion, kept here so the walking study
    can report against it. It describes the student in its nominal double-stance
    posture, which is not the posture a walking robot is in, and it answers
    "could this be brought to rest without stepping" -- a question about
    strategy, not about the body.

    Do not weight an imitation loss by it and do not terminate on it in a
    walking task. Use :func:`reachability_mask` for feasibility and read
    :func:`standing_margin` as a measurement.

    Defaults are the NAO in its nominal standing posture, derived in
    ``nao.assets.nao_kinematics``.
    """

    com_height: float = 0.26890
    com_xy: tuple[float, float] = (0.01263, 0.0)
    polygon_x: tuple[float, float] = (-0.0607, 0.1033)
    polygon_y: tuple[float, float] = (-0.1026, 0.1026)
    """Double-stance polygon, feet at +/-50 mm.

    For single support this is 2.58x too wide laterally: one foot spans
    ``(-0.0526, 0.0397)``, giving a lateral bound of 0.240 m/s rather than
    0.620 m/s. Pass ``polygon_y=(-0.0526, 0.0397)`` when the robot is on one
    foot, which during a gait is most of the time.
    """

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


def standing_margin(com_vel_xy: np.ndarray, envelope: StandingEnvelope | None = None) -> np.ndarray:
    """How much of the zero-step capturable bound each velocity uses.

    **Instrumentation.** Reporting that a learned gait kept, say, 0.21 of its
    zero-step margin is a result: it says the policy chose to stay recoverable
    without being told to. Requiring 0.21 would make the same number an
    assumption, and the finding would be circular.

    Args:
        com_vel_xy: Centre-of-mass velocities, shape ``(n, 2)``, in m/s, in the
            *student's* units -- after whatever Froude scaling the retargeting
            applies, not the teacher's own.
        envelope: Student envelope. Defaults to the NAO in double stance.

    Returns:
        Array of shape ``(n,)``. At or below 1.0 the state could be brought to
        rest without stepping; above 1.0 a step is needed, which for a walking
        gait is the normal condition and not a fault.
    """
    envelope = envelope or StandingEnvelope()
    velocities = np.atleast_2d(np.asarray(com_vel_xy, dtype=float))
    speeds = np.linalg.norm(velocities, axis=1)
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    bounds = np.array([envelope.bound(float(a)) for a in angles])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bounds > 0.0, speeds / bounds, np.inf)


__all__ = [
    "JointReachability",
    "STUDENT_JOINT_LIMITS",
    "StandingEnvelope",
    "joint_reachability",
    "reachability_mask",
    "reachability_report",
    "standing_margin",
]
