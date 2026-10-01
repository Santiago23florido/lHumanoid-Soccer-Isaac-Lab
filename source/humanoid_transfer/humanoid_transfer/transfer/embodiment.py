"""Each body as a short list of dimensionless numbers, and the gap between two of them.

In its own units every humanoid obeys the same centroidal equations
(:mod:`~humanoid_transfer.common.similarity`). So what distinguishes the G1 from
the NAO, as far as a skill is concerned, is not the equations but the set of
motions each body *admits*: how long its feet are, how much torque and speed
its joints have, how its mass is spread. Expressed in the body's own units,
each of those is a pure number -- a dimensionless group -- and two bodies are
dynamically similar exactly when all of them agree.

They do not agree, and this module says by how much, group by group:

.. math::

    \\delta_k = \\ln \\Pi_k^{\\text{student}} - \\ln \\Pi_k^{\\text{teacher}}

``delta = 0`` on a group means that aspect of the body scales perfectly; a
positive value means the student has relatively *more* of it. There is no
weighted total. Collapsing the defect into one number would hide exactly the
information it exists to give -- which aspect of the body the skill will
strain.

What this is not
----------------
The list is a truncation. Matching every group here is necessary for dynamic
similarity and not sufficient: the shape of each joint's workspace, the
actuator dynamics and the contact model are not summarised by a handful of
ratios. A small defect is evidence, not a guarantee.

Body groups and task groups
---------------------------
Friction and the control period are dimensionless too, but they belong to the
*task* a body is put in, not to the body: the student task will choose them.
:func:`task_groups` reports them separately so that matching them stays a
decision of that task rather than a property read off the robot.

Sources
-------
Every number below is restated from its source with a comment naming it, as
:mod:`feasibility` does for joint limits. ``tests/test_embodiment.py``
re-derives the NAO's from its URDF and the G1's from ``g1/body/body_card.json``
and ``g1/policy/teacher_env.yaml``, so neither can drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..common.similarity import (
    ANGULAR_VELOCITY,
    INERTIA,
    LENGTH,
    TIME,
    TORQUE,
    UnitSystem,
)

LEG_ROLES: tuple[str, ...] = ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
"""Leg joints compared by kinematic role, as :mod:`correspondence` maps them."""


@dataclass(frozen=True)
class EmbodimentSpec:
    """The physical facts about one body that the dimensionless groups are built from.

    All in SI units. Postural quantities are taken in the **reference posture**
    -- legs straight, arms hanging straight down -- so they compare bodies
    rather than the stances two tasks happened to choose.
    """

    name: str
    leg_length: float
    """Hip-pitch joint to sole, leg straight. The length unit ``L``."""

    mass: float
    """Total mass. The mass unit ``m``."""

    com_height: float
    """CoM height above the soles in the reference posture."""

    composite_inertia: tuple[float, float, float]
    """Whole-body inertia about the CoM in the reference posture: roll, pitch, yaw axes."""

    foot_length: float
    foot_width: float
    sole_separation: float
    """Lateral distance between the two sole points in the reference posture."""

    joint_effort: dict[str, float]
    """Torque limit per leg role, N m, as the simulation applies it."""

    joint_velocity: dict[str, float]
    """Speed limit per leg role, rad/s, as the simulation applies it."""

    @property
    def units(self) -> UnitSystem:
        return UnitSystem(length=self.leg_length, mass=self.mass)


def pi_groups(spec: EmbodimentSpec) -> dict[str, float]:
    """The body's dimensionless groups, by name."""
    u = spec.units
    groups = {
        "com_height": float(u.to_dimensionless(spec.com_height, LENGTH)),
        "inertia_roll": float(u.to_dimensionless(spec.composite_inertia[0], INERTIA)),
        "inertia_pitch": float(u.to_dimensionless(spec.composite_inertia[1], INERTIA)),
        "inertia_yaw": float(u.to_dimensionless(spec.composite_inertia[2], INERTIA)),
        "foot_length": float(u.to_dimensionless(spec.foot_length, LENGTH)),
        "foot_width": float(u.to_dimensionless(spec.foot_width, LENGTH)),
        "sole_separation": float(u.to_dimensionless(spec.sole_separation, LENGTH)),
    }
    for role in LEG_ROLES:
        groups[f"torque_{role}"] = float(u.to_dimensionless(spec.joint_effort[role], TORQUE))
    for role in LEG_ROLES:
        groups[f"speed_{role}"] = float(
            u.to_dimensionless(spec.joint_velocity[role], ANGULAR_VELOCITY)
        )
    return groups


def similarity_defect(teacher: EmbodimentSpec, student: EmbodimentSpec) -> dict[str, float]:
    """``ln(Pi_student / Pi_teacher)`` for every group. Zero is perfect similarity."""
    source, target = pi_groups(teacher), pi_groups(student)
    return {name: math.log(target[name] / source[name]) for name in source}


def task_groups(units: UnitSystem, control_dt: float, static_friction: float) -> dict[str, float]:
    """Dimensionless numbers a *task* sets: control period in body time, and friction."""
    return {
        "control_period": float(units.to_dimensionless(control_dt, TIME)),
        "static_friction": static_friction,
    }


# ---------------------------------------------------------------------------
# The two bodies of this study
# ---------------------------------------------------------------------------

NAO = EmbodimentSpec(
    name="nao",
    # nao_kinematics.forward_kinematics({}): |LThigh - l_sole|, all joints at zero.
    leg_length=0.24801,
    # nao_kinematics.load_model(): sum of link masses.
    mass=5.30540,
    # Reference posture: zero, shoulders pitched to pi/2 (arms down), elbow rolls
    # at the bound nearest zero (+/-0.0349; zero is outside both ranges).
    com_height=0.27626,
    composite_inertia=(0.145725, 0.127487, 0.024251),
    # nao_kinematics.SUPPORT_POLYGON_X / _Y_SINGLE, re-derived there from the
    # collision mesh; soles at +/- STANCE_HALF_WIDTH.
    foot_length=0.1640,
    foot_width=0.0923,
    sole_separation=0.1000,
    # URDF <limit effort velocity>, left side; nao.assets leaves effort_limit_sim
    # unset, so these are what the simulation applies.
    joint_effort={
        "hip_pitch": 3.023,
        "hip_roll": 3.348,
        "knee": 3.023,
        "ankle_pitch": 3.023,
        "ankle_roll": 3.348,
    },
    joint_velocity={
        "hip_pitch": 6.40239,
        "hip_roll": 4.16174,
        "knee": 6.40239,
        "ankle_pitch": 6.40239,
        "ankle_roll": 4.16174,
    },
)
"""The student. Every value from the vendored URDF."""

G1 = EmbodimentSpec(
    name="g1",
    # g1/body/body_card.json: leg_length, total_mass.
    leg_length=0.65457,
    mass=32.2389,
    # body_card.json postures.zero: the G1's zero pose already has legs straight
    # and arms hanging; composite inertia recomputed from bodies.* at that pose.
    com_height=0.66384,
    composite_inertia=(3.212663, 2.933481, 0.527000),
    # body_card.json foot.left_ankle_roll_link: one box per foot.
    foot_length=0.20311,
    foot_width=0.06547,
    sole_separation=0.23574,
    # g1/policy/teacher_env.yaml actuators: effort_limit_sim 300 for legs, 20 for
    # the ankle group. Speed limits from the USD (velocity_limit_sim is null):
    # body_card.json joints.vel_limits.
    joint_effort={
        "hip_pitch": 300.0,
        "hip_roll": 300.0,
        "knee": 300.0,
        "ankle_pitch": 20.0,
        "ankle_roll": 20.0,
    },
    joint_velocity={
        "hip_pitch": 32.0,
        "hip_roll": 32.0,
        "knee": 20.0,
        "ankle_pitch": 53.0,
        "ankle_roll": 53.0,
    },
)
"""The teacher, as the simulation it was trained in integrates it."""

G1_TEACHER_TASK = {"control_dt": 0.02, "static_friction": 0.8}
"""teacher_env.yaml: sim dt 0.005 x decimation 4; robot static friction 0.8 against
ground 1.0 combined by multiplication."""


__all__ = [
    "EmbodimentSpec",
    "G1",
    "G1_TEACHER_TASK",
    "LEG_ROLES",
    "NAO",
    "pi_groups",
    "similarity_defect",
    "task_groups",
]
