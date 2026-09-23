"""Which joint on one robot stands for which on the other.

A teacher policy emits a vector of joint targets for the source robot. The
student has a different number of joints, in different places, with different
ranges. Something has to relate the two before any imitation objective can be
written down, and that something is a correspondence map.

This module builds the map and, as importantly, records what it cannot express.
The failures are not incidental; they are the content of the research question.
A correspondence that loses nothing would mean the embodiments were the same and
there would be nothing to study.

Status
------
The name map is implemented and tested. The value map is deliberately not: a
joint angle does not transfer as a number, and the three candidate treatments
(direct copy, range-normalised, end-effector retargeted) are alternatives to be
measured against each other rather than a decision to make in advance. See
``docs/transfer.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The NAO's 19 commanded joints, grouped by kinematic role. Mirrors
# ``nao.assets.nao_kinematics.ACTUATED_JOINTS`` but written out here because the
# transfer package should not need the student's URDF to state a correspondence.
STUDENT_JOINTS: dict[str, tuple[str, ...]] = {
    "hip_yaw": ("LHipYawPitch",),
    "hip_roll": ("LHipRoll", "RHipRoll"),
    "hip_pitch": ("LHipPitch", "RHipPitch"),
    "knee_pitch": ("LKneePitch", "RKneePitch"),
    "ankle_pitch": ("LAnklePitch", "RAnklePitch"),
    "ankle_roll": ("LAnkleRoll", "RAnkleRoll"),
    "shoulder_pitch": ("LShoulderPitch", "RShoulderPitch"),
    "shoulder_roll": ("LShoulderRoll", "RShoulderRoll"),
    "elbow_yaw": ("LElbowYaw", "RElbowYaw"),
    "elbow_roll": ("LElbowRoll", "RElbowRoll"),
}

# Unitree naming for the same roles. The G1 splits the ankle into pitch and
# roll as the NAO does, which is the reason the leg chain maps cleanly at all.
TEACHER_JOINTS: dict[str, tuple[str, ...]] = {
    "hip_yaw": ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    "hip_roll": ("left_hip_roll_joint", "right_hip_roll_joint"),
    "hip_pitch": ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    "knee_pitch": ("left_knee_joint", "right_knee_joint"),
    "ankle_pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    "ankle_roll": ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    "shoulder_pitch": ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    "shoulder_roll": ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
    "elbow_yaw": ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint"),
    "elbow_roll": ("left_elbow_joint", "right_elbow_joint"),
}


@dataclass(frozen=True)
class Correspondence:
    """A role-level map between two embodiments, with its gaps made explicit."""

    roles: tuple[str, ...]
    """Kinematic roles present on both robots."""

    student_only: tuple[str, ...] = field(default_factory=tuple)
    """Roles the student has and the teacher does not."""

    teacher_only: tuple[str, ...] = field(default_factory=tuple)
    """Roles the teacher has and the student does not.

    These are the degrees of freedom whose teacher signal has nowhere to go. On
    a 29-DOF G1 they include the waist and the wrists: the teacher may be using
    trunk rotation to regulate angular momentum, and the student has no joint
    that can reproduce it. Whatever the teacher achieves through them has to be
    achieved some other way or not at all.
    """

    asymmetric: tuple[str, ...] = field(default_factory=tuple)
    """Roles where the two robots disagree about how many joints exist.

    ``hip_yaw`` is the standing example. The NAO drives both hips from a single
    motor through a mimic constraint, so it has one; the G1 has two,
    independently. A teacher that yaws its hips differentially is asking for a
    configuration the student cannot reach, not one it is merely bad at.
    """


def build_correspondence() -> Correspondence:
    """Relate the student's joint roles to the teacher's.

    Roles are matched by name. That is enough because both robots are
    anthropomorphic bipeds and the roles were chosen to be the ones a human
    anatomist would recognise -- it would not survive a quadruped teacher, and
    is not meant to.
    """
    shared = tuple(sorted(set(STUDENT_JOINTS) & set(TEACHER_JOINTS)))
    student_only = tuple(sorted(set(STUDENT_JOINTS) - set(TEACHER_JOINTS)))
    teacher_only = tuple(sorted(set(TEACHER_JOINTS) - set(STUDENT_JOINTS)))
    asymmetric = tuple(
        role
        for role in shared
        if len(STUDENT_JOINTS[role]) != len(TEACHER_JOINTS[role])
    )
    return Correspondence(
        roles=shared,
        student_only=student_only,
        teacher_only=teacher_only,
        asymmetric=asymmetric,
    )


def unmapped_teacher_joints(teacher_joint_names: list[str]) -> list[str]:
    """Teacher joints with no student counterpart, given an actual joint list.

    Takes the names as the loaded articulation reports them, because the shipped
    configuration is the authority on what joints exist and a hand-written list
    goes stale. Anything returned here is a channel of the teacher's action that
    the student will have to do without.
    """
    mapped = {name for names in TEACHER_JOINTS.values() for name in names}
    return [name for name in teacher_joint_names if name not in mapped]


__all__ = [
    "Correspondence",
    "STUDENT_JOINTS",
    "TEACHER_JOINTS",
    "build_correspondence",
    "unmapped_teacher_joints",
]
