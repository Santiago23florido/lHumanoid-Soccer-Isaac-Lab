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
``transfer/docs/plan.md``.
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
    "upper_arm_yaw": ("LElbowYaw", "RElbowYaw"),
    "elbow_flexion": ("LElbowRoll", "RElbowRoll"),
}

# Unitree naming for the same roles. The G1 splits the ankle into pitch and
# roll as the NAO does, which is the reason the leg chain maps cleanly at all.
#
# The arms need care, and an earlier version of this map got them wrong by
# matching on the word "elbow" rather than on what the joint does:
#
#   NAO   ShoulderPitch -> ShoulderRoll -> ElbowYaw     -> ElbowRoll
#   G1    shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow_pitch -> elbow_roll
#
# NAO's ElbowYaw rotates the upper arm about its own axis, which is the G1's
# shoulder_yaw. NAO's ElbowRoll is elbow *flexion* -- its limits, [-1.545,
# -0.035], are why a NAO elbow cannot straighten -- and that is the G1's
# elbow_pitch. The G1's elbow_roll is forearm pronation, whose NAO counterpart
# is WristYaw, which this project does not command.
#
# The roles are named after the motion rather than after either robot's
# nomenclature, so the mismatch cannot recur.
TEACHER_JOINTS: dict[str, tuple[str, ...]] = {
    "hip_yaw": ("left_hip_yaw_joint", "right_hip_yaw_joint"),
    "hip_roll": ("left_hip_roll_joint", "right_hip_roll_joint"),
    "hip_pitch": ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    "knee_pitch": ("left_knee_joint", "right_knee_joint"),
    "ankle_pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
    "ankle_roll": ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    "shoulder_pitch": ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    "shoulder_roll": ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
    "upper_arm_yaw": ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint"),
    "elbow_flexion": ("left_elbow_pitch_joint", "right_elbow_pitch_joint"),
}

# Sign to apply to a teacher angle to express it in the student's convention,
# per role and per side. A name map is not enough: the two robots disagree about
# which direction is positive for one of these joints, and the disagreement is
# systematic rather than arbitrary.
#
# In the NAO's URDF the left and right axis vectors are *identical* -- both
# ``LElbowRoll`` and ``RElbowRoll`` rotate about ``+z`` -- and the mirroring
# lives in the rotated origin of the parent link. So the same positive rotation
# produces opposite anatomical motions on the two sides, which is visible in the
# limits: roll and yaw joints have negated ranges across sides,
#
#     LElbowRoll     [-1.545, -0.035]      RElbowRoll     [+0.035, +1.545]
#     LShoulderRoll  [-0.314, +1.326]      RShoulderRoll  [-1.326, +0.314]
#     LHipRoll       [-0.379, +0.790]      RHipRoll       [-0.790, +0.379]
#
# while pitch joints, whose axis lies in the sagittal plane and is shared, have
# identical ranges across sides,
#
#     LKneePitch     [-0.092, +2.113]      RKneePitch     [-0.092, +2.113]
#
# A sign is required exactly where the two robots place the same motion on axes
# of *different mirroring class*. That happens once: the NAO puts elbow flexion
# on a roll axis (mirrored) and the G1 puts it on a pitch axis (not mirrored).
# The captured teacher gait shows both of its elbows flexing positive,
# ``left_elbow_pitch_joint`` in [+0.830, +0.910] and the right in
# [+0.825, +0.909], so copying the angle across unsigned lands inside
# ``RElbowRoll`` and outside ``LElbowRoll`` for every sample.
#
# This is the same root cause as the naming error above, one level deeper: the
# names were corrected to match the motion, but matching the motion is not
# enough when the sign conventions for that motion differ.
JOINT_SIGN: dict[str, tuple[float, ...]] = {
    "hip_yaw": (1.0,),
    "hip_roll": (1.0, 1.0),
    "hip_pitch": (1.0, 1.0),
    "knee_pitch": (1.0, 1.0),
    "ankle_pitch": (1.0, 1.0),
    "ankle_roll": (1.0, 1.0),
    "shoulder_pitch": (1.0, 1.0),
    "shoulder_roll": (1.0, 1.0),
    "upper_arm_yaw": (1.0, 1.0),
    "elbow_flexion": (-1.0, 1.0),
}
"""Per-side sign from the teacher's convention to the student's.

Only ``elbow_flexion`` is not the identity, for the reason derived above. The
entries that are ``1.0`` are not all equally well established: a joint whose
range is symmetric about zero, such as ``LShoulderPitch`` at
:math:`\\pm 2.086`, admits either sign without ever leaving its limits, so a
reachability check cannot reveal a sign error there. For those joints the
evidence is the matching mirroring class, not the measurement, and
:mod:`feasibility` says so where it reports.
"""

EXPECTED_UNMAPPED = (
    "torso_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
)
"""Teacher joints with a known, understood absence of a student counterpart.

Distinguished from an unexpected one because the two need different responses.
A surprise means the map is wrong; these mean the robots differ.

``torso_joint``
    Waist rotation. The NAO has no waist: its torso is rigid from pelvis to
    neck. If the teacher regulates angular momentum through the waist, the
    student has nothing that reproduces it.
``*_elbow_roll_joint``
    Forearm pronation and supination. The NAO's counterpart is ``WristYaw``,
    which this project holds at a fixed posture rather than commanding, so
    there is no channel to send it to.

Hand and finger joints are excluded separately by :func:`is_hand_joint`: there
are fourteen of them and listing each would obscure the three above, which are
the ones that matter for a gait.
"""


def is_hand_joint(name: str) -> bool:
    """Whether a teacher joint belongs to a hand.

    The shipped G1 asset carries seven joints per hand, named by ordinal --
    ``left_zero_joint`` through ``left_six_joint``. They contribute nothing to
    a walking gait and the NAO's own hands are driven by a single mimic joint
    per side, so they are excluded from the correspondence by construction
    rather than reported as gaps.
    """
    ordinals = ("zero", "one", "two", "three", "four", "five", "six")
    return any(f"_{ordinal}_joint" in name for ordinal in ordinals)


def signed_channels() -> tuple[tuple[str, str, float], ...]:
    """Every mapped joint as ``(teacher_joint, student_joint, sign)``.

    The flat form the retargeting needs: one entry per commanded student joint
    that has a teacher counterpart, carrying the sign from :data:`JOINT_SIGN`
    so a caller cannot copy an angle across while forgetting the convention.

    Returns 19 entries minus the unpaired ones -- the NAO's single
    ``LHipYawPitch`` takes the teacher's left hip yaw and the teacher's right
    hip yaw has no destination, which :class:`Correspondence` reports as an
    asymmetric role.
    """
    channels: list[tuple[str, str, float]] = []
    for role in sorted(set(STUDENT_JOINTS) & set(TEACHER_JOINTS)):
        student = STUDENT_JOINTS[role]
        teacher = TEACHER_JOINTS[role]
        signs = JOINT_SIGN[role]
        if len(signs) != len(student):
            raise ValueError(
                f"role {role!r} has {len(student)} student joints but {len(signs)} signs"
            )
        for side, student_joint in enumerate(student):
            if side < len(teacher):
                channels.append((teacher[side], student_joint, signs[side]))
    return tuple(channels)


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
        role for role in shared if len(STUDENT_JOINTS[role]) != len(TEACHER_JOINTS[role])
    )
    return Correspondence(
        roles=shared,
        student_only=student_only,
        teacher_only=teacher_only,
        asymmetric=asymmetric,
    )


def unmapped_teacher_joints(
    teacher_joint_names: list[str], include_expected: bool = True
) -> list[str]:
    """Teacher joints with no student counterpart, given an actual joint list.

    Takes the names as the loaded articulation reports them, because the shipped
    configuration is the authority on what joints exist and a hand-written list
    goes stale. Anything returned here is a channel of the teacher's action that
    the student will have to do without.

    Hand joints are always excluded; see :func:`is_hand_joint`.

    Args:
        teacher_joint_names: Joint names from the loaded articulation.
        include_expected: When False, also drops :data:`EXPECTED_UNMAPPED`,
            leaving only gaps that were not anticipated. An empty result then
            means the map accounts for every joint it should.
    """
    mapped = {name for names in TEACHER_JOINTS.values() for name in names}
    skip = set() if include_expected else set(EXPECTED_UNMAPPED)
    return [
        name
        for name in teacher_joint_names
        if name not in mapped and not is_hand_joint(name) and name not in skip
    ]


def unexpected_unmapped_joints(teacher_joint_names: list[str]) -> list[str]:
    """Gaps the map did not anticipate. A non-empty result is a defect.

    This is the assertion that would have caught ``left_elbow_joint``: a name
    the map used that the robot does not have, which fails as a silently
    dropped channel rather than as an error.
    """
    return unmapped_teacher_joints(teacher_joint_names, include_expected=False)


__all__ = [
    "Correspondence",
    "EXPECTED_UNMAPPED",
    "JOINT_SIGN",
    "STUDENT_JOINTS",
    "TEACHER_JOINTS",
    "build_correspondence",
    "is_hand_joint",
    "signed_channels",
    "unexpected_unmapped_joints",
    "unmapped_teacher_joints",
]
