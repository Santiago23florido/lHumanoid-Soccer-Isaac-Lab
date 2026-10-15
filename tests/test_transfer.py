"""Checks on the cross-embodiment machinery.

None of this needs Isaac Lab. The theory in ``common`` and the correspondence
and feasibility logic in ``transfer`` are deliberately free of the simulator so
that they can be verified directly, which is the same reason
``nao_kinematics`` is.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from humanoid_transfer.common.capturability import (
    capturable_velocity_in_direction,
    dcm,
    lipm_omega,
    ray_box_distance,
)
from humanoid_transfer.g1.assets.g1 import SOURCE_ROBOT, describe_source_robot
from humanoid_transfer.transfer.correspondence import (
    EXPECTED_UNMAPPED,
    JOINT_SIGN,
    STUDENT_JOINTS,
    TEACHER_JOINTS,
    build_correspondence,
    is_hand_joint,
    signed_channels,
    unexpected_unmapped_joints,
    unmapped_teacher_joints,
)
from humanoid_transfer.transfer.distillation import DistillationCfg, TeacherSignal
from humanoid_transfer.transfer.feasibility import (
    STUDENT_JOINT_LIMITS,
    StandingEnvelope,
    joint_reachability,
    reachability_mask,
    standing_margin,
)

# --- the theory, stated for any robot ---------------------------------------


def test_lipm_frequency_matches_the_nao_value_derived_from_its_urdf() -> None:
    """The generalised function must reproduce what the NAO module computed.

    0.2689 m and 6.040 rad/s are the numbers the whole standing study rests on.
    If extracting the theory into ``common`` changed them, it changed the
    results too.
    """
    assert lipm_omega(0.26890) == pytest.approx(6.0400, abs=1e-4)


def test_a_lower_centre_of_mass_raises_the_capturable_speed() -> None:
    """Crouching makes a robot harder to push over, which is easy to get backwards."""
    assert lipm_omega(0.20) > lipm_omega(0.30)


def test_dcm_reduces_to_the_centre_of_mass_when_it_is_not_moving() -> None:
    position = np.array([0.01, 0.0])
    still = dcm(position, np.zeros(2), omega=6.04)
    assert still == pytest.approx(position)


def test_dcm_leads_the_centre_of_mass_in_the_direction_of_travel() -> None:
    forward = dcm(np.zeros(2), np.array([0.3, 0.0]), omega=6.04)
    assert forward[0] > 0.0


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0907), (math.pi, 0.0733), (math.pi / 2.0, 0.1026)],
)
def test_ray_box_distance_reproduces_the_measured_nao_margins(
    angle: float, expected: float
) -> None:
    """Forward, backward and lateral margins, from the nominal standing pose."""
    distance = ray_box_distance(
        origin=(0.01263, 0.0),
        angle=angle,
        box_x=(-0.0607, 0.1033),
        box_y=(-0.1026, 0.1026),
    )
    assert distance == pytest.approx(expected, abs=5e-4)


def test_a_ray_from_outside_the_polygon_is_rejected() -> None:
    """The question is only defined from inside, so it should fail loudly."""
    with pytest.raises(ValueError, match="outside the box"):
        ray_box_distance((5.0, 0.0), 0.0, (-0.06, 0.10), (-0.10, 0.10))


def test_the_capturable_bound_is_direction_dependent_by_a_factor_near_1_87() -> None:
    """The anisotropy is the reason direction-blind curricula and transfers fail."""
    low, high = StandingEnvelope().extremes()
    assert low == pytest.approx(0.443, abs=2e-3)
    assert high == pytest.approx(0.827, abs=2e-3)
    assert high / low == pytest.approx(1.87, abs=0.01)


def test_the_extremes_are_exact_rather_than_sampled() -> None:
    """A polar sweep converges to the corner from below and never reaches it.

    This is how 0.775 entered the documentation as the maximum: a 15-degree
    grid misses the corner at 48.5 degrees. The assertion pins the exact value
    against the best a fine sweep can do, so the summary statistic cannot
    silently drift back to a sampled one.
    """
    envelope = StandingEnvelope()
    _, exact = envelope.extremes()
    sweep = max(envelope.bound(2.0 * math.pi * i / 360) for i in range(360))
    assert sweep < exact
    assert exact - sweep < 0.01


def test_the_weakest_direction_is_backwards() -> None:
    envelope = StandingEnvelope()
    assert envelope.bound(math.pi) < envelope.bound(0.0)
    assert envelope.bound(math.pi) < envelope.bound(math.pi / 2.0)


def test_generalised_and_directional_helpers_agree() -> None:
    envelope = StandingEnvelope()
    direct = capturable_velocity_in_direction(
        0.0,
        com_xy=envelope.com_xy,
        com_height=envelope.com_height,
        polygon_x=envelope.polygon_x,
        polygon_y=envelope.polygon_y,
    )
    assert direct == pytest.approx(envelope.bound(0.0))


# --- correspondence ----------------------------------------------------------


def test_every_student_role_has_a_teacher_counterpart() -> None:
    correspondence = build_correspondence()
    assert correspondence.student_only == ()
    assert set(correspondence.roles) == set(STUDENT_JOINTS)


def test_the_hip_yaw_asymmetry_is_recorded_rather_than_hidden() -> None:
    """The NAO drives both hips from one motor; the G1 has two.

    This is the clearest case of the teacher being able to command something the
    student cannot reach, and the map has to say so rather than silently
    dropping one side.
    """
    correspondence = build_correspondence()
    assert "hip_yaw" in correspondence.asymmetric
    assert len(STUDENT_JOINTS["hip_yaw"]) == 1
    assert len(TEACHER_JOINTS["hip_yaw"]) == 2


def test_teacher_joints_outside_the_map_are_reported() -> None:
    """Joints with no student counterpart must surface; hands must not."""
    unmapped = unmapped_teacher_joints(
        ["left_knee_joint", "torso_joint", "left_elbow_roll_joint", "left_two_joint"]
    )
    assert unmapped == ["torso_joint", "left_elbow_roll_joint"]


def test_hand_joints_are_excluded_by_construction() -> None:
    """Seven per hand, contributing nothing to a gait, and the NAO drives its
    own hands from one mimic joint per side."""
    for ordinal in ("zero", "one", "two", "three", "four", "five", "six"):
        assert is_hand_joint(f"left_{ordinal}_joint")
    assert not is_hand_joint("left_knee_joint")
    assert not is_hand_joint("torso_joint")


def test_the_elbow_maps_to_flexion_not_to_the_matching_word() -> None:
    """NAO ElbowRoll is elbow flexion; the G1 elbow_roll is forearm pronation.

    Matching on the word rather than on the motion is the defect this pins:
    the map named left_elbow_joint, which the robot does not have, and the
    channel would have been dropped silently."""
    assert STUDENT_JOINTS["elbow_flexion"] == ("LElbowRoll", "RElbowRoll")
    assert TEACHER_JOINTS["elbow_flexion"] == (
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
    )
    named = {n for names in TEACHER_JOINTS.values() for n in names}
    assert "left_elbow_joint" not in named


def test_the_waist_and_forearm_are_recorded_as_understood_gaps() -> None:
    """The NAO has no waist and does not command its wrist."""
    assert "torso_joint" in EXPECTED_UNMAPPED
    assert "left_elbow_roll_joint" in EXPECTED_UNMAPPED


def test_a_joint_the_map_names_but_the_robot_lacks_is_a_defect() -> None:
    """An unexpected gap differs in kind from an understood one."""
    joints = ["left_knee_joint", "torso_joint", "some_new_joint"]
    assert unexpected_unmapped_joints(joints) == ["some_new_joint"]
    assert unexpected_unmapped_joints(["left_knee_joint", "torso_joint"]) == []


# --- feasibility -------------------------------------------------------------


TEACHER_ARM_NAMES = ["left_elbow_pitch_joint", "right_elbow_pitch_joint"]


def test_the_nao_elbow_cannot_straighten() -> None:
    """The fact the whole reachability criterion rests on.

    Zero is outside both elbow ranges, so a straight-arm teacher pose is not
    something the student is merely bad at -- it is unreachable.
    """
    assert STUDENT_JOINT_LIMITS["LElbowRoll"] == (-1.54462, -0.0349066)
    assert STUDENT_JOINT_LIMITS["RElbowRoll"] == (0.0349066, 1.54462)
    for joint in ("LElbowRoll", "RElbowRoll"):
        lower, upper = STUDENT_JOINT_LIMITS[joint]
        assert not lower <= 0.0 <= upper


def test_the_teacher_elbow_angle_is_reachable_on_both_sides() -> None:
    """The captured gait flexes both elbows to about +0.87 rad.

    With the sign applied this lands inside both NAO elbow ranges. Without it
    the left side is outside for every sample, which is the defect the sign
    exists to fix -- pinned by the next test.
    """
    angles = np.full((16, 2), 0.87)
    reach = joint_reachability(angles, TEACHER_ARM_NAMES)
    assert not reach.any_outside.any()
    assert reachability_mask(angles, TEACHER_ARM_NAMES) == pytest.approx(1.0)


def test_dropping_the_elbow_sign_makes_the_left_arm_unreachable() -> None:
    """Regression: the bug was invisible without this.

    A name-only map sends +0.87 rad to ``LElbowRoll``, whose range is
    [-1.545, -0.035]. Every sample is outside, by 0.905 rad. Nothing raises --
    the command would simply clamp, holding the left arm straight and the right
    at 50 degrees of flexion, asymmetrically and silently.
    """
    assert JOINT_SIGN["elbow_flexion"] == (-1.0, 1.0)

    unsigned = np.full((16, 2), 0.87)
    reach = joint_reachability(unsigned * np.array([-1.0, 1.0]), TEACHER_ARM_NAMES)
    left = reach.joint_names.index("LElbowRoll")
    assert reach.outside[:, left].all()
    assert float(np.abs(reach.excess[:, left]).max()) == pytest.approx(0.905, abs=1e-3)


def test_every_signed_channel_names_a_joint_with_a_known_limit() -> None:
    """A channel pointing at a joint with no limit would skip the check."""
    for teacher_joint, student_joint, sign in signed_channels():
        assert student_joint in STUDENT_JOINT_LIMITS
        assert teacher_joint.endswith("_joint")
        assert sign in (-1.0, 1.0)


def test_the_hip_yaw_asymmetry_drops_a_teacher_joint_not_a_student_one() -> None:
    """Every student joint gets a channel; one teacher joint has no destination.

    The NAO drives both hips from a single motor, so ``RHipYawPitch`` is mimic
    driven and never commanded -- it is not a channel that went missing. What
    is missing is a destination for the teacher's *right* hip yaw, which is why
    ``build_correspondence`` reports ``hip_yaw`` as asymmetric rather than
    reporting a count mismatch.
    """
    channels = signed_channels()
    assert len(channels) == 19

    students = [student for _, student, _ in channels]
    assert "LHipYawPitch" in students
    assert "RHipYawPitch" not in students
    assert len(set(students)) == 19

    teachers = [teacher for teacher, _, _ in channels]
    assert "left_hip_yaw_joint" in teachers
    assert "right_hip_yaw_joint" not in teachers
    assert "hip_yaw" in build_correspondence().asymmetric


def test_reachability_says_nothing_about_velocity_or_stepping() -> None:
    """The division the module now keeps.

    A configuration well inside every joint range is fully weighted no matter
    how fast the robot is moving through it. Whether that state is survivable
    is the student objective's problem, not a weight's.
    """
    angles = np.full((4, 2), 0.87)
    assert reachability_mask(angles, TEACHER_ARM_NAMES) == pytest.approx(1.0)


def test_the_reachability_weight_ramps_rather_than_cliffs() -> None:
    """A gradient through a cliff is worse behaved than one through a ramp."""
    tolerance = 0.05
    lower, _ = STUDENT_JOINT_LIMITS["RElbowRoll"]
    # The left column must stay reachable, or it becomes the worst joint and
    # drives the weight instead of the one under test.
    half = np.array([[0.87, lower - tolerance / 2.0]])
    past = np.array([[0.87, lower - tolerance * 2.0]])
    assert reachability_mask(half, TEACHER_ARM_NAMES, tolerance)[0] == pytest.approx(0.5, abs=1e-6)
    assert reachability_mask(past, TEACHER_ARM_NAMES, tolerance)[0] == 0.0


def test_reachability_rejects_a_nonpositive_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        reachability_mask(np.zeros((2, 2)), TEACHER_ARM_NAMES, tolerance=0.0)


def test_reachability_rejects_names_from_the_wrong_robot() -> None:
    """Silently checking nothing is the failure mode worth preventing."""
    with pytest.raises(ValueError, match="do not look like"):
        joint_reachability(np.zeros((2, 2)), ["LHipPitch", "RHipPitch"])


def test_symmetric_ranges_are_flagged_as_blind_to_a_sign_error() -> None:
    """Honesty about what a clean report does not prove."""
    reach = joint_reachability(
        np.zeros((4, 2)),
        ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"],
    )
    assert set(reach.sign_blind) == {"LShoulderPitch", "RShoulderPitch"}


# --- capturability is instrumentation, not a constraint ----------------------


def test_the_standing_margin_is_reported_not_enforced() -> None:
    """``standing_margin`` returns a measurement and has no masking behaviour.

    A state past the zero-step bound returns a fraction above 1.0 rather than a
    weight of zero. For a walking gait that condition is normal: the DCM has to
    leave the current polygon for the next step to mean anything.
    """
    margin = standing_margin(np.array([[-0.6, 0.0]]))
    assert margin[0] > 1.0
    assert not hasattr(standing_margin, "margin")


def test_the_captured_teacher_gait_never_approaches_the_standing_bound() -> None:
    """Why the old zero-step mask was inert as well as misdirected.

    At the Froude-matched speed the teacher uses at most 0.56 of the standing
    envelope, so a mask decaying from 0.9 never engaged on a single one of the
    12 800 captured samples. These are the scaled velocities, reproduced here
    as a closed form rather than loaded, so the test runs without the archive.
    """
    scale = 0.15 / 0.2488
    for teacher_speed in (0.20, 0.2488, 0.30):
        margin = standing_margin(np.array([[teacher_speed * scale, 0.0]]))
        assert margin[0] < 0.9


def test_the_single_support_polygon_is_much_tighter_than_double() -> None:
    """The envelope's default describes a pose a walking robot is not in."""
    double = StandingEnvelope()
    single = StandingEnvelope(polygon_y=(-0.0526, 0.0397))
    lateral = math.pi / 2.0
    assert double.bound(lateral) / single.bound(lateral) == pytest.approx(2.58, abs=0.01)


# --- distillation configuration ---------------------------------------------


def test_the_task_reward_outweighs_the_teacher_by_default() -> None:
    """When the two disagree, the robot that has to stay standing wins."""
    cfg = DistillationCfg()
    assert cfg.task_weight > cfg.imitation_weight


def test_annealing_is_off_by_default_and_decays_to_zero_when_enabled() -> None:
    assert DistillationCfg().imitation_weight_at(10_000) == pytest.approx(0.5)

    annealed = DistillationCfg(anneal_steps=1_000)
    assert annealed.imitation_weight_at(0) == pytest.approx(0.5)
    assert annealed.imitation_weight_at(500) == pytest.approx(0.25)
    assert annealed.imitation_weight_at(2_000) == pytest.approx(0.0)


def test_the_default_teacher_signal_is_embodiment_invariant() -> None:
    """Joint matching is the baseline, not the default: it transfers worst."""
    assert DistillationCfg().signal is TeacherSignal.CENTROIDAL


# --- source robot ------------------------------------------------------------


def test_the_source_robot_is_more_articulated_than_the_student() -> None:
    """The premise of the study: the teacher is the more capable machine."""
    source = describe_source_robot()
    student_commanded = 19
    assert source["body_dof"] > student_commanded


def test_the_reported_joint_counts_match_the_loaded_articulation() -> None:
    """23 body joints plus seven per hand is 37, which is what the simulator
    reports. Isaac Lab names the configuration G1_29DOF_CFG and loads the same
    USD as G1_CFG; repeating that 29 is where the discrepancy came from."""
    source = describe_source_robot("g1")
    assert source["body_dof"] == 23
    assert source["hand_dof"] == 14
    assert source["total_dof"] == 37


def test_every_selectable_source_robot_is_described() -> None:
    for name in ("g1", "h1"):
        described = describe_source_robot(name)
        assert described["body_dof"] > 0
        assert described["nominal_base_height"] > 0.5


def test_an_unknown_source_robot_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source robot"):
        describe_source_robot("atlas")


def test_the_default_source_robot_is_selectable() -> None:
    assert SOURCE_ROBOT in {"g1", "h1"}
