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
    STUDENT_JOINTS,
    TEACHER_JOINTS,
    build_correspondence,
    is_hand_joint,
    unexpected_unmapped_joints,
    unmapped_teacher_joints,
)
from humanoid_transfer.transfer.distillation import DistillationCfg, TeacherSignal
from humanoid_transfer.transfer.feasibility import (
    StudentEnvelope,
    feasible_fraction,
    imitation_mask,
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
    low, high = StudentEnvelope().extremes()
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
    envelope = StudentEnvelope()
    _, exact = envelope.extremes()
    sweep = max(envelope.bound(2.0 * math.pi * i / 360) for i in range(360))
    assert sweep < exact
    assert exact - sweep < 0.01


def test_the_weakest_direction_is_backwards() -> None:
    envelope = StudentEnvelope()
    assert envelope.bound(math.pi) < envelope.bound(0.0)
    assert envelope.bound(math.pi) < envelope.bound(math.pi / 2.0)


def test_generalised_and_directional_helpers_agree() -> None:
    envelope = StudentEnvelope()
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


def test_a_slow_forward_velocity_is_fully_feasible() -> None:
    mask = imitation_mask(np.array([[0.1, 0.0]]))
    assert mask[0] == pytest.approx(1.0)


def test_a_velocity_past_the_backward_bound_is_masked_out() -> None:
    """0.6 m/s backward exceeds the 0.443 m/s bound, so the teacher is ignored."""
    fraction = feasible_fraction(np.array([[-0.6, 0.0]]))
    assert fraction[0] > 1.0
    assert imitation_mask(np.array([[-0.6, 0.0]]))[0] == pytest.approx(0.0)


def test_the_same_speed_is_feasible_forward_and_not_backward() -> None:
    """The whole point of a direction-aware mask, in one assertion."""
    speed = 0.50
    assert imitation_mask(np.array([[speed, 0.0]]))[0] > 0.0
    assert imitation_mask(np.array([[-speed, 0.0]]))[0] == pytest.approx(0.0)


def test_the_mask_rejects_a_margin_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="margin"):
        imitation_mask(np.array([[0.1, 0.0]]), margin=1.5)


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
