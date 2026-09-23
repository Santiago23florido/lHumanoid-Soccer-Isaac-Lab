"""Checks on the teacher track.

Two kinds. The dynamic-similarity maths is pure and tested directly. The task
configuration imports Isaac Lab, so it is checked by reading the source -- less
satisfying, but it catches the failure that actually matters: a command range
left at the simulator's default, which wastes a training run on a gait the
student can never reach and produces no error at all.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from humanoid_transfer.common.scaling import (
    froude_matched_speed,
    froude_matched_time,
    froude_number,
)
from humanoid_transfer.g1.assets.g1 import NOMINAL_BASE_HEIGHT, describe_source_robot

ROOT = Path(__file__).resolve().parents[1]
TASK_CFG = (
    ROOT
    / "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_walk/g1_walk_env_cfg.py"
)

STUDENT_LEG = 0.2689
TEACHER_LEG = 0.74


# --- dynamic similarity ------------------------------------------------------


def test_matched_speeds_have_equal_froude_numbers() -> None:
    """The defining property. If this fails, the scaling is not Froude matching."""
    student = 0.15
    teacher = froude_matched_speed(student, STUDENT_LEG, TEACHER_LEG)
    assert froude_number(student, STUDENT_LEG) == pytest.approx(
        froude_number(teacher, TEACHER_LEG)
    )


def test_the_teacher_speed_for_this_pair_is_about_a_quarter_metre_per_second() -> None:
    assert froude_matched_speed(0.15, STUDENT_LEG, TEACHER_LEG) == pytest.approx(
        0.2488, abs=1e-3
    )


def test_scaling_goes_with_the_square_root_not_the_ratio() -> None:
    """Halving the leg length divides the similar speed by 1.41, not by 2.

    Getting this wrong is not a rounding error: at this pair's ratio it is the
    difference between 0.249 and 0.413 m/s, and the second is past the point
    where the student's gait stops being recoverable between steps.
    """
    halved = froude_matched_speed(1.0, 1.0, 0.5)
    assert halved == pytest.approx(1.0 / math.sqrt(2.0))
    assert halved != pytest.approx(0.5)


def test_matching_is_its_own_inverse() -> None:
    there = froude_matched_speed(0.15, STUDENT_LEG, TEACHER_LEG)
    back = froude_matched_speed(there, TEACHER_LEG, STUDENT_LEG)
    assert back == pytest.approx(0.15)


def test_a_larger_system_is_similar_at_a_higher_speed() -> None:
    assert froude_matched_speed(0.15, STUDENT_LEG, TEACHER_LEG) > 0.15


def test_durations_scale_the_same_way_as_speeds() -> None:
    """The contact schedule needs this; it carries no length information itself."""
    assert froude_matched_time(1.0, STUDENT_LEG, TEACHER_LEG) == pytest.approx(
        math.sqrt(TEACHER_LEG / STUDENT_LEG)
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_lengths_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        froude_matched_speed(0.15, bad, TEACHER_LEG)
    with pytest.raises(ValueError, match="must be positive"):
        froude_number(0.15, bad)


# --- the teacher is the more capable machine ---------------------------------


def test_every_selectable_teacher_is_taller_than_the_student() -> None:
    for name in ("g1", "h1"):
        assert NOMINAL_BASE_HEIGHT[name] > STUDENT_LEG


def test_the_default_teacher_has_more_joints_than_the_student_commands() -> None:
    assert describe_source_robot()["body_dof"] > 19


# --- the task configuration ---------------------------------------------------


def test_the_task_config_exists() -> None:
    assert TASK_CFG.is_file()


def test_the_task_builds_on_the_shipped_locomotion_environment() -> None:
    """Reproducing a working G1 gait from scratch would add risk and no result."""
    source = TASK_CFG.read_text(encoding="utf-8")
    assert "G1FlatEnvCfg" in source
    assert "class G1WalkTeacherEnvCfg(G1FlatEnvCfg)" in source


def test_the_forward_command_is_capped_at_the_matched_speed() -> None:
    """The one change this task makes, and the one that is easy to lose.

    A command range left at the shipped default samples to 1.0 m/s, four times
    the useful speed, and nothing reports it.
    """
    source = TASK_CFG.read_text(encoding="utf-8")
    assert "ranges.lin_vel_x = (0.0, FROUDE_MATCHED_SPEED)" in source


def test_the_unreachable_command_dimensions_are_zeroed() -> None:
    """Lateral and turning commands have no student counterpart yet."""
    source = TASK_CFG.read_text(encoding="utf-8")
    for line in (
        "ranges.lin_vel_y = (0.0, 0.0)",
        "ranges.ang_vel_z = (0.0, 0.0)",
    ):
        assert line in source


def test_the_scaling_is_imported_rather_than_reimplemented() -> None:
    """It is embodiment-agnostic theory and belongs in common/, tested once."""
    source = TASK_CFG.read_text(encoding="utf-8")
    assert "from ....common.scaling import froude_matched_speed" in source
    assert "def froude_matched_speed" not in source


def test_the_student_target_stays_inside_its_weakest_capturable_direction() -> None:
    """0.443 m/s backward is the binding limit, and a gait has to stay
    recoverable between steps rather than only at the boundary."""
    source = TASK_CFG.read_text(encoding="utf-8")
    assert "STUDENT_TARGET_SPEED = 0.15" in source
    assert 0.15 < 0.443 / 2.0


def test_the_play_variant_fixes_the_speed_for_comparable_rollouts() -> None:
    source = TASK_CFG.read_text(encoding="utf-8")
    assert "class G1WalkTeacherEnvCfg_PLAY" in source
    assert "ranges.lin_vel_x = (FROUDE_MATCHED_SPEED, FROUDE_MATCHED_SPEED)" in source


# --- the scripts that make the track usable ----------------------------------


@pytest.mark.parametrize(
    "script", ["check_env.py", "capture_rollouts.py", "render_gait.py"]
)
def test_teacher_scripts_exist(script: str) -> None:
    assert (ROOT / "g1" / "scripts" / script).is_file()


def test_rollout_capture_records_all_three_candidate_signals() -> None:
    """Deciding later which to use is free; re-running the capture is not."""
    source = (ROOT / "g1/scripts/capture_rollouts.py").read_text(encoding="utf-8")
    for key in ("joint_pos", "com", "angular_momentum", "foot_force"):
        assert f'"{key}"' in source


def test_rollout_capture_stores_joint_names() -> None:
    """The correspondence map is applied afterwards, not baked into the capture."""
    source = (ROOT / "g1/scripts/capture_rollouts.py").read_text(encoding="utf-8")
    assert "joint_names=" in source
