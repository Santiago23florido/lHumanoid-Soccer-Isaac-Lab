"""The joint-target baseline's retargeting (``transfer/retarget.py``).

Each value map is held to the one property that defines it, with the correspondence's
signs in play (the left elbow is the joint whose sign flips). The resampling is held to
the property the clock depends on: a NAO step reads the teacher at ``k × factor``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.correspondence import signed_channels  # noqa: E402
from humanoid_transfer.transfer.retarget import (  # noqa: E402
    JointRecord,
    episodes,
    resample,
    retarget_angles,
)

CHANNELS = signed_channels()


def records() -> tuple[JointRecord, JointRecord]:
    teacher_names = tuple(t for t, _, _ in CHANNELS) + ("torso_joint",)
    student_names = tuple(s for _, s, _ in CHANNELS) + ("HeadYaw",)
    rng = np.random.default_rng(0)
    teacher = JointRecord(
        teacher_names,
        default=rng.uniform(-0.3, 0.3, len(teacher_names)),
        lower=np.full(len(teacher_names), -2.0),
        upper=np.full(len(teacher_names), 2.5),
    )
    student = JointRecord(
        student_names,
        default=rng.uniform(-0.2, 0.2, len(student_names)),
        lower=np.full(len(student_names), -1.0),
        upper=np.full(len(student_names), 1.5),
    )
    return teacher, student


def test_offset_puts_the_teacher_default_on_the_student_default() -> None:
    teacher, student = records()
    out = retarget_angles(teacher.default[None], teacher, student, "offset")[0]
    np.testing.assert_allclose(out, student.default)


def test_copy_carries_the_angle_with_its_sign() -> None:
    teacher, student = records()
    angles = np.zeros(len(teacher.names))
    angles[teacher.index("left_elbow_pitch_joint")] = 0.8
    angles[teacher.index("right_elbow_pitch_joint")] = 0.8
    out = retarget_angles(angles[None], teacher, student, "copy")[0]
    assert out[student.index("LElbowRoll")] == pytest.approx(-0.8)
    assert out[student.index("RElbowRoll")] == pytest.approx(0.8)


def test_range_maps_the_teacher_bounds_onto_the_student_bounds() -> None:
    teacher, student = records()
    low = retarget_angles(teacher.lower[None], teacher, student, "range")[0]
    high = retarget_angles(teacher.upper[None], teacher, student, "range")[0]
    for t, s, sign in CHANNELS:
        j = student.index(s)
        # A flipped sign turns the teacher's lower bound into the student's upper one.
        expected_low, expected_high = (student.lower[j], student.upper[j])[:: int(sign)]
        assert low[j] == pytest.approx(expected_low), (t, s)
        assert high[j] == pytest.approx(expected_high), (t, s)


@pytest.mark.parametrize("mode", ["copy", "offset", "range"])
def test_every_map_stays_inside_the_student_limits_and_holds_unmapped_joints(mode: str) -> None:
    teacher, student = records()
    angles = np.random.default_rng(1).uniform(-3.0, 3.0, (64, len(teacher.names)))
    out = retarget_angles(angles, teacher, student, mode)
    assert np.all(out >= student.lower - 1e-12) and np.all(out <= student.upper + 1e-12)
    np.testing.assert_allclose(out[:, student.index("HeadYaw")], student.default[-1])


def test_resampling_reads_the_source_at_k_times_the_factor() -> None:
    series = np.stack([np.arange(100.0), 3.0 * np.arange(100.0)], -1)
    out = resample(series, 1.625)
    np.testing.assert_allclose(out[:, 0], np.arange(len(out)) * 1.625)
    np.testing.assert_allclose(out[:, 1], 3.0 * np.arange(len(out)) * 1.625)
    assert out[-1, 0] <= 99.0


def test_episodes_split_at_each_reset() -> None:
    done = np.zeros((10, 2), dtype=bool)
    done[3, 0] = True
    done[9, 1] = True
    assert episodes(done) == [(0, 0, 3), (0, 4, 9), (1, 0, 9)]
