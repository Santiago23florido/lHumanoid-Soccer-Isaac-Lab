"""The tracker's mirror maps: each its own inverse, and consistent with the skill state's."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT, mirror  # noqa: E402
from humanoid_transfer.transfer.symmetry import (  # noqa: E402
    joint_mirror,
    normalised_skill_mirror,
    observation_mirror,
)

G1_MIRROR = REPO_ROOT / "g1" / "body" / "mirror.json"
VIEW = 8


@pytest.fixture(scope="module")
def g1() -> dict:
    return json.loads(G1_MIRROR.read_text(encoding="utf-8"))


def _tracker_terms(joints: int) -> list[tuple[str, int]]:
    skill = SKILL_LAYOUT.size
    return [
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", joints),
        ("joint_vel", joints),
        ("actions", joints),
        ("contract", VIEW * skill + skill),
        ("privileged", VIEW * skill + skill + 1),
        ("contract_path", VIEW * 3),
    ]


def test_the_g1_mirror_was_measured_unambiguously(g1: dict) -> None:
    for row in g1["joints"]:
        assert row["residual"] < 0.05 < 1.0 < row["other_residual"], row["name"]
    signs = {row["name"]: row["sign"] for row in g1["joints"]}
    for row in g1["joints"]:
        assert signs[row["partner"]] == row["sign"], row["name"]


def test_the_g1_joint_mirror_is_an_involution_and_keeps_the_default_pose(g1: dict) -> None:
    names = [row["name"] for row in g1["joints"]]
    p = joint_mirror(names, g1)
    np.testing.assert_allclose(p @ p, np.eye(len(names)))
    # Isaac Lab's default G1 pose, which joint positions are measured from.
    default = np.zeros(len(names))
    values = {"hip_pitch": -0.2, "knee": 0.42, "ankle_pitch": -0.23, "elbow_pitch": 0.87}
    for i, name in enumerate(names):
        for key, value in values.items():
            if key in name:
                default[i] = value
    special = {
        "left_shoulder_roll_joint": 0.16,
        "right_shoulder_roll_joint": -0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_pitch_joint": 0.35,
        "left_one_joint": 1.0,
        "right_one_joint": -1.0,
        "left_two_joint": 0.52,
        "right_two_joint": -0.52,
    }
    for name, value in special.items():
        default[names.index(name)] = value
    np.testing.assert_allclose(p @ default, default)


def test_a_normalised_state_reflects_as_the_physical_one() -> None:
    rng = np.random.default_rng(0)
    mean = rng.normal(size=SKILL_LAYOUT.size)
    std = rng.uniform(0.1, 2.0, size=SKILL_LAYOUT.size)
    z = rng.normal(size=(5, SKILL_LAYOUT.size))
    linear, offset = normalised_skill_mirror(mean, std)
    reflected = ((z - mean) / std) @ linear.T + offset
    np.testing.assert_allclose(reflected, (mirror(z) - mean) / std, atol=1e-12)


def test_every_observation_mirror_is_its_own_inverse(g1: dict) -> None:
    names = [row["name"] for row in g1["joints"]]
    rng = np.random.default_rng(1)
    mean = rng.normal(size=SKILL_LAYOUT.size)
    std = rng.uniform(0.1, 2.0, size=SKILL_LAYOUT.size)
    m = observation_mirror(_tracker_terms(len(names)), joint_mirror(names, g1), mean, std)
    x = rng.normal(size=(7, m.offset.shape[0]))
    once = x @ m.linear.T + m.offset
    np.testing.assert_allclose(once @ m.linear.T + m.offset, x, atol=1e-10)
    assert not np.allclose(once, x)


def test_an_unknown_observation_is_refused(g1: dict) -> None:
    names = [row["name"] for row in g1["joints"]]
    with pytest.raises(KeyError):
        observation_mirror(
            [("height_scan", 4)],
            joint_mirror(names, g1),
            np.zeros(SKILL_LAYOUT.size),
            np.ones(SKILL_LAYOUT.size),
        )


def test_a_joint_shared_by_both_sides_reflects_onto_itself() -> None:
    # The NAO's hip yaw-pitch: one motor, the right joint a mimic of the left.
    spec = {
        "joints": [
            {"name": "LHipYawPitch", "partner": "RHipYawPitch", "sign": 1},
            {"name": "RHipYawPitch", "partner": "LHipYawPitch", "sign": 1},
            {"name": "LHipRoll", "partner": "RHipRoll", "sign": -1},
            {"name": "RHipRoll", "partner": "LHipRoll", "sign": -1},
        ],
        "mimic": {"RHipYawPitch": ["LHipYawPitch", -1.0]},
    }
    p = joint_mirror(["LHipYawPitch", "LHipRoll", "RHipRoll"], spec)
    np.testing.assert_allclose(p, [[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
    np.testing.assert_allclose(p @ p, np.eye(3))


def test_an_asymmetric_default_pose_still_reflects_exactly() -> None:
    from humanoid_transfer.transfer.symmetry import joint_offsets

    rng = np.random.default_rng(2)
    names = ["LHipRoll", "RHipRoll", "LKnee", "RKnee"]
    spec = {
        "joints": [
            {"name": "LHipRoll", "partner": "RHipRoll", "sign": -1},
            {"name": "RHipRoll", "partner": "LHipRoll", "sign": -1},
            {"name": "LKnee", "partner": "RKnee", "sign": 1},
            {"name": "RKnee", "partner": "LKnee", "sign": 1},
        ]
    }
    p = joint_mirror(names, spec)
    default = np.array([0.05, 0.02, 0.4, 0.3])
    scale = 0.25
    position, action = joint_offsets(p, default, scale)
    q = rng.normal(size=(6, 4))
    reflected_q = q @ p.T
    # Positions relative to the default pose, and actions as scaled offsets from it.
    np.testing.assert_allclose((q - default) @ p.T + position, reflected_q - default)
    np.testing.assert_allclose(
        ((q - default) / scale) @ p.T + action, (reflected_q - default) / scale
    )
    np.testing.assert_allclose(p @ action, -action)
    m = observation_mirror(
        [("joint_pos", 4), ("actions", 4)],
        p,
        np.zeros(SKILL_LAYOUT.size),
        np.ones(SKILL_LAYOUT.size),
        position,
        action,
    )
    x = rng.normal(size=(3, 8))
    once = x @ m.linear.T + m.offset
    np.testing.assert_allclose(once @ m.linear.T + m.offset, x, atol=1e-12)


def test_the_priors_feedforward_reflects_as_a_joint_velocity(g1: dict) -> None:
    names = [row["name"] for row in g1["joints"]]
    p = joint_mirror(names, g1)
    mean, std = np.zeros(SKILL_LAYOUT.size), np.ones(SKILL_LAYOUT.size)
    torque = observation_mirror([("feedforward", len(names))], p, mean, std)
    velocity = observation_mirror([("joint_vel", len(names))], p, mean, std)
    np.testing.assert_allclose(torque.linear, velocity.linear)
    np.testing.assert_allclose(torque.offset, 0.0)
