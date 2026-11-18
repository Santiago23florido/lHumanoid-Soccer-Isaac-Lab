"""Forward kinematics from a card reproduces what the simulator measured."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from humanoid_transfer.transfer.latent.pact.kinematics import CardKinematics

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
BODIES = [name for name in REGISTRY if not name.startswith("$")]


def _card(body: str) -> dict:
    return json.loads((REPO / REGISTRY[body]["card"]).read_text("utf-8"))


@pytest.mark.parametrize("body", BODIES)
def test_the_zero_pose_is_the_identity(body):
    card = _card(body)
    kinematics = CardKinematics(card)
    zero = np.zeros(len(kinematics.joint_names))
    np.testing.assert_allclose(kinematics.link_positions(zero), kinematics.link_pos0, atol=1e-9)
    measured = card["postures"]["zero"]
    np.testing.assert_allclose(kinematics.com(zero), measured["com"], atol=2e-6)
    np.testing.assert_allclose(kinematics.soles(zero), measured["soles"], atol=2e-6)


@pytest.mark.parametrize("body", BODIES)
def test_the_default_pose_matches_the_simulator(body):
    """The card's default posture was read from PhysX; the formula must land on it."""
    card = _card(body)
    kinematics = CardKinematics(card)
    measured = card["postures"]["default"]
    q = kinematics.default
    np.testing.assert_allclose(kinematics.link_positions(q), measured["link_pos"], atol=2e-4)
    np.testing.assert_allclose(kinematics.com(q), measured["com"], atol=2e-4)
    np.testing.assert_allclose(kinematics.soles(q), measured["soles"], atol=2e-4)
    assert kinematics.com_height_above_soles(q) == pytest.approx(
        measured["com_height_above_soles"], abs=2e-4
    )


@pytest.mark.parametrize(
    "body, chains",
    [
        (
            "h1",
            [
                ("left_hip_pitch", "left_knee", "left_ankle"),
                ("right_hip_pitch", "right_knee", "right_ankle"),
            ],
        ),
        (
            "g1",
            [
                ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint"),
                ("right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"),
            ],
        ),
    ],
)
def test_a_crouch_reaches_its_height_with_flat_feet(body, chains):
    from humanoid_transfer.transfer.latent.pact.kinematics import sagittal_crouch

    card = _card(body)
    kinematics = CardKinematics(card)
    angles = sagittal_crouch(card, chains, target=0.95, kinematics=kinematics)
    q = kinematics.default.copy()
    for name, value in angles.items():
        q[kinematics.joint_names.index(name)] = value
    assert kinematics.com_height_above_soles(q) / card["leg_length"] == pytest.approx(
        0.95, abs=1e-6
    )
    for foot in kinematics.feet:
        bent = kinematics.motions(q)[foot][:3, :3] @ kinematics.link_rot0[foot]
        # the sole is as flat as in the zero pose (the default pose tilts it 0.01 rad)
        np.testing.assert_allclose(bent[:, 2], kinematics.link_rot0[foot][:, 2], atol=1e-6)
