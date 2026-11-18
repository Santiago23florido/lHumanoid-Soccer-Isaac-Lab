"""Analytic z* anchors: consistent with the card's kinematics and their own derivatives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    KinematicAnchors,
    tilt_from_gravity,
)
from humanoid_transfer.transfer.latent.pact.kinematics import CardKinematics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
BODIES = [name for name in REGISTRY if not name.startswith("$")]
UPRIGHT = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)


def _setup(body: str):
    card = json.loads((REPO / REGISTRY[body]["card"]).read_text("utf-8"))
    anchors = KinematicAnchors(card).double()
    kinematics = CardKinematics(card)
    rng = np.random.default_rng(0)
    limits = np.clip(np.asarray(card["joints"]["pos_limits"]), -1.0, 1.0)
    q = kinematics.default + 0.2 * rng.uniform(-1, 1, len(kinematics.joint_names))
    q = np.clip(q, limits[:, 0], limits[:, 1])
    return card, anchors, kinematics, q, rng


@pytest.mark.parametrize("body", BODIES)
def test_feet_match_the_numpy_kinematics_when_upright(body):
    card, anchors, kinematics, q, _ = _setup(body)
    zero = torch.zeros(1, len(q), dtype=torch.float64)
    out = anchors(torch.as_tensor(q)[None], zero, UPRIGHT, torch.zeros(1, 3, dtype=torch.float64))
    expected = (kinematics.soles(q) - kinematics.com(q)) / card["leg_length"]
    # the card's constants are stored in float32
    np.testing.assert_allclose(out["foot_position"][0].numpy(), expected, atol=1e-6)
    # nothing moves: no relative foot velocity, no angular momentum
    assert out["foot_velocity_relative"].abs().max() < 1e-9
    assert out["angular_momentum"].abs().max() < 1e-9


@pytest.mark.parametrize("body", BODIES)
def test_relative_foot_velocity_is_the_derivative_of_foot_position(body):
    _, anchors, _, q, rng = _setup(body)
    qd = rng.uniform(-1, 1, len(q))
    dt = 1e-6
    still = torch.zeros(1, 3, dtype=torch.float64)

    def feet(angles):
        return anchors(
            torch.as_tensor(angles)[None],
            torch.zeros(1, len(q), dtype=torch.float64),
            UPRIGHT,
            still,
        )["foot_position"][0]

    numeric = (feet(q + qd * dt) - feet(q - qd * dt)) / (2 * dt)
    out = anchors(torch.as_tensor(q)[None], torch.as_tensor(qd)[None], UPRIGHT, still)
    # both are dimensionless; position over L, velocity over sqrt(gL): convert the time unit
    time_unit = np.sqrt(anchors.units.length / 9.81)
    np.testing.assert_allclose(
        out["foot_velocity_relative"][0].numpy(), numeric.numpy() * time_unit, atol=1e-6
    )


def test_the_tilt_undoes_the_projected_gravity():
    rng = np.random.default_rng(1)
    g = torch.as_tensor(rng.normal(size=(64, 3)))
    g = g / g.norm(dim=-1, keepdim=True)
    g[:, 2] = -g[:, 2].abs() - 0.2  # upright enough: the base points up
    g = g / g.norm(dim=-1, keepdim=True)
    tilt = tilt_from_gravity(g)
    down = torch.einsum("nij,nj->ni", tilt, g)
    np.testing.assert_allclose(down.numpy(), np.tile([0.0, 0.0, -1.0], (64, 1)), atol=1e-9)
    # a pure roll-pitch rotation keeps the base's forward axis in the vertical x-z plane
    assert tilt[:, 1, 0].abs().max() < 1e-12
