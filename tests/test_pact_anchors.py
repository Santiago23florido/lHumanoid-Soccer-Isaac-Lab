"""Analytic z* anchors: consistent with the card's kinematics and their own derivatives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
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


@pytest.mark.parametrize("body", BODIES)
def test_drive_loads_recover_the_ground_wrench_that_made_the_torques(body):
    """Torques built from a known wrench on one foot and the legs' weight give it back."""
    card, anchors, _, q, rng = _setup(body)
    drive = DriveLoads(card, anchors).double()
    pose = torch.as_tensor(q)[None]
    statics = drive.balance(pose, UPRIGHT)
    weight = 9.81 * float(anchors.mass.sum())
    wrench = torch.tensor([[5.0, -3.0, 0.6 * weight, 0.0, 0.0, 0.0]], dtype=torch.float64)
    tau = -statics["gravity_tau"] - torch.einsum("njk,nk->nj", statics["wrench_map"][:, 0], wrench)
    # a target that makes the drives apply exactly these torques at rest
    drive.effort.fill_(1e9)
    target = pose + tau / drive.stiffness.clamp_min(1e-9)
    out = drive(pose, torch.zeros_like(pose), UPRIGHT, target=target)
    np.testing.assert_allclose(out["loads"][0, 0].item(), 0.6, atol=0.02)
    # the other foot's chain only carries its own weight: no load
    assert abs(out["loads"][0, 1].item()) < 0.02


def test_drive_targets_follow_the_action_on_the_joints_it_drives():
    card = json.loads((REPO / REGISTRY["g1"]["card"]).read_text("utf-8"))
    names = card["joints"]["names"]
    anchors = KinematicAnchors(card)
    acted = [names[3], names[0]]
    drive = DriveLoads(card, anchors, action_joints=acted)
    pose = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32)[None]
    tau = drive.torque(pose, torch.zeros_like(pose), action=torch.tensor([[1.0, -2.0]]))
    expected = torch.zeros_like(pose)
    expected[0, 3] = drive.stiffness[3] * drive.scale * 1.0
    expected[0, 0] = drive.stiffness[0] * drive.scale * -2.0
    expected = expected.clamp(-drive.effort, drive.effort)
    assert torch.allclose(tau, expected, atol=1e-4)
