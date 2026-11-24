"""Targets read as the torque they command: inputs, loss and the deployed driver alike."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from pact_gate_d import drive_gain, drive_loss, drive_offset, drive_ratio  # noqa: E402

KP = np.array([40.0, 100.0, 0.0, 11.0, 200.0])
KD = np.array([2.0, 5.0, 0.0, 0.5, 10.0])
TAU = np.array([300.0, 200.0, 0.0, 3.0, 100.0])
GAINS = (KP, KD, TAU)


def _state(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    q0 = rng.uniform(-0.5, 0.5, 5)
    q = q0 + rng.uniform(-1, 1, (n, 5))
    qd = rng.normal(0, 3, (n, 5))
    target = q0 + rng.normal(0, 8, (n, 5))  # far past any range, as the teachers drift
    return q0, q, qd, target


def test_the_ratio_is_the_commanded_torque_over_its_limit_and_zero_without_a_drive():
    q0, q, qd, target = _state()
    ratio = drive_ratio(target, q, qd, GAINS)
    exact = (KP * (target - q) - KD * qd) / np.where(TAU > 0, TAU, 1.0)
    active = TAU > 0
    assert np.allclose(ratio[:, active], np.tanh(exact[:, active]), atol=1e-5)
    assert np.all(ratio[:, ~active] == 0)
    assert np.all(np.abs(ratio) <= 1)
    # The affine split the loss uses: gain * (T - q0) + offset is the same ratio.
    affine = drive_gain(GAINS) * (target - q0) + drive_offset(q, qd, q0, GAINS)
    assert np.allclose(np.tanh(affine[:, active]), ratio[:, active], atol=1e-5)


def test_two_saturated_targets_cost_nothing_and_an_unsaturated_error_does():
    q0, q, qd, _ = _state(n=64, seed=1)
    gain = torch.as_tensor(drive_gain(GAINS))[None]
    offset = torch.as_tensor(drive_offset(q, qd, q0, GAINS))
    moved = torch.as_tensor(np.broadcast_to(TAU > 0, (64, 5)).copy())
    far = torch.full((64, 5), 1e3)  # both far past saturation, different by 100 rad
    assert drive_loss(far, far + 100.0, gain, offset, moved).item() < 1e-6
    exact = torch.as_tensor(q - q0, dtype=torch.float32)  # the target at the current position
    assert drive_loss(exact + 0.05, exact, gain, offset, moved).item() > 1e-5


def test_the_deployed_driver_sees_what_training_saw():
    from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads

    _, q, qd, target = _state(n=64, seed=2)
    drive = DriveLoads.__new__(DriveLoads)
    torch.nn.Module.__init__(drive)
    for name, value in (("stiffness", KP), ("damping", KD), ("effort", TAU)):
        drive.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    deployed = drive.drive_ratio(as_t(q), as_t(qd), as_t(target)).numpy()
    assert np.allclose(deployed, drive_ratio(target, q, qd, GAINS), atol=1e-5)


def test_unlimited_effort_is_capped_consistently_on_digit():
    """Digit's implicit drives report FLT_MAX (~3.4e38) effort_limit; capping at
    EFFORT_LIMIT_CLIP * m g L keeps both drive_ratio and DriveLoads.drive_ratio informative
    and matching, avoiding zero BC loss and ZeroDivisionError."""
    import json

    from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
    from humanoid_transfer.transfer.latent.pact.descriptors import (
        EFFORT_LIMIT_CLIP,
        GRAVITY,
        describe,
    )

    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    card = json.loads((REPO / registry["digit"]["card"]).read_text("utf-8"))
    desc = describe(card)
    anchors = KinematicAnchors(card)
    drive = DriveLoads(card, anchors)
    cap = float(EFFORT_LIMIT_CLIP * desc.mass * GRAVITY * desc.leg_length)
    gains = (drive.stiffness.numpy(), drive.damping.numpy(), drive.effort.numpy())
    rng = np.random.default_rng(7)
    j = len(desc.joint_names)
    q0 = np.asarray(card["joints"]["default_pos"], dtype=np.float32)
    q = q0 + 0.1 * rng.normal(size=(64, j)).astype(np.float32)
    qd = rng.normal(size=(64, j)).astype(np.float32)
    target = q + 0.2 * rng.normal(size=(64, j)).astype(np.float32)
    offline = drive_ratio(target, q, qd, gains, effort_cap=cap)
    as_t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    deployed = drive.drive_ratio(as_t(q), as_t(qd), as_t(target)).numpy()
    assert np.allclose(deployed, offline, atol=1e-5)
    active = drive_gain(gains, effort_cap=cap) > 0
    assert active.sum() == 26
    assert float(offline[:, active].var(0).mean()) > 5e-3
    gain_t = torch.as_tensor(drive_gain(gains, effort_cap=cap))[None]
    off_t = torch.as_tensor(drive_offset(q, qd, q0, gains, effort_cap=cap))
    moved_t = torch.as_tensor(np.broadcast_to(active, (64, j)).copy())
    loss = drive_loss(as_t(target - q0 + 0.05), as_t(target - q0), gain_t, off_t, moved_t)
    assert loss.item() > 1e-4
