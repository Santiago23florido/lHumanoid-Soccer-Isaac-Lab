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
