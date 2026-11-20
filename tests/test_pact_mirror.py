"""PACT's mirror augmentation: an involution, and the reflected joint state's anchors are
the reflected anchors (the measured mirror agrees with the body's kinematics)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.descriptors import describe  # noqa: E402
from humanoid_transfer.transfer.latent.pact.driver import anchor_inputs  # noqa: E402
from humanoid_transfer.transfer.latent.pact.mirror import SampleMirror  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
BODIES = ("g1", "nao", "h1", "cassie")


def _body(name: str):
    card = json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))
    unit = REGISTRY[name].get("leg_length_unit")
    q0 = np.asarray(card["joints"]["default_pos"], dtype=np.float32)
    mirror = SampleMirror.for_body(REPO, name, describe(card, leg_length=unit).joint_names, q0)
    return card, unit, q0, mirror


@pytest.mark.parametrize("body", BODIES)
def test_the_mirror_is_an_involution(body):
    _, _, q0, m = _body(body)
    rng = np.random.default_rng(0)
    j = len(q0)
    obs = rng.normal(size=(5, j, 4, 3)).astype(np.float32)
    np.testing.assert_allclose(m.joint_obs(m.joint_obs(obs)), obs, atol=1e-5)
    for fn, width in ((m.imu, 6), (m.skill, 27), (m.path, 3), (m.learned, 6)):
        x = rng.normal(size=(5, 2, width)).astype(np.float32)
        np.testing.assert_allclose(fn(fn(x)), x, atol=1e-5)


@pytest.mark.parametrize("body", BODIES)
def test_reflected_joint_states_give_the_reflected_anchors(body):
    card, unit, q0, m = _body(body)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    rng = np.random.default_rng(1)
    n, j = 64, len(q0)
    q = q0 + 0.1 * rng.normal(size=(n, j)).astype(np.float32)
    qd = rng.normal(size=(n, j)).astype(np.float32)
    target = q + 0.05 * rng.normal(size=(n, j)).astype(np.float32)
    g = np.array([0.0, 0.0, -1.0]) + 0.1 * rng.normal(size=(n, 3))
    g = (g / np.linalg.norm(g, axis=1, keepdims=True)).astype(np.float32)
    omega = (0.5 * rng.normal(size=(n, 3))).astype(np.float32)
    imu = np.concatenate([g, omega], axis=1)
    reflected = m.imu(imu)
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32))  # noqa: E731
    with torch.no_grad():
        a = anchor_inputs(anchors, drive, t(q), t(qd), t(g), t(omega), t(target)).numpy()
        b = anchor_inputs(
            anchors, drive,
            t(q0 + m.positions(q - q0)), t(m.speeds(qd)), t(reflected[:, :3]), t(reflected[:, 3:]),
            t(q0 + m.positions(target - q0)),
        ).numpy()  # fmt: skip
    error = np.abs(b - m.skill(a))
    scale = np.abs(a).mean(0) + 1e-3
    assert np.median(error / scale) < 0.02, (body, np.round(np.max(error / scale, 0), 3))
