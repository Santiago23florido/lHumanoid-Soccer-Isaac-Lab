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
BODIES = ("g1", "nao", "h1", "cassie", "digit")


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


@pytest.mark.parametrize("body", BODIES)
def test_reflected_drive_ratio_and_offset_match_mirrored_state(body):
    import sys

    sys.path.insert(0, str(REPO / "transfer" / "scripts"))
    from humanoid_transfer.transfer.latent.pact.descriptors import EFFORT_LIMIT_CLIP, GRAVITY
    from pact_gate_d import drive_gain, drive_offset, drive_ratio

    card, unit, q0, m = _body(body)
    desc = describe(card, leg_length=unit)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    cap = float(EFFORT_LIMIT_CLIP * desc.mass * GRAVITY * desc.leg_length)
    gains = (drive.stiffness.numpy(), drive.damping.numpy(), drive.effort.numpy())
    rng = np.random.default_rng(4)
    n, j = 32, len(q0)
    q = q0 + 0.1 * rng.normal(size=(n, j)).astype(np.float32)
    qd = rng.normal(size=(n, j)).astype(np.float32)
    target = q + 0.05 * rng.normal(size=(n, j)).astype(np.float32)
    q_m = q0 + m.positions(q - q0)
    qd_m = m.speeds(qd)
    t_m = q0 + m.positions(target - q0)
    r = drive_ratio(target, q, qd, gains, cap)
    r_m = drive_ratio(t_m, q_m, qd_m, gains, cap)
    gain = drive_gain(gains, cap)
    # Exclude asymmetric mimic pairs (e.g. NAO's LHipYawPitch is driven while RHipYawPitch is a mimic).
    sym = np.isclose(gain, np.abs(m.joints) @ gain, atol=1e-5)
    assert sym.sum() >= 12
    np.testing.assert_allclose(r_m[:, sym], m.speeds(r)[:, sym], atol=1e-4)
    off = drive_offset(q, qd, q0, gains, cap)
    off_m = drive_offset(q_m, qd_m, q0, gains, cap)
    np.testing.assert_allclose(off_m[:, sym], (m.speeds(off) - gain * m.offset)[:, sym], atol=1e-4)


@pytest.mark.parametrize("body", ["g1", "h1", "digit"])
def test_reflected_operational_anchor_matches_mirrored_state(body):
    card, unit, q0, m = _body(body)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    rng = np.random.default_rng(13)
    n, j = 8, len(q0)
    q = q0 + 0.1 * rng.normal(size=(n, j)).astype(np.float32)
    qd = rng.normal(size=(n, j)).astype(np.float32)
    target = q + 0.05 * rng.normal(size=(n, j)).astype(np.float32)
    g = np.tile(np.array([0.08, -0.06, -0.995], np.float32), (n, 1))
    g_m = g * np.array([1.0, -1.0, 1.0], np.float32)
    q_m = q0 + m.positions(q - q0)
    qd_m = m.speeds(qd)
    t_m = q0 + m.positions(target - q0)
    with torch.no_grad():
        op = drive.operational_anchor(torch.as_tensor(q), torch.as_tensor(qd), torch.as_tensor(g), torch.as_tensor(target))
        op_m = drive.operational_anchor(torch.as_tensor(q_m), torch.as_tensor(qd_m), torch.as_tensor(g_m), torch.as_tensor(t_m))
    np.testing.assert_allclose(op_m["tau_g"].numpy(), m.speeds(op["tau_g"].numpy()), atol=1e-3)
    np.testing.assert_allclose(op_m["j_eff"].numpy(), m.jacobian(op["j_eff"].numpy()), atol=1e-3)
    np.testing.assert_allclose(op_m["j_pinv"].numpy(), m.jacobian(op["j_pinv"].numpy()), atol=1e-3)
    np.testing.assert_allclose(op_m["w_t"].numpy(), m.wrench(op["w_t"].numpy()), atol=1e-3)
