"""PACT driving a body: the anchors computed online match the offline gates', and the
driver's history is the windows the policy was trained on."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.driver import PactDriver, anchor_inputs  # noqa: E402
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import PactPolicy, plan_from_view  # noqa: E402

REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))


def _card(name: str) -> dict:
    return json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))


def _state(card: dict, n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    j = len(card["joints"]["names"])
    q0 = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32)
    q = q0 + 0.1 * torch.randn(n, j, generator=gen)
    qd = torch.randn(n, j, generator=gen)
    g = torch.tensor([0.0, 0.0, -1.0]) + 0.1 * torch.randn(n, 3, generator=gen)
    g = g / g.norm(dim=-1, keepdim=True)
    omega = 0.5 * torch.randn(n, 3, generator=gen)
    target = q + 0.05 * torch.randn(n, j, generator=gen)
    return q0, q, qd, g, omega, target


@pytest.mark.parametrize("body", ["g1", "nao"])
def test_anchor_inputs_match_the_offline_gates(body):
    from pact_gate_a2 import analytic_z
    from pact_gate_a3 import physics_bases, with_physics

    card = _card(body)
    unit = REGISTRY[body].get("leg_length_unit")
    descriptor = describe(card, leg_length=unit)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
    _, q, qd, g, omega, target = _state(card, 64, seed=1)
    w = SimpleNamespace(
        joint_obs=torch.stack([q, qd * time_unit], -1)[:, :, None].numpy(),
        imu=torch.cat([g, omega * time_unit], -1)[:, None].numpy(),
        z=np.zeros((64, 27), np.float32),
        extra={"joint_pos_target": target.numpy()},
    )
    analytic = analytic_z(anchors, w, time_unit)
    offline = with_physics(analytic, physics_bases(drive, w, analytic, time_unit))
    with torch.no_grad():
        online = anchor_inputs(anchors, drive, q, qd, g, omega, target).numpy()
    np.testing.assert_allclose(online, offline, atol=1e-4, rtol=1e-4)


def test_the_driver_reads_the_windows_the_policy_was_trained_on():
    card = _card("nao")
    unit = REGISTRY["nao"].get("leg_length_unit")
    descriptor = describe(card, leg_length=unit)
    contacts = contact_joints(card)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    dims = PactDims(
        width=32, heads=4, history=3, free_tokens=2, body_tokens=2, observation=3,
        anchor_inputs=True, timed_plan=True,
    )  # fmt: skip
    torch.manual_seed(0)
    policy = PactPolicy(dims).eval()
    gen = torch.Generator().manual_seed(2)
    checkpoint = {
        "state": policy.state_dict(),
        "dims": vars(dims),
        "z_mean": torch.randn(27, generator=gen).numpy(),
        "z_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
        "a_mean": torch.randn(27, generator=gen).numpy(),
        "a_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
    }
    view, n, control_dt = 4, 3, 0.02
    skill_mean, skill_std = torch.randn(27, generator=gen), torch.rand(27, generator=gen) + 0.5
    action_joints = [5, 2, 7, 11]
    driver = PactDriver(
        checkpoint, descriptor, contacts, anchors, drive,
        torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32),
        action_joints, 0.25, view, skill_mean, skill_std, control_dt, n, "cpu",
    )  # fmt: skip
    steps = [_state(card, n, seed=10 + k) for k in range(5)]
    obs = [torch.randn(n, 70 + view * 30 + 27, generator=gen) for _ in steps]
    actions = [driver(o, q, qd, t, g, w) for o, (_, q, qd, g, w, t) in zip(obs, steps, strict=True)]

    # The last action, rebuilt as the offline gate batches it: the last 3 steps, oldest first.
    q0, *_ = steps[0]
    time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
    last = steps[-dims.history :]
    joint_obs = torch.stack(
        [torch.stack([q - q0, qd * time_unit, drive.drive_ratio(q, qd, t)], -1) for _, q, qd, _, _, t in last], dim=2
    )
    imu = torch.stack([torch.cat([g, w * time_unit], -1) for _, _, _, g, w, _ in last], dim=1)
    _, q, qd, g, w, t = steps[-1]
    with torch.no_grad():
        anchor = anchor_inputs(anchors, drive, q, qd, g, w, t)
    up, est, path = plan_from_view(obs[-1][:, -(view * 30 + 27) :], view, skill_mean, skill_std)
    z_mean, z_std = (torch.as_tensor(checkpoint[k]) for k in ("z_mean", "z_std"))
    a_mean, a_std = (torch.as_tensor(checkpoint[k]) for k in ("a_mean", "a_std"))
    with torch.no_grad():
        _, out = policy(
            stack_bodies([descriptor], [contacts]).take(torch.zeros(n, dtype=torch.long)),
            joint_obs,
            imu,
            (anchor - a_mean) / a_std,
            (up - z_mean) / z_std,
            (up - est[:, None]) / z_std,
            (torch.arange(1, view + 1) * control_dt / time_unit).expand(n, -1),
            path,
            torch.ones(n, view, dtype=torch.bool),
        )
    torch.testing.assert_close(actions[-1], out[:, action_joints] / 0.25, atol=1e-5, rtol=1e-4)
    assert actions[0].shape == (n, len(action_joints))


def test_a_reset_refills_the_history_with_the_current_reading():
    card = _card("h1")
    descriptor = describe(card)
    anchors = KinematicAnchors(card)
    dims = PactDims(
        width=32, heads=4, history=3, free_tokens=2, body_tokens=2, observation=3,
        anchor_inputs=True, timed_plan=True,
    )  # fmt: skip
    policy = PactPolicy(dims).eval()
    zeros, ones = np.zeros(27, np.float32), np.ones(27, np.float32)
    checkpoint = {"state": policy.state_dict(), "dims": vars(dims), "z_mean": zeros,
                  "z_std": ones, "a_mean": zeros, "a_std": ones}  # fmt: skip
    driver = PactDriver(
        checkpoint, descriptor, contact_joints(card), anchors, DriveLoads(card, anchors),
        torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32), slice(None), 0.5,
        9, torch.zeros(27), torch.ones(27), 0.02, 2, "cpu",
    )  # fmt: skip
    obs = torch.zeros(2, 63 + 9 * 30 + 27)
    for seed in range(4):
        _, q, qd, g, w, t = _state(card, 2, seed)
        driver(obs, q, qd, t, g, w)
    driver.reset(torch.tensor([1]))
    _, q, qd, g, w, t = _state(card, 2, 9)
    driver(obs, q, qd, t, g, w)
    history = driver.joint_history
    assert torch.equal(history[1, :, 0], history[1, :, -1])
    assert not torch.equal(history[0, :, 0], history[0, :, -1])


@pytest.mark.parametrize("body", ["nao", "h1", "digit"])
@pytest.mark.parametrize("state_anchor", [False, True])
def test_impedance_anchor_driver_matches_offline_policy_and_pd_inversion(body, state_anchor):
    import math

    card = _card(body)
    unit = REGISTRY[body].get("leg_length_unit")
    descriptor = describe(card, leg_length=unit)
    contacts = contact_joints(card)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    dims = PactDims(
        width=32, heads=4, history=3, free_tokens=2, body_tokens=2, observation=3,
        anchor_inputs=True, timed_plan=True, impedance_anchor=True, op_width=48,
        state_anchor=state_anchor, leg_cap=2.0, arm_cap=3.0,
    )  # fmt: skip
    torch.manual_seed(3)
    policy = PactPolicy(dims).eval()
    gen = torch.Generator().manual_seed(4)
    checkpoint = {
        "state": policy.state_dict(),
        "dims": vars(dims),
        "z_mean": torch.randn(27, generator=gen).numpy(),
        "z_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
        "a_mean": torch.randn(27, generator=gen).numpy(),
        "a_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
    }
    view, n, control_dt = 4, 3, 0.02
    skill_mean, skill_std = torch.randn(27, generator=gen), torch.rand(27, generator=gen) + 0.5
    action_joints = [0, 2, 4, 6]
    # Shift env default pose away from the card's default_pos (as H1Track-Play-v0 does):
    # targets_from_torque must subtract the env default pose q0, never the card's default.
    q0 = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32) + 0.35
    driver = PactDriver(
        checkpoint, descriptor, contacts, anchors, drive, q0,
        action_joints, 0.25, view, skill_mean, skill_std, control_dt, n, "cpu",
    )  # fmt: skip
    steps = [_state(card, n, seed=20 + k) for k in range(4)]
    obs = [torch.randn(n, 70 + view * 30 + 27, generator=gen) for _ in steps]
    actions = [driver(o, q, qd, t, g, w) for o, (_, q, qd, g, w, t) in zip(obs, steps, strict=True)]
    assert bool(torch.isfinite(actions[-1]).all())

    time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
    last = steps[-dims.history :]
    joint_obs = torch.stack(
        [torch.stack([q - q0, qd * time_unit, drive.drive_ratio(q, qd, t)], -1) for _, q, qd, _, _, t in last], dim=2
    )
    imu = torch.stack([torch.cat([g, w * time_unit], -1) for _, _, _, g, w, _ in last], dim=1)
    _, q, qd, g, w, t = steps[-1]
    with torch.no_grad():
        anchor = anchor_inputs(anchors, drive, q, qd, g, w, t)
        phys = drive.operational_anchor(q, qd, g, t)
        phys["alpha_dt"] = q.new_full((n,), 1.0 - math.exp(-(control_dt / time_unit) / 0.08))
        up, est, path = plan_from_view(obs[-1][:, -(view * 30 + 27) :], view, skill_mean, skill_std)
        z_mean, z_std = (torch.as_tensor(checkpoint[k]) for k in ("z_mean", "z_std"))
        a_mean, a_std = (torch.as_tensor(checkpoint[k]) for k in ("a_mean", "a_std"))
        _, out_tau = policy(
            stack_bodies([descriptor], [contacts]).take(torch.zeros(n, dtype=torch.long)),
            joint_obs, imu, (anchor - a_mean) / a_std, (up - z_mean) / z_std,
            (up - est[:, None]) / z_std,
            (torch.arange(1, view + 1) * control_dt / time_unit).expand(n, -1),
            path, torch.ones(n, view, dtype=torch.bool), phys=phys,
        )
        out_rad = drive.targets_from_torque(q, qd, out_tau, default_pos=q0)
    torch.testing.assert_close(actions[-1], out_rad[:, action_joints] / 0.25, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("body", ["g1", "nao", "h1", "cassie", "digit"])
@pytest.mark.parametrize("mode", ["radian", "imp_v20", "imp_v21_state"])
def test_symmetrized_driver_is_exact_equivariant(body, mode):
    from humanoid_transfer.transfer.latent.pact.mirror import SampleMirror

    card = _card(body)
    unit = REGISTRY[body].get("leg_length_unit")
    descriptor = describe(card, leg_length=unit)
    contacts = contact_joints(card)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    dims = PactDims(
        width=32, heads=4, history=3, free_tokens=2, body_tokens=2, observation=3,
        anchor_inputs=True, timed_plan=True,
        impedance_anchor=(mode != "radian"),
        state_anchor=(mode == "imp_v21_state"),
        leg_cap=2.0 if mode == "imp_v21_state" else 0.8,
        arm_cap=3.0 if mode == "imp_v21_state" else 1.5,
        op_width=48,
    )  # fmt: skip
    torch.manual_seed(7)
    policy = PactPolicy(dims).eval()
    gen = torch.Generator().manual_seed(8)
    checkpoint = {
        "state": policy.state_dict(),
        "dims": vars(dims),
        "z_mean": torch.randn(27, generator=gen).numpy(),
        "z_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
        "a_mean": torch.randn(27, generator=gen).numpy(),
        "a_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
    }
    view, n, control_dt = 4, 2, 0.02
    skill_mean, skill_std = torch.randn(27, generator=gen), torch.rand(27, generator=gen) + 0.5
    q0_np = np.asarray(card["joints"]["default_pos"], dtype=np.float32)
    q0 = torch.as_tensor(q0_np, dtype=torch.float32)
    mirror = SampleMirror.for_body(REPO, body, descriptor.joint_names, q0_np)
    spec_path = (
        REPO / "transfer/bodies" / body / "mirror.json"
        if (REPO / "transfer/bodies" / body / "mirror.json").exists()
        else REPO / body / "body/mirror.json"
    )
    spec = json.loads(spec_path.read_text("utf-8"))
    jnames = list(descriptor.joint_names)
    d1 = PactDriver(
        checkpoint, descriptor, contacts, anchors, drive, q0, slice(None), 1.0,
        view, skill_mean, skill_std, control_dt, n, "cpu", mirror=mirror,
    )
    d2 = PactDriver(
        checkpoint, descriptor, contacts, anchors, drive, q0, slice(None), 1.0,
        view, skill_mean, skill_std, control_dt, n, "cpu", mirror=mirror,
    )
    smean_np, sstd_np = skill_mean.numpy(), skill_std.numpy()
    w_tail = view * 30 + 27
    for step_idx in range(4):
        _, q, qd, g, w, t = _state(card, n, seed=50 + step_idx)
        for follower, (leader, mult) in spec.get("mimic", {}).items():
            i_f, i_l = jnames.index(follower), jnames.index(leader)
            q[:, i_f] = float(mult) * q[:, i_l]
            qd[:, i_f] = float(mult) * qd[:, i_l]
            t[:, i_f] = float(mult) * t[:, i_l]
        obs = torch.randn(n, 40 + w_tail, generator=gen)
        # Mirror policy_obs plan tail exactly
        obs_m = obs.clone().numpy()
        tail = obs_m[:, -w_tail:]
        rel_n = tail[:, : view * 27].reshape(n, view, 27)
        est_n = tail[:, view * 27 : (view + 1) * 27]
        path = tail[:, (view + 1) * 27 :].reshape(n, view, 3)
        est_m_n = (mirror.skill(est_n * sstd_np + smean_np) - smean_np) / sstd_np
        up_m_n = (mirror.skill((rel_n + est_n[:, None, :]) * sstd_np + smean_np) - smean_np) / sstd_np
        obs_m[:, -w_tail:] = np.concatenate(
            [(up_m_n - est_m_n[:, None, :]).reshape(n, -1), est_m_n, mirror.path(path).reshape(n, -1)],
            axis=-1,
        )
        out1 = d1(obs, q, qd, t, g, w)
        qm = torch.as_tensor(q0_np + mirror.positions(q.numpy() - q0_np))
        qdm = torch.as_tensor(mirror.speeds(qd.numpy()))
        tm = torch.as_tensor(q0_np + mirror.positions(t.numpy() - q0_np))
        imu_m = mirror.imu(torch.cat([g, w], dim=-1).numpy())
        out2 = d2(
            torch.as_tensor(obs_m), qm, qdm, tm,
            torch.as_tensor(imu_m[:, :3]), torch.as_tensor(imu_m[:, 3:]),
        )
    # Check equivariance on driven joints: out2 == M_drive(out1)
    driven = (drive.stiffness > 0) & (drive.effective_effort() > 0)
    out1_m = torch.as_tensor(mirror.drive_positions(out1.numpy()))
    gap = float((out2[:, driven] - out1_m[:, driven]).abs().max())
    assert gap < (1e-2 if body == "digit" else 1e-3), (body, impedance_anchor, gap)
