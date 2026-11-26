"""Drive loads with loop closures: the loops' structure, and the tree bodies left as they were."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors  # noqa: E402


def _drive(body: str) -> tuple[DriveLoads, dict]:
    card = json.loads((REPO / f"transfer/bodies/{body}/body_card.json").read_text("utf-8"))
    anchors = KinematicAnchors(card)
    return DriveLoads(card, anchors), card


def test_a_tree_body_has_no_loops():
    drive, _ = _drive("g1")
    assert drive.loop_sign.shape[1] == 0 and not bool(drive.on_loop.any())


def test_each_loop_is_carried_by_the_joints_between_its_two_links():
    drive, card = _drive("digit")
    names = card["joints"]["names"]
    assert drive.loop_sign.shape[1] == len(card["loops"]) == 6
    on_loop = {names[j] for j in torch.nonzero(drive.on_loop)[:, 0].tolist()}
    # The achilles loop closes rod to tarsus: the rod's own joints and the knee and tarsus.
    for side in ("left", "right"):
        assert {f"{side}_leg_knee", f"{side}_leg_tarsus", f"{side}_leg_achilles_rod:0"} <= on_loop
        assert f"{side}_leg_hip_roll" not in on_loop  # both of a loop's links move with it


def test_a_closed_leg_solves_both_feet_and_returns_their_loads():
    drive, card = _drive("digit")
    n, j = 4, len(card["joints"]["names"])
    q = torch.zeros(n, j)
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, 3)
    out = drive(q, torch.zeros(n, j), gravity, target=q)
    assert out["loads"].shape == (n, 2) and bool(torch.isfinite(out["loads"]).all())


@pytest.mark.parametrize("body", ["g1", "nao", "h1", "cassie", "digit"])
def test_wrench_anchor_is_finite_and_dual_to_forward_loads(body):
    drive, card = _drive(body)
    n, j = 6, len(card["joints"]["names"])
    gen = torch.Generator().manual_seed(7)
    q = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32)[None, :] + 0.15 * torch.randn(
        n, j, generator=gen
    )
    gravity = torch.tensor([[0.08, -0.05, -0.995]]).expand(n, 3)
    tau_g, j_eff = drive.wrench_anchor(q, gravity)
    assert tau_g.shape == (n, j) and j_eff.shape == (n, j, 12)
    assert bool(torch.isfinite(tau_g).all()) and bool(torch.isfinite(j_eff).all())
    # Physical duality: torques synthesized from a vertical foot force w = [0, 0, Fz, 0, 0, 0]
    # must invert back to the same foot loads through DriveLoads forward static balance.
    w = torch.zeros(n, 12)
    w[:, 2] = torch.linspace(0.2, 0.8, n)
    w[:, 8] = 1.0 - w[:, 2]
    tau_dimless = -(tau_g + (j_eff @ w[..., None])[..., 0])
    mg_l = float(9.81 * drive.anchors.mass.sum() * drive.anchors.units.length)
    tau_nm = tau_dimless * mg_l
    # Construct PD target that commands exact tau_nm on driven joints (with high cap to avoid clipping)
    driven = drive.stiffness > 0
    drive.effort.fill_(1e9)
    target = torch.where(driven, q + tau_nm / drive.stiffness.clamp_min(1.0), q)
    out = drive(q, torch.zeros(n, j), gravity, target=target)
    assert torch.allclose(out["loads"][:, 0], w[:, 2], atol=2e-2)
    assert torch.allclose(out["loads"][:, 1], w[:, 8], atol=2e-2)


@pytest.mark.parametrize("body", ["g1", "digit"])
def test_wrench_anchor_is_scale_invariant(body):
    from tests.test_pact_descriptors import _scaled

    card = json.loads((REPO / f"transfer/bodies/{body}/body_card.json").read_text("utf-8"))
    s, k = 1.7, 2.3
    scaled = _scaled(card, s, k)
    for loop in scaled.get("loops", ()):
        for key in ("pos0", "pos1"):
            loop[key] = [x * s for x in loop[key]]
    d0 = DriveLoads(card, KinematicAnchors(card))
    d1 = DriveLoads(scaled, KinematicAnchors(scaled))
    gen = torch.Generator().manual_seed(11)
    q = 0.2 * torch.randn(4, len(card["joints"]["names"]), generator=gen)
    g = torch.tensor([[0.1, 0.05, -0.99]]).expand(4, 3)
    tg0, je0 = d0.wrench_anchor(q, g)
    tg1, je1 = d1.wrench_anchor(q, g)
    assert torch.allclose(tg0, tg1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(je0, je1, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("body", ["g1", "digit"])
def test_wrench_anchor_is_mirror_equivariant(body):
    import numpy as np
    from humanoid_transfer.transfer.latent.pact.mirror import SampleMirror

    drive, card = _drive(body)
    q0 = np.asarray(card["joints"]["default_pos"], dtype=np.float32)
    m = SampleMirror.for_body(REPO, body, card["joints"]["names"], q0)
    gen = torch.Generator().manual_seed(19)
    n, j = 4, len(card["joints"]["names"])
    q = torch.as_tensor(q0, dtype=torch.float32)[None, :] + 0.15 * torch.randn(n, j, generator=gen)
    g = torch.tensor([[0.09, -0.07, -0.993]]).expand(n, 3)
    q_m = torch.as_tensor(m.positions(q.numpy()), dtype=torch.float32)
    g_m = g * torch.tensor([1.0, -1.0, 1.0])
    tg, je = drive.wrench_anchor(q, g)
    tg_m, je_m = drive.wrench_anchor(q_m, g_m)
    assert torch.allclose(tg_m, torch.as_tensor(m.speeds(tg.numpy()), dtype=torch.float32), atol=1e-3)
    w = torch.randn(n, 12, generator=gen)
    sign_f = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    w_m = torch.cat([w[:, 6:] * sign_f, w[:, :6] * sign_f], dim=-1)
    tau_w = (je @ w[..., None])[..., 0]
    tau_w_m = (je_m @ w_m[..., None])[..., 0]
    assert torch.allclose(tau_w_m, torch.as_tensor(m.speeds(tau_w.numpy()), dtype=torch.float32), atol=1e-3)
