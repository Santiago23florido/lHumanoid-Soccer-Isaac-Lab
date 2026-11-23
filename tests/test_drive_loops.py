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
