"""A plan's loaded feet put on the ground, and held still (``latent/plan_contact.py``)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.contact import foot_height  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.model import Normaliser  # noqa: E402
from humanoid_transfer.transfer.latent.plan_contact import (  # noqa: E402
    apply_plan_contact,
    ground_loaded_feet,
    stick_loaded_feet,
)

REPO = Path(__file__).resolve().parents[1]
LOADS = SKILL_LAYOUT.slice_of("foot_load")
FEET = SKILL_LAYOUT.slice_of("foot_position")
HEIGHT = SKILL_LAYOUT.slice_of("height")
VELOCITY = SKILL_LAYOUT.slice_of("foot_velocity")


def _norm() -> Normaliser:
    torch.manual_seed(0)
    norm = Normaliser(SKILL_LAYOUT.size)
    norm.fit(torch.randn(500, SKILL_LAYOUT.size) * 0.3 + 0.1)
    return norm


def _plan() -> torch.Tensor:
    """Two envs × 5 rows: the left foot loaded in rows 0-2 but held above the ground."""
    z = torch.randn(2, 5, SKILL_LAYOUT.size, dtype=torch.float64) * 0.05
    z[..., HEIGHT] = 0.9
    z[..., FEET.start + 2] = -0.88  # left sole 0.02 L up
    z[..., FEET.start + 5] = -0.85  # right sole 0.05 L up
    z[..., LOADS.start] = torch.tensor([0.5, 0.2, 0.06, 0.04, 0.0], dtype=torch.float64)
    z[..., LOADS.start + 1] = 0.0
    return z


def test_loaded_feet_rest_on_the_ground_and_nothing_else_moves():
    norm = _norm().double()
    physical = _plan()
    rows = norm(physical)
    out = norm.inverse(ground_loaded_feet(rows, norm))
    sole = foot_height(out.numpy())
    assert sole[..., :3, 0] == pytest.approx(0.0, abs=1e-12)  # loaded above 5 %: grounded
    assert sole[..., 3:, 0] == pytest.approx(0.02, abs=1e-12)  # 4 % and unloaded: kept
    assert sole[..., 1] == pytest.approx(0.05, abs=1e-12)  # the unloaded right foot: kept
    left_z = FEET.start + 2
    changed = torch.zeros(SKILL_LAYOUT.size, dtype=torch.bool)
    changed[left_z] = True
    grounded = ground_loaded_feet(rows, norm)
    assert torch.equal(grounded[..., ~changed], rows[..., ~changed])  # bit for bit
    assert torch.equal(grounded[..., 3:, left_z], rows[..., 3:, left_z])


def test_a_consistent_plan_is_unchanged():
    norm = _norm().double()
    physical = _plan()
    physical[..., FEET.start + 2] = -0.9  # the left sole on the ground where loaded
    rows = norm(physical)
    grounded = ground_loaded_feet(rows, norm)
    assert torch.allclose(grounded, rows, atol=1e-12)


@pytest.mark.parametrize(
    "cfg",
    [
        "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_track/g1_track_env_cfg.py",
        "source/humanoid_transfer/humanoid_transfer/nao/tasks/nao_track/nao_track_env_cfg.py",
    ],
)
def test_both_tasks_read_the_switch_with_the_plan_as_drawn_by_default(cfg):
    tree = ast.parse((REPO / cfg).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "HUMANOID_PLAN_CONTACT"
    ]
    assert len(calls) == 1 and calls[0].args[1].value == "as_drawn"


def test_stick_grounds_and_stills_loaded_feet_and_nothing_else_moves():
    norm = _norm().double()
    physical = _plan()
    physical[..., VELOCITY] = 0.1  # every foot moving, loaded or not
    rows = norm(physical)
    stuck = stick_loaded_feet(rows, norm)
    out = norm.inverse(stuck)
    left_velocity = slice(VELOCITY.start, VELOCITY.start + 3)
    assert out[..., :3, left_velocity] == pytest.approx(0.0, abs=1e-12)  # loaded: still
    assert foot_height(out.numpy())[..., :3, 0] == pytest.approx(0.0, abs=1e-12)  # and grounded
    grounded = ground_loaded_feet(rows, norm)
    assert torch.equal(stuck[..., 3:, :], grounded[..., 3:, :])  # unloaded rows: as grounding
    changed = torch.zeros(SKILL_LAYOUT.size, dtype=torch.bool)
    changed[FEET.start + 2] = True
    changed[left_velocity] = True
    assert torch.equal(stuck[..., ~changed], rows[..., ~changed])  # bit for bit
    right_velocity = slice(VELOCITY.start + 3, VELOCITY.start + 6)
    assert torch.equal(stuck[..., right_velocity], rows[..., right_velocity])  # never loaded


def test_stick_is_idempotent():
    norm = _norm().double()
    physical = _plan()
    physical[..., VELOCITY] = 0.1
    once = stick_loaded_feet(norm(physical), norm)
    assert torch.allclose(stick_loaded_feet(once, norm), once, atol=1e-12)


def test_the_modes_dispatch_and_as_drawn_is_the_plan_itself():
    norm = _norm().double()
    rows = norm(_plan())
    assert apply_plan_contact(rows, norm, "as_drawn") is rows
    assert torch.equal(apply_plan_contact(rows, norm, "ground"), ground_loaded_feet(rows, norm))
    assert torch.equal(apply_plan_contact(rows, norm, "stick"), stick_loaded_feet(rows, norm))
    with pytest.raises(ValueError):
        apply_plan_contact(rows, norm, "slide")
