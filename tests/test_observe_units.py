"""The tracker's view in the units it trained in (``latent/units.py``)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.model import Normaliser  # noqa: E402
from humanoid_transfer.transfer.latent.units import change_of_units, tracker_view  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIZE = SKILL_LAYOUT.size
LIMIT = 50.0
CONFIGS = [
    "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_track/g1_track_env_cfg.py",
    "source/humanoid_transfer/humanoid_transfer/nao/tasks/nao_track/nao_track_env_cfg.py",
]


def _norm(seed: int) -> Normaliser:
    torch.manual_seed(seed)
    norm = Normaliser(SIZE)
    norm.fit(torch.randn(500, SIZE) * (0.2 + 0.1 * seed) + 0.05 * seed)
    return norm


def _view_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    upcoming = torch.randn(4, 6, SIZE)
    estimate = torch.randn(4, SIZE)
    upcoming[1, 2, 5] = float("nan")  # a missing row is observed as zero
    upcoming[2] *= 100.0  # far out: clamped
    has_plan = torch.tensor([True, True, True, False])
    return upcoming, estimate, has_plan


def test_off_is_the_view_every_earlier_run_observed_bit_for_bit():
    upcoming, estimate, has_plan = _view_inputs()
    # The body of ReferenceCommand.observation in the generator's units, verbatim.
    relative = upcoming - estimate[:, None]
    relative = torch.where(~has_plan[:, None, None], 0.0, relative)
    view = torch.cat([relative.flatten(1), estimate], dim=-1)
    before = torch.nan_to_num(view, nan=0.0).clamp(-LIMIT, LIMIT)
    assert torch.equal(tracker_view(upcoming, estimate, has_plan, LIMIT), before)
    assert torch.equal(tracker_view(upcoming, estimate, has_plan, LIMIT, None), before)


def test_shared_statistics_are_the_identity():
    norm = _norm(1)
    same = Normaliser(SIZE)
    same.load_state_dict(norm.state_dict())
    assert change_of_units(norm, same) is None


def test_the_map_is_the_change_of_normalisation_and_its_inverse_returns_the_input():
    task, own = _norm(1).double(), _norm(2).double()
    scale, shift = change_of_units(task, own)
    x = torch.randn(64, SIZE, dtype=torch.float64)
    assert torch.allclose(scale * x + shift, own(task.inverse(x)), atol=1e-12)
    back_scale, back_shift = change_of_units(own, task)
    assert torch.allclose(back_scale * (scale * x + shift) + back_shift, x, atol=1e-12)


def test_the_view_is_what_the_tracker_would_see_in_its_own_units():
    task, own = _norm(1).double(), _norm(2).double()
    units = change_of_units(task, own)
    upcoming, estimate, has_plan = (
        t.double() if t.is_floating_point() else t for t in _view_inputs()
    )
    upcoming = upcoming.clamp(-5, 5).nan_to_num(0.0)  # inside the clamp, no missing rows
    mapped = tracker_view(upcoming, estimate, has_plan, LIMIT, units)
    expected = tracker_view(
        own(task.inverse(upcoming)), own(task.inverse(estimate)), has_plan, LIMIT
    )
    assert torch.allclose(mapped, expected, atol=1e-10)
    # The relative part carries no shift: only the scale moves it.
    view = upcoming.shape[1] * SIZE
    plain = tracker_view(upcoming, estimate, has_plan, LIMIT)
    scale = units[0].repeat(upcoming.shape[1])
    assert torch.allclose(mapped[:3, :view], plain[:3, :view] * scale, atol=1e-12)
    assert torch.equal(mapped[3, :view], torch.zeros(view, dtype=torch.float64))


def test_the_command_term_observes_through_the_view_and_defaults_to_the_contract():
    source = REPO / "source/humanoid_transfer/humanoid_transfer/transfer/tasks/tracking/command.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cfg = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ReferenceCommandCfg"
    )
    field = next(
        n
        for n in cfg.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "observe_units"
    )
    assert isinstance(field.value, ast.Constant) and field.value.value is None
    term = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ReferenceCommand"
    )
    observation = next(
        n for n in term.body if isinstance(n, ast.FunctionDef) and n.name == "observation"
    )
    called = {
        n.func.id
        for n in ast.walk(observation)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert called == {"tracker_view"}


@pytest.mark.parametrize("cfg", CONFIGS)
def test_both_tasks_read_the_switch_and_pass_it_unset_by_default(cfg):
    text = (REPO / cfg).read_text(encoding="utf-8")
    tree = ast.parse(text)
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "HUMANOID_OBSERVE_UNITS"
    ]
    assert len(reads) == 1 and len(reads[0].args) == 1  # no default: unset is None
    passed = [
        k
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for k in node.keywords
        if k.arg == "observe_units"
    ]
    assert len(passed) == 1 and isinstance(passed[0].value, ast.Name)
    assert passed[0].value.id == "OBSERVE_UNITS"
