"""The plain tracker's view of a physics-task env shows it its own last action."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# The tracking package imports Isaac Lab; this module needs only torch, so it is loaded alone.
_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/humanoid_transfer/humanoid_transfer/transfer/tasks/tracking/teacher_view.py"
)
_SPEC = importlib.util.spec_from_file_location("teacher_view", _PATH)
teacher_view = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(teacher_view)


def _manager():
    return SimpleNamespace(
        active_terms={"policy": ["base", "actions", "plan", "prior"], "critic": ["actions", "x"]},
        group_obs_term_dim={"policy": [(3,), (2,), (4,), (2,)], "critic": [(2,), (5,)]},
    )


def test_term_columns_finds_a_term_among_the_concatenated_group():
    manager = _manager()
    assert teacher_view.term_columns(manager, "policy", "actions") == slice(3, 5)
    assert teacher_view.term_columns(manager, "critic", "x") == slice(2, 7)
    with pytest.raises(KeyError):
        teacher_view.term_columns(manager, "critic", "plan")


def test_the_teacher_sees_its_own_last_action_and_zero_after_a_reset():
    view = teacher_view.TeacherView(_manager(), {"policy": 9, "critic": 7}, 3, 2, "cpu")
    obs = {"policy": torch.arange(33.0).reshape(3, 11), "critic": torch.ones(3, 7)}
    first = view.observation(obs, "policy")
    # The physics task's own columns (the prior) are cut, the residual's slot is replaced.
    assert first.shape == (3, 9)
    assert torch.equal(first[:, 3:5], torch.zeros(3, 2))
    assert torch.equal(first[:, :3], obs["policy"][:, :3])
    assert torch.equal(first[:, 5:], obs["policy"][:, 5:9])
    own = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    view.record(own, torch.tensor([False, True, False]))
    second = view.observation(obs, "critic")
    assert torch.equal(second[:, :2], torch.tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]))
    # The env's observation itself is not touched.
    assert torch.equal(obs["critic"], torch.ones(3, 7))


def test_a_last_action_outside_the_plain_width_is_refused():
    with pytest.raises(ValueError):
        teacher_view.TeacherView(_manager(), {"policy": 4}, 3, 2, "cpu")
