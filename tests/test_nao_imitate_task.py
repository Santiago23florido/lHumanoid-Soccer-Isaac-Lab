"""Checks on the joint-target baseline's task (``nao/tasks/nao_imitate``).

The command and environment modules import Isaac Lab, so the timing is read from the
source, as ``test_nao_stand_task.py`` does. Isaac Lab scores a step after the physics
and before the command advances: a reward written against the reference's current
row would hold the body to where it was, one step behind, and still train.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

TASK = ROOT / "source" / "humanoid_transfer" / "humanoid_transfer" / "nao" / "tasks"
REFERENCE = (TASK / "nao_imitate" / "reference.py").read_text(encoding="utf-8")


def test_the_imitation_tasks_are_registered() -> None:
    gym = pytest.importorskip("gymnasium")
    from humanoid_transfer.nao.tasks import nao_imitate

    assert nao_imitate.REGISTERED
    assert nao_imitate.TASK_ID in gym.envs.registry
    assert nao_imitate.PLAY_TASK_ID in gym.envs.registry


@pytest.mark.parametrize("name", ["pose_tracking", "rate_tracking"])
def test_rewards_score_the_row_one_step_ahead(name: str) -> None:
    body = re.search(rf"def {name}\(.*?(?=\ndef |\Z)", REFERENCE, re.S).group(0)
    assert re.search(r"term\.target(_rate)?\(1\)", body), body


def test_a_reset_puts_the_body_one_row_ahead_of_the_frame() -> None:
    # The step that resets an environment advances its command once more.
    assert "self.frame[ids] = first - 1" in REFERENCE
    assert "q, qd = self.q[first], self.qd[first]" in REFERENCE
