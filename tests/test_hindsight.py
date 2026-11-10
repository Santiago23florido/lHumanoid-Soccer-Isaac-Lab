"""Hindsight commands: the command a future achieved (``latent/hindsight.py``, W17)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.hindsight import (  # noqa: E402
    achieved,
    hindsight_commands,
    relabel,
)
from humanoid_transfer.transfer.latent.sequences import Streams  # noqa: E402

SIZE = SKILL_LAYOUT.size
VELOCITY = SKILL_LAYOUT.slice_of("com_velocity").start
YAW = SKILL_LAYOUT.slice_of("base_angular_velocity").start + 2


def _streams(skill: torch.Tensor, episodes: list[tuple[int, int]]) -> Streams:
    total = len(skill)
    start = torch.zeros(total, dtype=torch.long)
    end = torch.zeros(total, dtype=torch.long)
    for first, last in episodes:
        start[first : last + 1] = first
        end[first : last + 1] = last
    return Streams(
        obs=torch.zeros(total, 0),
        skill=skill,
        command=torch.full((total, 3), 9.0),
        action=torch.zeros(total, 0),
        falling=torch.zeros(total, dtype=torch.bool),
        start=start,
        end=end,
        capture=torch.zeros(total, dtype=torch.long),
        env=torch.zeros(total, dtype=torch.long),
    )


def test_achieved_reads_forward_and_lateral_velocity_and_the_yaw_rate():
    z = torch.zeros(2, SIZE)
    z[:, VELOCITY], z[:, VELOCITY + 1], z[:, YAW] = 0.3, -0.1, 0.2
    assert torch.allclose(achieved(z), torch.tensor([[0.3, -0.1, 0.2]] * 2))


def test_the_command_is_the_mean_over_the_next_samples_inside_the_episode():
    skill = torch.zeros(10, SIZE)
    skill[:, VELOCITY] = torch.arange(10).float()
    streams = _streams(skill, [(0, 5), (6, 9)])
    command = hindsight_commands(streams, span=3)
    # Step 0: the mean of 1, 2, 3. Step 4: only step 5 is left in its episode.
    assert command[0, 0] == pytest.approx(2.0)
    assert command[4, 0] == pytest.approx(5.0)
    # The last step of an episode has no successor and keeps its own value; the next
    # episode's samples are never read across the boundary.
    assert command[5, 0] == pytest.approx(5.0)
    assert command[6, 0] == pytest.approx(8.0)
    assert command[9, 0] == pytest.approx(9.0)


def test_relabel_replaces_only_the_command():
    skill = torch.randn(12, SIZE)
    streams = _streams(skill, [(0, 11)])
    out = relabel(streams, span=4)
    assert torch.equal(out.skill, streams.skill) and torch.equal(out.end, streams.end)
    assert torch.allclose(out.command, hindsight_commands(streams, 4))
    assert not torch.allclose(out.command, streams.command)
    with pytest.raises(ValueError):
        hindsight_commands(streams, span=0)
