"""Checks on the contact schedule: its labels, its planner and the policy that commits to it.

The schedule is the part of a skill no instantaneous state carries -- when
each contact changes next, and where the next landing is. The tests pin down
the labels on hand-built sequences, the planner's likelihood and sampling on
known distributions, and the runtime commitment step by step, without
training anything large.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.events import contact_events  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import Streams  # noqa: E402

LOAD = SKILL_LAYOUT.slice_of("foot_load")
FEET = SKILL_LAYOUT.slice_of("foot_position")


def _streams(skill: torch.Tensor, episode_lengths: list[int]) -> Streams:
    """One column made of consecutive episodes of the given lengths."""
    total = skill.shape[0]
    assert sum(episode_lengths) == total
    start, end = [], []
    first = 0
    for length in episode_lengths:
        start += [first + 1] * length
        end += [first + length - 1] * length
        first += length
    return Streams(
        obs=torch.zeros(total, 123),
        skill=skill,
        command=torch.zeros(total, 3),
        action=torch.zeros(total, 37),
        falling=torch.zeros(total, dtype=torch.bool),
        start=torch.tensor(start),
        end=torch.tensor(end),
        capture=torch.zeros(total, dtype=torch.long),
        env=torch.zeros(total, dtype=torch.long),
    )


def _skill(left: list[int], right: list[int]) -> torch.Tensor:
    """A skill state whose feet carry the given contacts and sit at ``x = step``."""
    total = len(left)
    z = torch.zeros(total, 27)
    z[:, LOAD] = torch.tensor([left, right], dtype=torch.float32).T * 0.5
    feet = z[:, FEET].reshape(total, 2, 3)
    feet[..., 0] = torch.arange(total, dtype=torch.float32)[:, None]
    feet[:, 1, 1] = -0.1
    z[:, FEET] = feet.reshape(total, 6)
    return z


def test_events_give_time_to_the_next_change_and_the_next_landing() -> None:
    left = [1, 1, 1, 0, 0, 1, 1, 1]
    right = [1, 1, 1, 1, 1, 1, 0, 0]
    events = contact_events(_streams(_skill(left, right), [8]))

    # The left foot lifts off at 3 and lands at 5; the right foot lifts off at 6, never lands.
    assert events.to_change[:, 0].tolist()[1:6] == [2.0, 1.0, 2.0, 1.0, float("inf")]
    assert events.to_change[:, 1].tolist()[1:7] == [5.0, 4.0, 3.0, 2.0, 1.0, float("inf")]
    assert events.landing_known[:5, 0].all() and not events.landing_known[5:, 0].any()
    assert torch.allclose(events.landing[1:5, 0], torch.tensor([5.0, 0.0]).expand(4, 2))
    assert not events.landing_known[:, 1].any()
    assert events.remaining.tolist() == [7, 6, 5, 4, 3, 2, 1, 0]


def test_events_never_cross_an_episode_or_count_the_reset_as_one() -> None:
    # Episode 1 is steps 0-3, episode 2 is 4-7. The left foot is loaded only in episode 2,
    # and loaded already at its untrustworthy first step and the one after.
    left = [0, 0, 0, 0, 1, 1, 1, 0]
    right = [1, 1, 1, 1, 1, 1, 1, 1]
    events = contact_events(_streams(_skill(left, right), [4, 4]))

    assert torch.isinf(events.to_change[:4, 0]).all(), "a change in the next episode leaked"
    assert not events.landing_known[:4, 0].any()
    # In episode 2 the only event is the lift-off at 7, the reset at 4 is not a landing.
    assert events.to_change[4:7, 0].tolist() == [3.0, 2.0, 1.0]
    assert not events.landing_known[4:, 0].any()
    assert events.remaining.tolist() == [3, 2, 1, 0, 3, 2, 1, 0]
