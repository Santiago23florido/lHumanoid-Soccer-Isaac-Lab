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


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

from humanoid_transfer.transfer.latent.memory import MemoryDims, MemorySkill, TimeGrid  # noqa: E402
from humanoid_transfer.transfer.latent.model import (  # noqa: E402
    G1_OBSERVATION,
    BodyDecoder,
    LatentDims,
)
from humanoid_transfer.transfer.latent.plan import (  # noqa: E402
    ContactPlanner,
    EventClock,
    calibration,
    cdf,
    fit_landing_normaliser,
    plan_targets,
    planned_losses,
    sample_bins,
    survival_nll,
)

G1_DT_STAR = 0.02 / (0.65457 / 9.81) ** 0.5
NAO_DT_STAR = 0.02 / (0.24801 / 9.81) ** 0.5


def test_the_clock_counts_bins_of_body_time_on_any_body() -> None:
    clock = EventClock(step=G1_DT_STAR, bins=26)
    steps = torch.tensor([1.0, 2.0, 26.0, 27.0, float("inf")])
    assert clock.event_bin(steps, G1_DT_STAR).tolist() == [1, 2, 26, 27, 27]
    assert clock.observed_bins(torch.tensor([0.0, 5.0, 40.0]), G1_DT_STAR).tolist() == [0, 5, 26]

    # The NAO's step is 1.63 of the G1's: 3 of its steps end 4.9 bins ahead, inside bin 5.
    assert clock.event_bin(torch.tensor([1.0, 3.0]), NAO_DT_STAR).tolist() == [2, 5]
    assert clock.observed_bins(torch.tensor([3.0]), NAO_DT_STAR).tolist() == [4]


def test_the_survival_likelihood_counts_events_and_censoring() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    h = torch.sigmoid(logits[0])
    event = survival_nll(logits, torch.tensor([2]), torch.tensor([0]))
    assert torch.isclose(event, -(torch.log(1 - h[0]) + torch.log(h[1])))[0]
    censored = survival_nll(logits, torch.tensor([4]), torch.tensor([2]))
    assert torch.isclose(censored, -(torch.log(1 - h[0]) + torch.log(1 - h[1])))[0]


def test_a_constant_hazard_is_recovered_through_censoring() -> None:
    generator = torch.Generator().manual_seed(0)
    truth, bins, total = 0.2, 10, 20000
    times = torch.distributions.Geometric(probs=torch.tensor(truth)).sample((total,)) + 1
    seen = torch.randint(0, bins + 1, (total,), generator=generator)
    event_bin = torch.where((times <= seen) & (times <= bins), times.long(), bins + 1)
    observed = seen.clamp(max=bins)

    logit = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.Adam([logit], lr=0.05)
    for _ in range(300):
        loss = survival_nll(logit.expand(total, bins), event_bin, observed).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert abs(float(torch.sigmoid(logit)) - truth) < 0.01


def test_sampling_follows_the_planned_distribution() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 12)
    draws = sample_bins(logits.expand(40000, 2, 12), torch.rand(40000, 2))
    k = torch.arange(1, 13)
    empirical = (draws[..., None] <= k).float().mean(dim=0)
    assert torch.allclose(empirical, cdf(logits)[0], atol=0.01)
    # Sampled outcomes are calibrated against the logits that produced them.
    error = calibration(logits.expand(40000, 2, 12), draws, torch.full_like(draws, 12))
    assert (error < 0.01).all()


def _planned_setup(parts: int = 2):
    from humanoid_transfer.transfer.latent.events import contact_events

    left = ([1] * 6 + [0] * 2) * 6
    right = ([1] * 2 + [0] * 2 + [1] * 4) * 6
    streams = _streams(_skill(left, right), [24, 24])
    streams.obs.normal_()
    events = contact_events(streams)
    dims = MemoryDims(hidden=32, channels=8, memory=4, plan=3 * parts)
    clock = EventClock(step=G1_DT_STAR, bins=8)
    skill = MemorySkill(dims, TimeGrid(step=G1_DT_STAR, samples=13))
    planner = ContactPlanner(dims, parts, clock)
    steps = streams.samples(future=2)
    skill.skill_norm.fit(streams.skill[steps])
    fit_landing_normaliser(planner, events, steps)
    decoder = BodyDecoder(74, 37, LatentDims(action=16, hidden=32))
    return streams, events, skill, planner, decoder, clock, steps


def test_the_hindsight_plan_is_the_schedule_that_followed() -> None:
    streams, events, _, planner, _, clock, _ = _planned_setup()
    steps = torch.tensor([1, 2, 5])
    targets = plan_targets(planner, streams, events, steps, clock, G1_DT_STAR, FEET)
    # Left foot lifts off at 6: 5, 4 and 1 steps ahead. Right foot lifts off at 2, lands at 4
    # and lifts off again at 10: 1, 2 and 5 steps ahead.
    assert targets.event_bin[:, 0].tolist() == [5, 4, 1]
    assert targets.event_bin[:, 1].tolist() == [1, 2, 5]
    # Its landing at 4 is known from steps 1 and 2; from 5 on, it is the one at 12.
    assert targets.landing_known[:, 1].all()
    fraction = targets.plan[:, :2]
    assert torch.allclose(fraction[:, 0], torch.tensor([5.0, 4.0, 1.0]) / 8)


def test_rollouts_a_student_drove_never_teach_the_planner() -> None:
    torch.manual_seed(0)
    streams, events, skill, planner, decoder, clock, steps = _planned_setup()
    sigma = torch.full((27,), 0.1)

    def losses(hindsight: bool):
        flags = torch.full((len(streams),), hindsight)
        return planned_losses(
            skill, planner, decoder, G1_OBSERVATION, streams, events, flags, steps, sigma,
            horizon=2, dynamics_weight=0.5, plan_weight=1.0, clock=clock, dt_star=G1_DT_STAR,
            feet=FEET,
        )  # fmt: skip

    student = losses(False)
    assert float(student["survival"]) == 0.0 and float(student["landing"]) == 0.0
    planner.zero_grad(set_to_none=True)
    student["total"].backward()
    assert all(p.grad is None or not p.grad.any() for p in planner.net.parameters())

    planner.zero_grad(set_to_none=True)
    losses(True)["total"].backward()
    assert any(p.grad is not None and p.grad.any() for p in planner.net.parameters())
