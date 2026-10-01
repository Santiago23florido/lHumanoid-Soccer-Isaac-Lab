"""Checks on the contract's generator: its clock, its sampling and its timing measurement.

The generator is the task side of the contract -- the history of ``z*`` and
a command in, a sampled future of ``z*`` out. These tests pin down what can
be checked without a teacher: that a plan covers what is held and seen of
it, that it is built from the history alone, that it does not collapse two
equally likely futures onto their mean, that noise in a padded window
repeats like an encoder's estimate at a reset, and that the timing read off
a sampled future is the one written into it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    Contract,
    ContractClock,
    ContractDims,
    ContractGenerator,
    contract_loss,
    load_contract,
    save_contract,
)
from humanoid_transfer.transfer.latent.sequences import Streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    change_times,
    contract_steps,
    windows,
)

SMALL = ContractDims(memory=4, intent=2, hidden=32, channels=8, flow_hidden=64)
KINDS = ("flow", "cvae")
CLOCK = ContractClock(step=0.08, history=13, horizon=13, commit=6, view=8)
LOAD = SKILL_LAYOUT.slice_of("foot_load")


def _streams(skill: torch.Tensor, episode_lengths: list[int]) -> Streams:
    total = skill.shape[0]
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


def test_a_plan_must_cover_what_is_held_and_what_is_seen_of_it() -> None:
    with pytest.raises(ValueError):
        ContractClock(step=0.08, horizon=12, commit=6, view=8)
    assert ContractClock(step=0.08).horizon == 13


@pytest.mark.parametrize("kind", KINDS)
def test_a_plan_is_a_future_built_on_the_current_state(kind: str) -> None:
    torch.manual_seed(0)
    generator = ContractGenerator(replace(SMALL, kind=kind), CLOCK)
    window = torch.randn(5, CLOCK.history, 27)
    command = torch.randn(5, 3)
    noise = torch.randn(5, generator.noise_size)

    plan = generator.sample(window, command, noise)
    assert plan.shape == (5, CLOCK.horizon, 27)
    assert torch.equal(plan, generator.sample(window, command, noise)), "noise alone draws"
    # The history enters through the memory, not only through the current state.
    other = window.clone()
    other[:, 0] += 1.0
    assert not torch.allclose(generator.sample(other, command, noise), plan)


@pytest.mark.parametrize("kind", KINDS)
def test_two_equally_likely_futures_are_sampled_not_averaged(kind: str) -> None:
    torch.manual_seed(0)
    generator = ContractGenerator(replace(SMALL, kind=kind), CLOCK)
    optimiser = torch.optim.Adam(generator.parameters(), lr=3e-3)
    for _ in range(400):
        window = 0.01 * torch.randn(128, CLOCK.history, 27)
        sign = torch.where(torch.rand(128) < 0.5, 1.0, -1.0)
        future = window[:, -1:] + 0.01 * torch.randn(128, CLOCK.horizon, 27)
        future[..., 0] += sign[:, None]
        losses = contract_loss(generator, window, torch.zeros(128, 3), future, free_bits=0.1)
        optimiser.zero_grad()
        losses["total"].backward()
        optimiser.step()

    generator.eval()
    with torch.no_grad():
        window = torch.zeros(2000, CLOCK.history, 27)
        noise = torch.randn(2000, generator.noise_size)
        plans = generator.sample(window, torch.zeros(2000, 3), noise)
    first = plans[:, :, 0].mean(dim=1)
    assert 0.35 < float((first > 0).float().mean()) < 0.65
    assert float(first.abs().mean()) > 0.7, "the generator answered with the mean future"


def test_noise_in_a_padded_window_repeats_one_estimate() -> None:
    torch.manual_seed(0)
    streams = _streams(torch.randn(40, 27), [40])
    generator = ContractGenerator(SMALL, CLOCK)
    steps = torch.tensor([3])  # the episode's trustworthy start is step 1
    window = windows(generator, streams, steps, torch.full((27,), 0.5))
    padded = CLOCK.history - 2  # steps 2 and 3 are real, the rest repeat step 1
    assert torch.allclose(window[0, :padded], window[0, :1].expand(padded, -1))
    assert not torch.allclose(window[0, padded - 1], window[0, padded])


def test_plans_never_reach_past_an_episode_or_into_a_fall() -> None:
    streams = _streams(torch.zeros(60, 27), [30, 30])
    streams.falling[50:60] = True
    steps = contract_steps(streams, CLOCK.horizon)
    assert int(steps.min()) == 1 and int(steps.max()) <= 59 - CLOCK.horizon
    assert not ((steps <= 29) & (steps + CLOCK.horizon > 29)).any()
    assert not ((steps + CLOCK.horizon) >= 50).any()


def test_the_timing_read_off_a_future_is_the_one_written_into_it() -> None:
    #           now  1  2  3  4  5
    left = [[1, 1, 0, 0, 1, 1]]
    right = [[1, 1, 1, 1, 1, 1]]
    contact = torch.tensor([left, right]).permute(1, 2, 0).bool()  # (1, 6, 2)
    assert change_times(contact).tolist() == [[2, 6]]


@pytest.mark.parametrize("kind", KINDS)
def test_a_saved_contract_samples_the_same_plans(tmp_path, kind: str) -> None:
    torch.manual_seed(0)
    dims = replace(SMALL, kind=kind)
    generator = ContractGenerator(dims, CLOCK).eval()
    generator.tolerance.uniform_()
    sigma = torch.rand(27)
    path = tmp_path / "generator.pt"
    save_contract(path, Contract(generator, None, sigma, {"name": "test"}), dims, CLOCK, 120)
    loaded = load_contract(path)
    window, command = torch.randn(3, CLOCK.history, 27), torch.randn(3, 3)
    noise = torch.randn(3, generator.noise_size)
    with torch.no_grad():
        assert torch.allclose(
            loaded.generator.sample(window, command, noise),
            generator.sample(window, command, noise),
        )
    assert torch.equal(loaded.sigma, sigma) and loaded.meta["name"] == "test"
    assert torch.equal(loaded.generator.tolerance, generator.tolerance)
    assert loaded.encoder is None
