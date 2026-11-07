"""The bodies' executions as a generator's data (``latent/body_data.py``; ``skill_space.md``
§10.17)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.body_data import (  # noqa: E402
    BodyPairs,
    _Reverse,
    body_pairs,
    leakage,
    load_body_streams,
    mmd2,
    train_on_bodies,
)
from humanoid_transfer.transfer.latent.contract import ContractClock, ContractDims  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import Streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    ContractConfig,
    contract_steps,
    futures,
    new_generator,
)

SIZE = SKILL_LAYOUT.size
LOAD = SKILL_LAYOUT.slice_of("foot_load")
CLOCK = ContractClock(step=0.08, history=13, horizon=13, commit=6, view=8)
DIMS = ContractDims(memory=4, intent=2, hidden=32, channels=8, flow_hidden=64)


def _gait(steps: int, envs: int, seed: int) -> np.ndarray:
    """``(T, N, skill)``: smooth signals, loads alternating between the feet."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)[:, None] * 0.3 + rng.uniform(0, 6, envs)[None]
    z = np.cumsum(rng.normal(0, 0.05, (steps, envs, SIZE)), axis=0).astype(np.float32)
    z[..., LOAD.start] = 0.5 + 0.5 * np.sin(t)
    z[..., LOAD.start + 1] = 0.5 - 0.5 * np.sin(t)
    return z


def _streams(z: np.ndarray) -> Streams:
    steps, envs = z.shape[:2]
    skill = torch.as_tensor(np.swapaxes(z, 0, 1).reshape(-1, SIZE))
    total = len(skill)
    column = torch.arange(total) // steps
    return Streams(
        obs=torch.zeros(total, 0),
        skill=skill,
        command=torch.full((total, 3), 0.1),
        action=torch.zeros(total, 0),
        falling=torch.zeros(total, dtype=torch.bool),
        start=column * steps + 1,
        end=column * steps + steps - 1,
        capture=torch.zeros(total, dtype=torch.long),
        env=column,
    )


def _capture(path, z: np.ndarray, leg: float, done_at: int | None = None) -> None:
    steps, envs = z.shape[:2]
    done = np.zeros((steps, envs), dtype=bool)
    if done_at is not None:
        done[done_at, 0] = True
    np.savez(
        path,
        skill_state=z,
        command=np.full((steps, envs, 3), 0.2, dtype=np.float32),
        done=done,
        terminated=np.zeros_like(done),
        leg_length=leg,
        total_mass=10.0,
    )


@pytest.fixture
def teacher():
    torch.manual_seed(0)
    streams = _streams(_gait(120, 4, 0))
    config = ContractConfig(epochs=1, batch_size=64, dims=DIMS)
    steps = contract_steps(streams, CLOCK.horizon)
    generator = new_generator(streams, steps, config, CLOCK)
    return streams, steps, generator, config


def test_a_body_capture_is_read_in_z_star_with_its_episodes_and_its_units(tmp_path):
    z = _gait(50, 3, 1)
    _capture(tmp_path / "a.npz", z, leg=0.25, done_at=19)
    _capture(tmp_path / "b.npz", z[:30], leg=0.65)
    streams, capture = load_body_streams([tmp_path / "a.npz", tmp_path / "b.npz"])
    assert streams.skill.shape == (50 * 3 + 30 * 3, SIZE)
    assert streams.obs.shape[1] == 0 and streams.action.shape[1] == 0
    assert torch.equal(capture, torch.tensor([0] * 150 + [1] * 90))
    # Column 0 of the first capture resets after step 19: two episodes.
    assert int(streams.end[0]) == 19 and int(streams.start[25]) == 21
    assert bool((streams.start <= torch.arange(240) + 1).all())
    # Commands are dimensionless: vx / sqrt(g L), as the teacher's are.
    assert math.isclose(float(streams.command[0, 0]), 0.2 / math.sqrt(9.81 * 0.25), rel_tol=1e-3)


def test_own_targets_are_the_future_the_body_went_on_to(teacher):
    streams, steps, generator, _ = teacher
    groups = torch.zeros(len(streams), dtype=torch.long)
    pairs = body_pairs(generator, streams, groups, streams, steps, "own", 50)
    assert len(pairs) == 50
    assert torch.allclose(pairs.future, futures(generator, streams, pairs.steps))


def test_teacher_targets_of_the_teachers_own_histories_are_its_own_futures(teacher):
    # Retrieval finds each history itself in the bank, so its target is where it went.
    streams, steps, generator, _ = teacher
    groups = torch.zeros(len(streams), dtype=torch.long)
    pairs = body_pairs(generator, streams, groups, streams, steps, "teacher", 40, bank=len(steps))
    own = futures(generator, streams, pairs.steps)
    assert torch.allclose(pairs.future, own, atol=1e-4)
    with pytest.raises(ValueError):
        body_pairs(generator, streams, groups, streams, steps, "expert", 4)


@pytest.mark.parametrize("invariance", [0.0, 1.0])
def test_training_on_the_bodies_moves_the_generator_and_logs_each_epoch(teacher, invariance):
    streams, steps, generator, config = teacher
    groups = (torch.arange(len(streams)) >= len(streams) // 2).long()
    pairs = body_pairs(generator, streams, groups, streams, steps, "teacher", 128, bank=256)
    before = [p.detach().clone() for p in generator.parameters()]
    sigma = torch.full((SIZE,), 0.01)
    log = train_on_bodies(
        streams, steps, steps[:64], generator, sigma, config, pairs, 2, invariance
    )
    assert [e["epoch"] for e in log] == [-1, 0, 1]
    assert ("adversary_loss" in log[-1]) == (invariance > 0)
    assert any(not torch.equal(a, b) for a, b in zip(before, generator.parameters(), strict=True))


def test_the_reversed_gradient_is_negated_and_scaled():
    x = torch.ones(3, requires_grad=True)
    _Reverse.apply(x, 0.5).sum().backward()
    assert torch.equal(x.grad, torch.full((3,), -0.5))


def test_the_probe_finds_a_body_the_summary_carries_and_not_one_it_does_not(teacher):
    streams, steps, generator, _ = teacher
    torch.manual_seed(1)
    window = torch.randn(400, CLOCK.history, SIZE)
    group = (torch.arange(400) % 2).long()
    blank = torch.zeros(400, CLOCK.horizon, SIZE)
    command = torch.zeros(400, 3)
    sigma = torch.zeros(SIZE)
    same = BodyPairs(window, command, blank, group, torch.arange(400))
    apart = replace(same, window=window + 3.0 * group[:, None, None])
    blind = leakage(generator, same, sigma)
    seen = leakage(generator, apart, sigma)
    assert blind["balanced_accuracy"] < 0.65 and blind["chance"] == 0.5
    assert seen["balanced_accuracy"] > 0.9 and seen["control"] < 0.65


def test_mmd_is_near_zero_within_a_distribution_and_large_between_two():
    torch.manual_seed(2)
    a, b = torch.randn(300, 4), torch.randn(300, 4)
    assert abs(float(mmd2(a, b))) < 0.05
    assert float(mmd2(a, b + 2.0)) > 0.5
    # Scale-free: shrinking both samples alike does not change it.
    small = float(mmd2(0.1 * a, 0.1 * (b + 2.0)))
    assert math.isclose(float(mmd2(a, b + 2.0)), small, rel_tol=1e-4)


def test_target_commands_are_the_teachers_at_each_target_and_need_teacher_targets(teacher):
    streams, steps, generator, _ = teacher
    marked = replace(streams, command=torch.arange(len(streams)).float()[:, None].repeat(1, 3))
    groups = torch.zeros(len(streams), dtype=torch.long)
    pairs = body_pairs(
        generator, marked, groups, marked, steps, "teacher", 40, bank=len(steps),
        command_from="target",
    )  # fmt: skip
    # Retrieval finds each of the teacher's own histories in the bank, at its own (marked)
    # command, so the target's step is the history's own, and its marked command names it.
    assert torch.equal(pairs.command[:, 0].long(), pairs.steps)
    with pytest.raises(ValueError):
        body_pairs(generator, streams, groups, streams, steps, "own", 4, command_from="target")
    with pytest.raises(ValueError):
        body_pairs(generator, streams, groups, streams, steps, "teacher", 4, command_from="x")


def test_the_mmd_constraint_trains_logs_its_two_sides_and_moves_its_multiplier(teacher):
    streams, steps, generator, config = teacher
    groups = (torch.arange(len(streams)) >= len(streams) // 2).long()
    pairs = body_pairs(generator, streams, groups, streams, steps, "teacher", 128, bank=256)
    sigma = torch.full((SIZE,), 0.01)
    log = train_on_bodies(
        streams, steps, steps[:64], generator, sigma, config, pairs, 2, 1.0,
        invariance_kind="mmd",
    )  # fmt: skip
    assert "adversary_loss" not in log[-1]
    assert {"mmd", "mmd_reference", "mmd_multiplier"} <= set(log[-1])
    assert log[-1]["mmd_multiplier"] >= 0.0
    with pytest.raises(ValueError):
        train_on_bodies(
            streams, steps, steps[:64], generator, sigma, config, pairs, 1, 1.0,
            invariance_kind="probe",
        )  # fmt: skip
