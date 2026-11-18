"""Replayed plans: read in body times, the same at the same instant on every body."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.replay import (  # noqa: E402
    ReplayChains,
    load_replay,
    replay_starts,
)

GRID = 0.0774


def _chains(count: int = 3, length: int = 50, skill: int = 27) -> ReplayChains:
    generator = torch.Generator().manual_seed(0)
    return ReplayChains(torch.randn(count, length, skill, generator=generator), GRID)


def test_rows_on_the_grid_are_the_chain_and_between_them_a_blend() -> None:
    replay = _chains()
    chain = torch.tensor([0, 2])
    on_grid = torch.tensor([[0.0, 3 * GRID], [7 * GRID, 49 * GRID]])
    rows = replay.rows(chain, on_grid)
    assert torch.allclose(rows[0, 1], replay.states[0, 3], atol=1e-6)
    assert torch.allclose(rows[1, 0], replay.states[2, 7], atol=1e-6)
    assert torch.allclose(rows[1, 1], replay.states[2, 49], atol=1e-6)
    middle = replay.rows(chain[:1], torch.tensor([[4.25 * GRID]]))[0, 0]
    expected = 0.75 * replay.states[0, 4] + 0.25 * replay.states[0, 5]
    assert torch.allclose(middle, expected, atol=1e-5)


def test_instants_outside_a_chain_hold_its_ends() -> None:
    replay = _chains()
    rows = replay.rows(torch.tensor([1]), torch.tensor([[-1.0, 1e3]]))
    assert torch.equal(rows[0, 0], replay.states[1, 0])
    assert torch.equal(rows[0, 1], replay.states[1, -1])


def test_two_bodies_are_handed_the_same_row_at_the_same_instant() -> None:
    """A G1 (dt* 0.0774) drawing at 3.0 and a NAO (dt* 0.1161) drawing at 2.3483 both reach
    instant 3.5806 -- the G1 at its 7th row, the NAO at its 11th -- and read the same row."""
    replay = _chains(length=400)
    chain = torch.tensor([1])
    g1 = replay.ahead(chain, torch.tensor([3.0]), 20, 0.0774)
    nao = replay.ahead(chain, torch.tensor([3.0 + 7 * 0.0774 - 11 * 0.1161]), 20, 0.1161)
    assert torch.allclose(g1[0, 6], nao[0, 10], atol=1e-5)
    direct = replay.rows(chain, torch.tensor([[3.0 + 7 * 0.0774]]))[0, 0]
    assert torch.allclose(g1[0, 6], direct, atol=1e-6)


def test_a_body_on_the_grid_reads_the_chain_row_by_row() -> None:
    replay = _chains(length=200)
    now = torch.tensor([10 * GRID, 31 * GRID])
    rows = replay.ahead(torch.tensor([0, 2]), now, 24, GRID)
    assert rows.shape == (2, 24, 27)
    assert torch.allclose(rows[0], replay.states[0, 11:35], atol=1e-5)
    assert torch.allclose(rows[1], replay.states[2, 32:56], atol=1e-5)


def test_the_window_ends_at_its_instant_oldest_first() -> None:
    replay = _chains(length=100)
    window = replay.window(torch.tensor([2]), torch.tensor([20 * GRID]), steps=5, dt=2 * GRID)
    expected = replay.states[2, [12, 14, 16, 18, 20]]
    assert torch.allclose(window[0], expected, atol=1e-5)


def test_starts_cover_the_history_and_space_envs_sharing_a_chain() -> None:
    starts = replay_starts(7, chains=3, window=1.5, spacing=4.0)
    assert torch.allclose(starts, torch.tensor([1.5, 1.5, 1.5, 5.5, 5.5, 5.5, 9.5]))


def test_load_normalises_and_checks_shapes(tmp_path) -> None:
    chains = np.random.default_rng(0).normal(size=(30, 4, 27))
    command = np.zeros((4, 3))
    path = tmp_path / "replay.npz"
    np.savez(path, chain_states=chains, grid_step=GRID, command=command)
    mean, std = 0.5, 2.0
    replay = load_replay(path, lambda z: (z - mean) / std)
    assert replay.count == 4 and replay.length == 30
    assert abs(replay.grid_step - GRID) < 1e-12
    expected = torch.as_tensor((chains.transpose(1, 0, 2) - mean) / std, dtype=torch.float32)
    assert torch.allclose(replay.states, expected, atol=1e-6)
    np.savez(path, chain_states=chains, grid_step=GRID, command=np.zeros((3, 3)))
    with pytest.raises(ValueError):
        load_replay(path, lambda z: z)
