"""Failure replay: the store keeps the last failures and a share of draws comes from it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.failure_replay import FailureStore  # noqa: E402


def test_an_empty_store_changes_nothing():
    store = FailureStore(3)
    fresh = torch.rand(10, 3)
    out, chosen = store.mix(fresh, 0.5)
    assert torch.equal(out, fresh) and not chosen.any()


def test_a_share_of_draws_comes_from_the_failures():
    store = FailureStore(3)
    store.add(torch.tensor([[1.0, 2.0, 3.0]]))
    g = torch.Generator().manual_seed(0)
    out, chosen = store.mix(torch.zeros(20000, 3), 0.5, g)
    assert abs(chosen.float().mean().item() - 0.5) < 0.02
    assert torch.equal(out[chosen], torch.tensor([1.0, 2.0, 3.0]).expand(int(chosen.sum()), 3))
    assert torch.equal(out[~chosen], torch.zeros(int((~chosen).sum()), 3))


def test_the_store_keeps_the_last_failures():
    store = FailureStore(1, capacity=4)
    store.add(torch.arange(10.0)[:, None])
    assert len(store) == 4 and store.count == 10
    assert sorted(store.rows[:, 0].tolist()) == [6.0, 7.0, 8.0, 9.0]


def _walking_biped(n: int = 16, share: float = 0.0):
    from humanoid_transfer.transfer.latent.procedural import GaitRanges, PointMassBiped, standing_skill

    biped = PointMassBiped(
        n, GaitRanges(duty=(0.55, 0.95)), body_step=0.08, substep=0.05, horizon=1.2, seed=5,
        graph=False, failure_replay=share,
    )  # fmt: skip
    biped.set_offsets(torch.arange(1, 16, dtype=torch.float64) * 0.08)
    biped.anchor(torch.arange(n), standing_skill(n))
    for _ in range(400):
        biped.advance()
    return biped


def test_the_segment_index_now_matches_the_segment_cell():
    biped = _walking_biped()
    index, cell = biped.index_now(), biped.cell_now()
    walking = index >= 0
    assert walking.any()
    assert torch.equal(biped.bank_cell[index[walking]].long(), cell[walking])
    assert bool((cell[~walking] < 0).all())


def test_a_failed_segment_is_drawn_again():
    biped = _walking_biped(share=1.0)
    index = biped.index_now()
    failed = torch.nonzero(index >= 0)[:1, 0]
    biped.record_failures(failed)
    assert len(biped.failures) == 1
    draws = biped._draw(256)["index"].long()
    assert bool((draws == index[failed]).all())


def test_without_replay_nothing_is_stored():
    biped = _walking_biped(share=0.0)
    biped.record_failures(torch.arange(4))
    assert biped.failures is None
