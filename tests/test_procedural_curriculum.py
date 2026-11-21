"""The contact curriculum over the procedural family (``latent/curriculum.py``)."""

from __future__ import annotations

import math

import pytest
import torch

from humanoid_transfer.transfer.latent.curriculum import (
    FORWARD_EDGES,
    NO_CELL,
    PERIOD_BINS,
    SegmentCurriculum,
    cell_count,
    segment_cells,
)
from humanoid_transfer.transfer.latent.procedural import GaitRanges, PointMassBiped, standing_skill

WALKS = GaitRanges(duty=(0.55, 0.95))


def _cell(period: float, duty: float, forward: float, ranges: GaitRanges = WALKS) -> int:
    return int(
        segment_cells(
            torch.tensor([period]),
            torch.tensor([[duty, duty]]),
            torch.tensor([forward]),
            ranges.period,
            ranges.duty,
        )[0]
    )


def _biped(n: int = 32, curriculum: bool = False, seed: int = 3) -> PointMassBiped:
    biped = PointMassBiped(
        n, WALKS, body_step=0.08, substep=0.05, horizon=1.2, seed=seed, graph=False,
        curriculum=curriculum,
    )  # fmt: skip
    biped.set_offsets(torch.arange(1, 16, dtype=torch.float64) * 0.08)
    biped.anchor(torch.arange(n), standing_skill(n))
    return biped


def test_the_probe_periods_and_duties_each_have_their_own_cell() -> None:
    assert cell_count(WALKS.duty) == PERIOD_BINS * 8 * (len(FORWARD_EDGES) + 1) == 96
    periods = {_cell(p, 0.55, 0.0) for p in (0.8, 1.2, 1.8, 2.6)}
    assert len(periods) == 4
    duties = {_cell(1.8, d, 0.0) for d in (0.55, 0.62, 0.70, 0.85, 0.94)}
    assert len(duties) == 5
    speeds = {_cell(1.8, 0.55, v) for v in (0.0, 0.16, 0.35)}
    assert len(speeds) == 3
    # Backward and forward at the same speed share a cell; every cell is in range.
    assert _cell(1.8, 0.55, -0.08) == _cell(1.8, 0.55, 0.08)
    assert all(0 <= c < 96 for c in periods | duties | speeds)


def test_every_bank_segment_has_a_cell_and_the_family_fills_them() -> None:
    biped = _biped()
    cells = biped.bank_cell
    assert int(cells.min()) >= 0 and int(cells.max()) < cell_count(WALKS.duty)
    # Feasibility removes some corners, but most cells hold segments.
    assert len(torch.unique(cells)) > 60


def test_scores_set_the_draw_in_proportion() -> None:
    cells = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    curriculum = SegmentCurriculum(cells, 4, refresh=1, decay=0.0, min_count=1.0)
    g = torch.Generator().manual_seed(0)
    assert curriculum.sample(10, g) is None, "uniform until the first refresh"
    cell = torch.tensor([0, 1, 2, 3])
    curriculum.credit(cell, torch.tensor([0.1, 0.1, 0.1, 0.3]), torch.ones(4, dtype=torch.bool))
    drawn = cells[curriculum.sample(200000, g)]
    share = torch.bincount(drawn, minlength=4).float() / len(drawn)
    torch.testing.assert_close(share, torch.tensor([1.0, 1.0, 1.0, 3.0]) / 6.0, atol=0.01, rtol=0)
    history = curriculum.record()
    torch.testing.assert_close(
        torch.from_numpy(history["draw_share"][-1]), torch.tensor([1.0, 1.0, 1.0, 3.0]) / 6.0
    )


def test_unseen_cells_take_the_mean_and_masked_steps_count_for_nothing() -> None:
    cells = torch.tensor([0, 1, 2])
    curriculum = SegmentCurriculum(cells, 3, refresh=1, decay=0.0, min_count=2.0)
    cell = torch.tensor([0, 0, 1, 1, 2, NO_CELL])
    miss = torch.tensor([0.2, 0.2, 0.4, 0.4, 1.0, 1.0])
    mask = torch.tensor([True, True, True, True, True, True])
    curriculum.credit(cell, miss, mask)
    # Cell 2 was seen once (under min_count): it takes the mean of the seen cells, 0.3.
    torch.testing.assert_close(curriculum.score, torch.tensor([0.2, 0.4, 0.3]))
    again = SegmentCurriculum(cells, 3, refresh=1, decay=0.0, min_count=1.0)
    again.credit(cell, miss, torch.tensor([True, True, False, False, True, True]))
    # Cell 1's steps were masked: unseen, it takes the count-weighted mean (0.4 + 1.0) / 3.
    torch.testing.assert_close(again.score, torch.tensor([0.2, 1.4 / 3.0, 1.0]))


def test_a_curriculum_before_its_first_refresh_is_the_uniform_biped_bit_for_bit() -> None:
    off, on = _biped(curriculum=False), _biped(curriculum=True)
    ids = torch.arange(off.num_envs)
    steps = on.curriculum.refresh_steps - 1
    for _ in range(steps):
        for biped in (off, on):
            biped.advance()
        on.credit(torch.rand(on.num_envs), torch.ones(on.num_envs, dtype=torch.bool))
    a, b = off.plan(ids), on.plan(ids)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_the_cell_now_is_none_while_standing_and_a_bank_cell_while_walking() -> None:
    biped = _biped(n=64)
    # The episode starts standing: no cell.
    assert bool((biped.cell_now() == NO_CELL).all())
    seen = set()
    for _ in range(600):
        biped.advance()
        now = biped.cell_now()
        assert bool(((now == NO_CELL) | ((now >= 0) & (now < 96))).all())
        seen |= set(now[now >= 0].tolist())
    assert len(seen) > 10


def test_a_probe_is_in_no_cell() -> None:
    n = 4
    biped = PointMassBiped(n, WALKS, body_step=0.08, substep=0.05, horizon=1.2, graph=False)
    biped.set_offsets(torch.arange(1, 16, dtype=torch.float64) * 0.08)
    ids = torch.arange(n)
    params = {
        "forward": torch.full((n,), 0.1),
        "period": torch.full((n,), 1.2),
        "duty": torch.full((n, 2), 0.6),
        "duration": torch.full((n,), 1e9),
    }
    biped.set_probes(ids, params)
    biped.anchor(ids, standing_skill(n))
    for _ in range(400):
        biped.advance()
        assert bool((biped.cell_now() == NO_CELL).all())


def test_the_draw_follows_the_credited_cell() -> None:
    biped = _biped(n=64, curriculum=True)
    curriculum = biped.curriculum
    hard = _cell(1.8, 0.57, 0.05)
    # Every other cell keeps its contacts perfectly but for a floor; one is missed often.
    cells = torch.arange(curriculum.count).repeat(4)
    miss = torch.where(cells == hard, 0.5, 0.02)
    mask = torch.ones(len(cells), dtype=torch.bool)
    big = SegmentCurriculum(biped.bank_cell, curriculum.count, refresh=1, decay=0.0, min_count=1.0)
    big.credit(cells, miss, mask)
    biped.curriculum = big
    draws = torch.cat([biped._draw(4096)["cell"] for _ in range(8)]).long()
    uniform = float(big.bank_share[hard])
    share = float((draws == hard).float().mean())
    score = torch.where(torch.arange(big.count) == hard, 0.5, 0.02)
    expected = 0.5 * uniform / float((score * big.bank_share).sum())
    assert math.isfinite(share) and share == pytest.approx(expected, rel=0.15)
    assert share > 10 * uniform


def test_the_bias_score_cancels_read_lag_and_keeps_half_the_draw_uniform() -> None:
    cells = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    curriculum = SegmentCurriculum(cells, 4, refresh=1, decay=0.0, min_count=1.0, score="bias")
    cell = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    # Cell 0 lags a step at a lift and at a touchdown (+1 then -1): no bias. Cell 1 holds its
    # stance too long (+0.5 twice), cell 2 too short (-0.5 twice), cell 3 a little (+0.25).
    error = torch.tensor([1.0, -1.0, 0.5, 0.5, -0.5, -0.5, 0.25, 0.25])
    curriculum.credit(cell, error, torch.ones(8, dtype=torch.bool))
    torch.testing.assert_close(curriculum.score, torch.tensor([0.0, 0.5, 0.5, 0.25]))
    g = torch.Generator().manual_seed(0)
    drawn = cells[curriculum.sample(200000, g)]
    share = torch.bincount(drawn, minlength=4).float() / len(drawn)
    expected = 0.5 * torch.full((4,), 0.25) + 0.5 * torch.tensor([0.0, 0.5, 0.5, 0.25]) / 1.25
    torch.testing.assert_close(share, expected, atol=0.01, rtol=0)
    torch.testing.assert_close(
        torch.from_numpy(curriculum.record()["draw_share"][-1]), expected, atol=1e-6, rtol=0
    )
    # Opposite biases cancel in a cell, and an unseen cell takes the count-weighted mean |bias|.
    again = SegmentCurriculum(cells, 4, refresh=1, decay=0.0, min_count=2.0, score="bias")
    again.credit(
        torch.tensor([0, 0, 1, 1, 2]),
        torch.tensor([0.4, -0.4, 0.3, 0.3, 0.9]),
        torch.ones(5, dtype=torch.bool),
    )
    torch.testing.assert_close(again.score, torch.tensor([0.0, 0.3, 0.15, 0.15]))


def test_a_bias_curriculum_biped_before_its_first_refresh_is_the_uniform_biped() -> None:
    off, on = _biped(curriculum=False), _biped(curriculum="bias")
    assert on.curriculum.kind == "bias" and _biped(curriculum=True).curriculum.kind == "disagreement"
    ids = torch.arange(off.num_envs)
    for _ in range(on.curriculum.refresh_steps - 1):
        for biped in (off, on):
            biped.advance()
        on.credit(torch.rand(on.num_envs) - 0.5, torch.ones(on.num_envs, dtype=torch.bool))
    for x, y in zip(off.plan(ids), on.plan(ids)):
        assert torch.equal(x, y)
    with pytest.raises(ValueError):
        SegmentCurriculum(torch.zeros(3), 1, score="rank")
