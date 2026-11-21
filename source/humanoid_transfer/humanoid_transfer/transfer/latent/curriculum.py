"""Draw the procedural family where a tracker executes it worst: a contact curriculum.

A tracker trained on the point-mass biped's family (:mod:`.procedural`) sees each gait as
often as the family's uniform draw makes it: the long, slow swings that every body's
tracker shortens are about one segment in a hundred. This module draws each new segment
in proportion to how badly the tracker keeps that kind of gait's contact schedule, which
is prioritised sampling of training conditions by the learner's own error (proportional
prioritisation, Schaul et al. 2016; per-level scores, Jiang et al. 2021).

* **Cells.** The bank's segments are grouped by what sets a contact schedule: the period
  (:data:`PERIOD_BINS` log-uniform bins over the family's range), the mean duty (bins of
  :data:`DUTY_BIN` from the family's lower bound) and the forward speed (``|v|`` cut at
  :data:`FORWARD_EDGES`). Everything else in a segment stays uniform within its cell.
* **Score.** Per cell, the share of feet whose contact state differs from the plan's,
  over the control steps spent in that cell's segments. Sums and counts decay by
  :data:`DECAY` at each refresh (every :data:`REFRESH_STEPS` control steps), so the score
  follows the tracker as it learns.
* **Draw.** A segment's probability is its cell's score. A cell seen less than
  :data:`MIN_COUNT` times takes the mean score; before the first refresh the draw is the
  family's uniform one. No mixture and no temperature: the best cells' score is above
  zero (a foot's touchdown read one step late is already a disagreement), so every cell
  keeps being drawn.

That floor also caps the draw: a proportional draw gives a cell at most its score over the
mean, and a floor near the mean keeps every boost under 2. The ``"bias"`` score removes it.

* **Bias score.** Per cell, ``|mean signed contact error|``: per step, each foot's reached
  minus planned contact (+1 in contact while the plan swings, -1 the other way), averaged
  over the feet. Over a cell's steps that is the duty the feet hold minus the duty asked,
  what the gait probes judge. One step of read lag adds +1 at a lift and -1 at a touchdown
  and cancels.
* **Bias draw.** :data:`UNIFORM_SHARE` of the draws are the family's uniform one, the rest in
  proportion to the score, so every cell keeps at least that share of its uniform draw.

The scores are the tracker's, not a body's range: the same rule runs on every body.
"""

from __future__ import annotations

import math

import numpy as np
import torch

PERIOD_BINS = 4
"""Log-uniform period bins over the family's range: over 0.6-3.0 T the edges are 0.90,
1.34 and 2.01, so the gait probes' periods 0.8, 1.2, 1.8 and 2.6 each have their own."""

DUTY_BIN = 0.05
"""Width of a mean-duty bin, a cycle's share: the probes' duty tolerance is 0.10 relative."""

FORWARD_EDGES = (0.12, 0.30)
"""``|forward|`` cuts, in ``V``: near standing, a walk, a fast walk."""

REFRESH_STEPS = 200
"""Control steps between refreshes of the draw (about 8 PPO iterations of 24 steps)."""

DECAY = 0.9
"""What a refresh keeps of the sums and counts before it: a memory of about ten refreshes."""

MIN_COUNT = 200.0
"""Decayed count under which a cell's score is the mean, not its own."""

NO_CELL = -1
"""The cell of a step that is in no segment of the bank: standing, recovering or a probe."""

SCORES = ("disagreement", "bias")
"""``"disagreement"``: the share of feet off the plan's contact, drawn in proportion.
``"bias"``: ``|mean signed contact error|``, drawn half uniformly and half in proportion."""

UNIFORM_SHARE = 0.5
"""Share of the bias curriculum's draws left to the family's uniform draw (as prioritised
level replay mixes new levels with replayed ones): no cell falls under half its share."""


def cell_count(duty_range: tuple[float, float]) -> int:
    """How many cells a family with this mean-duty range has."""
    duty_bins = max(1, math.ceil((duty_range[1] - duty_range[0]) / DUTY_BIN - 1e-9))
    return PERIOD_BINS * duty_bins * (len(FORWARD_EDGES) + 1)


def segment_cells(
    period: torch.Tensor,
    duty: torch.Tensor,
    forward: torch.Tensor,
    period_range: tuple[float, float],
    duty_range: tuple[float, float],
) -> torch.Tensor:
    """The cell of each segment, from its period ``(n,)``, feet's duties ``(n, 2)`` and
    forward speed ``(n,)``."""
    low, high = period_range
    span = math.log(high / low)
    period_bin = torch.floor(torch.log(period / low) / span * PERIOD_BINS).long()
    period_bin = period_bin.clamp(0, PERIOD_BINS - 1)
    duty_bins = max(1, math.ceil((duty_range[1] - duty_range[0]) / DUTY_BIN - 1e-9))
    duty_bin = torch.floor((duty.mean(dim=-1) - duty_range[0]) / DUTY_BIN + 1e-6).long()
    duty_bin = duty_bin.clamp(0, duty_bins - 1)
    edges = torch.tensor(FORWARD_EDGES, device=forward.device, dtype=forward.dtype)
    forward_bin = torch.bucketize(forward.abs(), edges, right=True)
    return (period_bin * duty_bins + duty_bin) * (len(FORWARD_EDGES) + 1) + forward_bin


class SegmentCurriculum:
    """Scores the bank's cells by the tracker's contact disagreement and draws by them.

    Args:
        cells: The cell of each segment of the bank, ``(B,)``.
        count: How many cells there are.
        score: One of :data:`SCORES`: what :meth:`credit` is fed and how the draw uses it.
    """

    def __init__(
        self,
        cells: torch.Tensor,
        count: int,
        *,
        refresh: int = REFRESH_STEPS,
        decay: float = DECAY,
        min_count: float = MIN_COUNT,
        score: str = "disagreement",
    ) -> None:
        if score not in SCORES:
            raise ValueError(f"score {score!r} is not one of {SCORES}")
        self.kind = score
        self.cells = cells.long()
        self.count = int(count)
        self.refresh_steps = int(refresh)
        self.decay = float(decay)
        self.min_count = float(min_count)
        device = self.cells.device
        # One slot past the cells takes the steps that belong to none, without a sync.
        self._window_sums = torch.zeros(self.count + 1, device=device)
        self._window_counts = torch.zeros(self.count + 1, device=device)
        self.sums = torch.zeros(self.count, device=device)
        self.counts = torch.zeros(self.count, device=device)
        self.bank_share = torch.bincount(self.cells, minlength=self.count).float()
        self.bank_share = self.bank_share / self.bank_share.sum()
        """Each cell's share of the bank: its share of a uniform draw."""
        self.score: torch.Tensor | None = None
        self.probability: torch.Tensor | None = None
        """Per segment of the bank, its weight in a draw; None until the first refresh."""
        self.steps = 0
        self.refreshes = 0
        self.history: list[dict[str, np.ndarray]] = []
        """At each refresh: the cells' scores, decayed counts and draw share."""

    def credit(self, cell: torch.Tensor, miss: torch.Tensor, mask: torch.Tensor) -> bool:
        """Add one control step: each env's cell ``(n,)`` (:data:`NO_CELL` for none), its
        error ``(n,)`` (share of feet off the plan's contact; for ``"bias"``, the feet's mean
        reached minus planned contact) and whether it counts ``(n,)``. Returns whether the
        draw was refreshed."""
        valid = mask & (cell >= 0) & (cell < self.count)
        slot = torch.where(valid, cell.long(), self.count)
        self._window_sums.index_add_(0, slot, torch.where(valid, miss.float(), 0.0))
        self._window_counts.index_add_(0, slot, valid.float())
        self.steps += 1
        if self.steps % self.refresh_steps:
            return False
        self._refresh()
        return True

    def _refresh(self) -> None:
        self.sums = self.decay * self.sums + self._window_sums[: self.count]
        self.counts = self.decay * self.counts + self._window_counts[: self.count]
        self._window_sums.zero_()
        self._window_counts.zero_()
        self.refreshes += 1
        seen = self.counts >= self.min_count
        if not bool(seen.any()):
            return
        value = self.sums / self.counts.clamp(min=1e-9)
        if self.kind == "disagreement":
            mean = self.sums[seen].sum() / self.counts[seen].sum()
            score = torch.where(seen, value, mean)
            self.probability = score[self.cells]
            share = score * self.bank_share
            share = share / share.sum()
        else:
            size = value.abs()
            mean = (self.counts * size)[seen].sum() / self.counts[seen].sum()
            score = torch.where(seen, size, mean)
            weight = score[self.cells]
            weight = weight / weight.sum().clamp(min=1e-12)
            self.probability = UNIFORM_SHARE / len(self.cells) + (1.0 - UNIFORM_SHARE) * weight
            share = score * self.bank_share
            share = UNIFORM_SHARE * self.bank_share + (1.0 - UNIFORM_SHARE) * share / share.sum()
        self.score = score
        self.history.append(
            {
                "step": np.array(self.steps),
                "score": score.cpu().numpy(),
                "value": value.cpu().numpy(),
                "counts": self.counts.cpu().numpy(),
                "draw_share": share.cpu().numpy(),
            }
        )

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor | None:
        """``n`` bank indices drawn by the scores, or None before the first refresh."""
        if self.probability is None:
            return None
        return torch.multinomial(self.probability, n, replacement=True, generator=generator)

    def record(self) -> dict[str, np.ndarray]:
        """The refreshes so far, stacked, with each cell's share of the bank."""
        out = {
            "bank_share": self.bank_share.cpu().numpy(),
            "refresh_steps": np.array(self.refresh_steps),
            "kind": np.array(self.kind),
        }
        if self.history:
            for key in self.history[0]:
                out[key] = np.stack([h[key] for h in self.history])
        return out


__all__ = [
    "DECAY",
    "DUTY_BIN",
    "FORWARD_EDGES",
    "MIN_COUNT",
    "NO_CELL",
    "PERIOD_BINS",
    "REFRESH_STEPS",
    "SCORES",
    "SegmentCurriculum",
    "UNIFORM_SHARE",
    "cell_count",
    "segment_cells",
]
