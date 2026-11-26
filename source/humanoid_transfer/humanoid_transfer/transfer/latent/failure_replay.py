"""Replay what a tracker failed on: prioritised replay of failing training draws.

A tracker's rare failures sit in rare parts of its training draw (a slow lateral command with a
turn, a fast lateral gait), which a uniform draw visits about as rarely as they fail. When a
training episode terminates (not by time-out), whatever it was drawn with -- its task's command
or its procedural segment -- is stored; a share of later draws comes from the store instead of
the uniform draw (prioritised level replay, Jiang et al. 2021). The rule reads only failures, so
it is the same for every body.

``FailureStore`` holds the last ``capacity`` failing draws as rows of a fixed width;
``FailureStore.mix`` replaces a share of a batch of fresh draws by stored ones.
"""

from __future__ import annotations

import torch


class FailureStore:
    """A ring buffer of failing draws, ``(capacity, width)`` on one device."""

    def __init__(self, width: int, capacity: int = 4096, device: torch.device | str = "cpu") -> None:
        self.rows = torch.zeros(capacity, width, device=device)
        self.capacity = int(capacity)
        self.count = 0
        """Rows ever added (the buffer holds the last ``capacity``)."""

    def __len__(self) -> int:
        return min(self.count, self.capacity)

    def add(self, rows: torch.Tensor) -> None:
        """Store ``rows`` ``(k, width)``; the oldest are overwritten once full."""
        rows = rows.reshape(-1, self.rows.shape[1]).to(self.rows)
        for start in range(0, len(rows), self.capacity):
            chunk = rows[start : start + self.capacity]
            slots = (self.count + torch.arange(len(chunk), device=self.rows.device)) % self.capacity
            self.rows[slots] = chunk
            self.count += len(chunk)

    def mix(
        self, fresh: torch.Tensor, share: float, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fresh`` ``(n, width)`` with a ``share`` of its rows replaced by stored ones drawn
        uniformly; returns the rows and which were replaced ``(n,)``. Empty: ``fresh`` as given."""
        n = len(fresh)
        chosen = torch.zeros(n, dtype=torch.bool, device=fresh.device)
        if len(self) == 0 or share <= 0.0 or n == 0:
            return fresh, chosen
        u = torch.rand(n, generator=generator, device=fresh.device)
        chosen = u < share
        picks = torch.randint(0, len(self), (int(chosen.sum()),), generator=generator, device=fresh.device)
        out = fresh.clone()
        out[chosen] = self.rows[picks.to(self.rows.device)].to(fresh)
        return out, chosen
