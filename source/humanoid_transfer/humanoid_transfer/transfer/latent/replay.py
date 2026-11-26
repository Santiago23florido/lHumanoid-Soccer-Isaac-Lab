"""Fixed plans every body follows alike: the same chain of ``z*``, read in body times.

A contract tracker is handed plans its task's generator draws from the body's own
history with fresh noise, so two bodies never follow the same plan and their states
cannot be paired. A *replay* fixes the plan instead: a chain of dimensionless ``z*``
(the generator run on its own plans, :func:`~.train_contract.chain_states`), sampled on
the generator's grid, which env ``i`` follows from a fixed instant whatever the body.
Since time is in body times (``T = sqrt(L / g)``), the plan is the same at the same
dimensionless instant on every body, and ``(chain, instant)`` pairs their states: a
co-execution.

A replayed plan is read at the body's own instants straight from the chain's grid
(:meth:`ReplayChains.ahead`), not through the plan matrix: a generator's plan starts on
its grid at the draw, and blending the chain onto that shifted grid and again onto the
body's steps would make the reference at an instant depend on when the plan was drawn.

File format (``.npz``): ``chain_states`` ``(T, K, 27)`` dimensionless ``z*`` (not
normalised), ``grid_step`` (body times), ``command`` ``(K, 3)`` dimensionless; any other
keys are provenance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_TOLERANCE = 1e-6


class ReplayChains:
    """``K`` chains of normalised ``z*`` on a grid, read at any instant by linear blending."""

    def __init__(
        self, states: torch.Tensor, grid_step: float, command: torch.Tensor | None = None
    ) -> None:
        if states.dim() != 3:
            raise ValueError(f"states must be (K, T, skill), got {tuple(states.shape)}")
        self.states = states
        self.grid_step = float(grid_step)
        self.command = command
        self.count, self.length = int(states.shape[0]), int(states.shape[1])

    @property
    def duration(self) -> float:
        """The last instant a chain holds, in body times."""
        return (self.length - 1) * self.grid_step

    def rows(self, chain: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """``chain (N,)``, ``times (N, R)`` -> ``(N, R, skill)``; instants outside a chain
        hold its first or last row."""
        position = (times / self.grid_step).clamp(0.0, self.length - 1)
        low = torch.floor(position + _TOLERANCE).long().clamp(max=self.length - 1)
        high = (low + 1).clamp(max=self.length - 1)
        weight = (position - low.to(position.dtype)).clamp(0.0, 1.0)[..., None]
        k = chain[:, None].expand_as(low)
        return (1.0 - weight) * self.states[k, low] + weight * self.states[k, high]

    def window(self, chain: torch.Tensor, end: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        """``(N, steps, skill)``: the history a body with step ``dt`` would hold at ``end``,
        oldest first (``end - (steps - 1) dt``, ..., ``end``)."""
        offsets = torch.arange(steps - 1, -1, -1, device=end.device, dtype=end.dtype) * dt
        return self.rows(chain, end[:, None] - offsets[None])

    def ahead(self, chain: torch.Tensor, now: torch.Tensor, count: int, dt: float) -> torch.Tensor:
        """``(N, count, skill)``: the chain at ``now + (j + 1) dt``, the instants a body with
        step ``dt`` reaches -- read from the chain's own grid once, so the row at an instant
        is the same whenever and on whatever body the plan was drawn."""
        steps = torch.arange(1, count + 1, device=now.device, dtype=now.dtype) * dt
        return self.rows(chain, now[:, None] + steps[None])

    def to(self, device: torch.device | str) -> ReplayChains:
        command = None if self.command is None else self.command.to(device)
        return ReplayChains(self.states.to(device), self.grid_step, command)


def load_replay(path: str | Path, normalise, device: torch.device | str = "cpu") -> ReplayChains:
    """Read a replay file; ``normalise`` maps dimensionless ``z*`` to the contract's units
    (``generator.skill_norm``)."""
    data = np.load(path)
    states = torch.as_tensor(data["chain_states"], dtype=torch.float32, device=device)
    if states.dim() != 3:
        raise ValueError(f"{path}: chain_states must be (T, K, skill)")
    command = torch.as_tensor(data["command"], dtype=torch.float32, device=device)
    if command.shape != (states.shape[1], 3):
        raise ValueError(f"{path}: command must be (K, 3)")
    with torch.no_grad():
        normalised = normalise(states.transpose(0, 1).contiguous())
    return ReplayChains(normalised, float(data["grid_step"]), command)


def replay_starts(count: int, chains: int, window: float, spacing: float) -> torch.Tensor:
    """``(count,)``: env ``i`` starts chain ``i % chains`` at ``window + (i // chains) spacing``,
    the same on every body; ``window`` is the history's span, so it lies inside the chain."""
    index = torch.arange(count, dtype=torch.float32)
    return window + torch.div(index, chains, rounding_mode="floor") * spacing


__all__ = ["ReplayChains", "load_replay", "replay_starts"]
