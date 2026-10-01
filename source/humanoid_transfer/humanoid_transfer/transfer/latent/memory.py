"""A memory for the shared skill, in the body's own time.

A memory that every body shares has to agree on what "the last two seconds"
means, and seconds are the wrong unit for it: the same gait is slower on a
longer leg. In body time ``T = sqrt(L / g)`` the G1 controls every 0.077 ``T``
and the NAO, at the same 50 Hz, every 0.126 ``T`` -- so a window of the last
26 control steps would be 1.6 times longer, physically, on the NAO. The shared
memory reads a fixed grid in ``T`` instead, and each body resamples its own
history onto it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TimeGrid:
    """``samples`` instants, ``step`` body times apart, ending now. Oldest first."""

    step: float
    samples: int

    @property
    def span(self) -> float:
        """Body times between the oldest sample and now."""
        return (self.samples - 1) * self.step

    def history_steps(self, dt_star: float) -> int:
        """How many of a body's control steps (``dt_star`` body times apart) the grid needs."""
        return math.ceil(self.span / dt_star - _TOLERANCE) + 1

    def interpolation(self, dt_star: float) -> np.ndarray:
        """``(samples, history_steps)``: linear interpolation of a step history onto the grid.

        The history is oldest first and ends at the current step, like the
        grid. When ``dt_star`` equals the grid step this is the identity.
        """
        steps = self.history_steps(dt_star)
        matrix = np.zeros((self.samples, steps))
        for j in range(self.samples):
            position = (steps - 1) - (self.samples - 1 - j) * self.step / dt_star
            nearest = round(position)
            if abs(position - nearest) < _TOLERANCE:
                matrix[j, nearest] = 1.0
                continue
            low = math.floor(position)
            weight = position - low
            matrix[j, low] = 1.0 - weight
            matrix[j, low + 1] = weight
        return matrix


__all__ = ["TimeGrid"]
