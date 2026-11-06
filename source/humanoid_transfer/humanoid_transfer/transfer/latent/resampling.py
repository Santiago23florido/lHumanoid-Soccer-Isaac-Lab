"""The contract between the generator's time grid and a body's control instants.

A generator plans on its teacher's grid (the G1's control step, ``Δ* = 0.0774``). A body
with another step reads its history onto that grid and the plan back onto its own
instants, by linear interpolation in time (:class:`~.memory.TimeGrid`,
:func:`plan_interpolation`).

**In which space.** Interpolating the feet's loads linearly and then reading contact as a
load above 5 % of the weight lengthens every stance: between a stance sample (about half
the weight) and a swing sample (zero), a linear blend stays above 5 % for nine tenths of
the interval. On the NAO (step 0.126 ``T``) the plans it was asked to follow carried
duty factors 0.03--0.06 above the same generator's on the G1, and the teacher's own gait
read the same way when resampled -- although a duty factor, a fraction of time, does not
depend on the rate it is sampled at (``skill_space.md`` §10). In the generator's own space
(loads as ``log(load + 0.01)``, :meth:`~.contract.ContractGenerator.warp`) zero load is far
from any stance, a blend crosses the threshold near the middle of the interval, and the
duty is kept. ``resample_rows(..., space="generator")`` interpolates there.

Where a body's step equals the grid's, every matrix here is a selection and both spaces
give the same numbers; :func:`resample_rows` then skips the round trip entirely, so such a
body (the G1) is bit-identical in either space.
"""

from __future__ import annotations

import math

import torch

from .contract import ContractGenerator

RESAMPLE_SPACES = ("linear", "generator")
_TOLERANCE = 1e-6


def plan_interpolation(grid_step: float, horizon: int, dt_star: float) -> torch.Tensor:
    """``(L, horizon + 1)``: a plan (now, then ``horizon`` grid samples) at the body's steps.

    Row ``j`` is the instant ``(j + 1) * dt_star`` after the plan was drawn, as
    long as it lies within the plan. When the body's step equals the grid's,
    this selects the plan's samples one by one.
    """
    length = math.floor(horizon * grid_step / dt_star + _TOLERANCE)
    matrix = torch.zeros(length, horizon + 1)
    for j in range(length):
        position = (j + 1) * dt_star / grid_step
        low = min(math.floor(position + _TOLERANCE), horizon)
        weight = position - low
        if weight < _TOLERANCE or low == horizon:
            matrix[j, low] = 1.0
        else:
            matrix[j, low] = 1.0 - weight
            matrix[j, low + 1] = weight
    return matrix


def is_selection(matrix: torch.Tensor) -> bool:
    """Whether every row of an interpolation matrix picks one sample (no blending)."""
    ones = (matrix == 1.0).sum(dim=-1)
    zeros = (matrix == 0.0).sum(dim=-1)
    return bool(((ones == 1) & (zeros == matrix.shape[-1] - 1)).all())


def resample_rows(
    generator: ContractGenerator,
    matrix: torch.Tensor,
    sequence: torch.Tensor,
    space: str = "linear",
) -> torch.Tensor:
    """``(N, L, D)``: ``matrix`` ``(L, K)`` applied along time to ``sequence`` ``(N, K, D)``.

    ``sequence`` is normalised ``z*``. ``linear`` blends it as it is; ``generator`` blends
    it in the generator's space (:meth:`~.contract.ContractGenerator.warp`) and maps the
    result back. A selection matrix is applied directly in either space.
    """
    if space not in RESAMPLE_SPACES:
        raise ValueError(f"unknown resample space {space!r}; expected one of {RESAMPLE_SPACES}")
    if space == "linear" or not generator.warp_loads or is_selection(matrix):
        return torch.einsum("lk,nkd->nld", matrix, sequence)
    warped = generator.warp(sequence)
    return generator.unwarp(torch.einsum("lk,nkd->nld", matrix, warped))


__all__ = ["RESAMPLE_SPACES", "is_selection", "plan_interpolation", "resample_rows"]
