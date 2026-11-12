"""What a tracker observes, in the units it trained in.

A contract normalises ``z*`` with its generator's statistics, ``N(x) = (x - m) / s``. L4
deploys a tracker under a task's contract, so the ``z*`` it observes is normalised with the
task's statistics, not with those it trained under. Read in another contract's units, a
normalised value is

    N_own(N_task⁻¹(x)) = a x + b,   a = s_task / s_own,   b = (m_task - m_own) / s_own,

affine and exact. On a difference of two normalised values the shift cancels.

The G1's contracts share one set of statistics, where the map is the identity. The NAO's
walk contract (``walk_dagger2e``) has its own (``transfer/scripts/units_shift.py``).
"""

from __future__ import annotations

import torch

from .model import Normaliser


def change_of_units(
    source: Normaliser, target: Normaliser
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(scale, shift)`` with ``target(source.inverse(x)) == scale * x + shift``.

    ``None`` when the two share their statistics (to float tolerance, as the command term's
    tasks are compared), where the map is the identity.
    """
    if torch.allclose(source.mean, target.mean) and torch.allclose(source.std, target.std):
        return None
    return source.std / target.std, (source.mean - target.mean) / target.std


def tracker_view(
    upcoming: torch.Tensor,
    estimate: torch.Tensor,
    has_plan: torch.Tensor,
    limit: float,
    units: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``(N, view * 27 + 27)``: the held plan relative to ``ẑ*``, then ``ẑ*``.

    ``upcoming`` ``(N, view, 27)`` and ``estimate`` ``(N, 27)`` are in the contract's
    normalised units; ``units`` (from :func:`change_of_units`) maps both into the tracker's.
    Without it this is the view in the generator's units, operation for operation.
    """
    relative = upcoming - estimate[:, None]
    if units is not None:
        scale, shift = units
        relative = relative * scale
        estimate = estimate * scale + shift
    relative = torch.where(~has_plan[:, None, None], 0.0, relative)
    view = torch.cat([relative.flatten(1), estimate], dim=-1)
    return torch.nan_to_num(view, nan=0.0).clamp(-limit, limit)
