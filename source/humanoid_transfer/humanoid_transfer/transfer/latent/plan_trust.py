"""How far to trust a plan: its own sampling noise, measured online, weighs it against the body.

A task's generator is a sampler: two plans drawn a commitment apart predict the
same stretch of the future differently, by how much its details are arbitrary.
Measured on the steps two successive plans share, that disagreement is the
plan's noise, component by component; the plan's own change from step to step
is the variation it is meant to carry. A prior that turns the plan into joint
targets fuses the two as a Kalman filter fuses a measurement with a prediction
(Kalman 1960; the adaptive form estimates the noise from the innovations,
Mehra 1970):

    target = now + K (plan - now),    K = variation / (variation + noise)

per component of ``z*``. The point-mass biped's plans are deterministic between
redraws, so they are trusted fully (``K ≈ 1``); a generator's are trusted where
its samples agree. No threshold is chosen: both terms are measured.

The same noise, applied to the biped's plans while a tracker trains
(:func:`smooth_noise`), lets the tracker practise the plans it will be given.
:func:`sway` adds the trunk's sway in the gait, which the point mass leaves free.

Pure torch, batched over environments; ``z*`` in the contract's normalised units.
"""

from __future__ import annotations

import math

import torch


def overlap_disagreement(
    old_plan: torch.Tensor, old_age: torch.Tensor, new_plan: torch.Tensor
) -> torch.Tensor:
    """``(n, size)``: mean squared difference of two plans over the steps they share.

    ``old_plan`` and ``new_plan`` are ``(n, length, size)``, on the body's control steps;
    the old plan has been held ``old_age`` steps, so its row ``old_age + k`` and the new
    plan's row ``k`` predict the same step.
    """
    length = old_plan.shape[1]
    shared = length - int(old_age.max())
    if shared <= 0:
        return torch.zeros(old_plan.shape[0], old_plan.shape[2], device=old_plan.device)
    rows = old_age[:, None] + torch.arange(shared, device=old_plan.device)
    index = rows.clamp(max=length - 1)[..., None].expand(-1, -1, old_plan.shape[2])
    old = torch.gather(old_plan, 1, index)
    return (new_plan[:, :shared] - old).pow(2).mean(dim=1)


def step_variation(plan: torch.Tensor) -> torch.Tensor:
    """``(n, size)``: mean squared change of a plan from one step to the next."""
    return (plan[:, 1:] - plan[:, :-1]).pow(2).mean(dim=1)


class PlanTrust:
    """Running estimates of a plan source's noise and variation, and the gain they give.

    One row per source (a task's generator, the biped). The disagreement of two
    independent samples has twice a sample's variance, so the noise is half of it.
    """

    def __init__(self, sources: int, size: int, rate: float, device) -> None:
        self.rate = rate
        self.noise = torch.zeros(sources, size, device=device)
        self.variation = torch.zeros(sources, size, device=device)
        self.seen = torch.zeros(sources, dtype=torch.bool, device=device)

    def update(self, source: int, disagreement: torch.Tensor, variation: torch.Tensor) -> None:
        if len(disagreement) == 0:
            return
        noise, var = 0.5 * disagreement.mean(dim=0), variation.mean(dim=0)
        if not bool(self.seen[source]):
            self.noise[source], self.variation[source] = noise, var
            self.seen[source] = True
            return
        self.noise[source] += self.rate * (noise - self.noise[source])
        self.variation[source] += self.rate * (var - self.variation[source])

    def gain(self, source: int) -> torch.Tensor:
        """``(size,)``: ``K = variation / (variation + noise)``; 1 before anything is measured."""
        variation, noise = self.variation[source], self.noise[source]
        return torch.where(
            variation + noise > 0, variation / (variation + noise).clamp(min=1e-12), 1.0
        )


def fuse(now: torch.Tensor, plan: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """``now + K (plan - now)``: the plan, trusted component by component."""
    return now[:, None] + gain * (plan - now[:, None])


def smooth_noise(
    shape: tuple[int, int, int], noise: torch.Tensor, generator=None, device=None
) -> torch.Tensor:
    """``(n, length, size)``: noise shaped like a sampler's -- an offset and a drift per plan.

    Two samples of a smooth generator differ by a smooth function of time, not by
    white noise; its two lowest modes (a constant and a ramp, each of half the
    variance) carry ``noise`` per component.
    """
    n, length, size = shape
    std = noise.clamp(min=0.0).sqrt()
    offset = torch.randn(n, 1, size, device=device, generator=generator) * std / math.sqrt(2)
    ramp = torch.linspace(-1.0, 1.0, length, device=device)[None, :, None] * math.sqrt(3)
    slope = torch.randn(n, 1, size, device=device, generator=generator) * std / math.sqrt(2)
    return offset + ramp * slope


def sway(gravity: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Gravity in the base frame after small trunk rotations, ``(..., 3)``, unit length.

    ``pitch`` and ``roll`` (radians, broadcast against ``gravity[..., 0]``) rotate the
    base about its own y and x axes; to first order gravity in the base turns by
    the opposite rotation: ``g' = g - ω x g`` with ``ω = (roll, pitch, 0)``.
    """
    omega = torch.stack([roll, pitch, torch.zeros_like(roll)], dim=-1)
    turned = gravity - torch.linalg.cross(omega, gravity, dim=-1)
    return turned / turned.norm(dim=-1, keepdim=True).clamp(min=1e-9)


__all__ = [
    "PlanTrust",
    "fuse",
    "overlap_disagreement",
    "smooth_noise",
    "step_variation",
    "sway",
]
