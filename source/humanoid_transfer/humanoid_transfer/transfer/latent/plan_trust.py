"""How far to trust a plan: its own sampling noise, measured online, weighs it against the body.

A task's generator is a sampler: two plans drawn from the same history at the
same moment differ by how much their details are arbitrary. Row by row -- each
row a lead time -- that disagreement is the plan's noise ``R_k``, component by
component; how far the plan's row lies from the body's state now,
``E[(plan_k - now)^2] = S_k + R_k``, holds the motion it is meant to carry,
``S_k``, plus that noise. A prior that turns the plan into joint targets fuses
the two as a Kalman filter fuses a measurement with a prediction (Kalman 1960;
the noise estimated online, Mehra 1970):

    target_k = now + K_k (plan_k - now),    K_k = S_k / (S_k + R_k) = 1 - R_k / E[(plan_k - now)^2]

per lead time and component of ``z*``. The point-mass biped's plans are
deterministic; their noise is the one added to them (:func:`smooth_noise`),
known exactly. No threshold is chosen: every term is measured.

(Two first versions are superseded. One compared each plan with the one it
replaced on the steps they share, where the replaced plan predicts them a
commitment further ahead: that measured how a forecast's uncertainty grows
with its lead, and trusted the generators' plans 0.2-0.3. One compared two
samples but pooled their noise over a commitment against a single step's
change: 0.35-0.67, still mixing lead times.)

:func:`sway` adds the trunk's sway in the gait, which the point mass leaves free.

Pure torch, batched over environments; ``z*`` in the contract's normalised units.
"""

from __future__ import annotations

import math

import torch


def step_variation(plan: torch.Tensor) -> torch.Tensor:
    """``(n, size)``: mean squared change of a plan from one step to the next."""
    return (plan[:, 1:] - plan[:, :-1]).pow(2).mean(dim=1)


class PlanTrust:
    """Running estimates, per plan source, lead time and component, of a plan's noise and reach.

    ``noise`` is half the disagreement of two independent samples (their difference
    has twice a sample's variance); ``reach`` is the mean squared distance of a plan's
    row from the body's state when it was drawn.
    """

    def __init__(self, sources: int, length: int, size: int, rate: float, device) -> None:
        self.rate = rate
        self.noise = torch.zeros(sources, length, size, device=device)
        self.reach = torch.zeros(sources, length, size, device=device)
        self.seen = torch.zeros(sources, dtype=torch.bool, device=device)

    def update(self, source: int, disagreement: torch.Tensor, reach: torch.Tensor) -> None:
        """``disagreement`` and ``reach``: ``(n, length, size)``, from ``n`` plans just drawn."""
        if len(disagreement) == 0:
            return
        noise, far = 0.5 * disagreement.mean(dim=0), reach.mean(dim=0)
        if not bool(self.seen[source]):
            self.noise[source], self.reach[source] = noise, far
            self.seen[source] = True
            return
        self.noise[source] += self.rate * (noise - self.noise[source])
        self.reach[source] += self.rate * (far - self.reach[source])

    def gain(self, source: int) -> torch.Tensor:
        """``(length, size)``: ``K = 1 - noise / reach`` in ``[0, 1]``; 1 before any measure."""
        reach, noise = self.reach[source], self.noise[source]
        gain = 1.0 - noise / reach.clamp(min=1e-12)
        return torch.where(reach > 0, gain.clamp(0.0, 1.0), 1.0)


def fuse(now: torch.Tensor, plan: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """``now + K (plan - now)``: the plan ``(n, rows, size)``, trusted row and component by
    component (``gain`` broadcast against it)."""
    return now[:, None] + gain * (plan - now[:, None])


def smooth_noise(
    shape: tuple[int, int, int], noise: torch.Tensor, generator=None, device=None
) -> torch.Tensor:
    """``(n, length, size)``: noise shaped like a sampler's -- an offset and a drift per plan.

    Two samples of a smooth generator differ by a smooth function of time, not by
    white noise; its two lowest modes (a constant and a ramp, of equal variance)
    carry ``noise``, per component or per row and component, on average over the plan.
    """
    n, length, size = shape
    offset = torch.randn(n, 1, size, device=device, generator=generator) / math.sqrt(2)
    slope = torch.randn(n, 1, size, device=device, generator=generator) / math.sqrt(2)
    ramp = torch.linspace(-1.0, 1.0, length, device=device)[None, :, None] * math.sqrt(3)
    return (offset + ramp * slope) * noise.clamp(min=0.0).sqrt()


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
    "smooth_noise",
    "step_variation",
    "sway",
]
