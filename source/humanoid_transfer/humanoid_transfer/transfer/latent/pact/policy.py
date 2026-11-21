"""PACT as a policy: the anchored state encoder and the joint decoder, one model for every body.

The encoder is G-A3's ``pact_kd`` (kinematic anchors and drive loads as inputs, the CoM
velocity, height correction and loads learned); the decoder reads its entity tokens, ``E_b``
and the task's plan, and gives every joint its position target in radians from the joint's
default. Nothing in it is per body: a body enters through its descriptor tokens only.

The plan is what each tracker observes (``ReferenceCommand.observation`` and ``.path``),
decoded out of its generator's normalised units into raw ``z*``
(:func:`plan_from_view`): the trackers' views hold different step counts (8 on the G1, 4 on
the NAO, 9 on H1 and Cassie) over the same span of each body's own time ``sqrt(L/g)``, and
the NAO's generator normalises with its own statistics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from humanoid_transfer.transfer.latent.pact.model import (
    BodyBatch,
    EmbodimentEncoder,
    JointDecoder,
    PactDims,
    StateEncoder,
)
from humanoid_transfer.transfer.latent.pact.tokens import split

SKILL = 27

PLAN_LIMIT = 50.0
"""The clamp on a tracker's plan view, normalised units (``tracking/command._PLAN_LIMIT``)."""


def view_steps(observation_width: int, body_width: int) -> int:
    """Plan steps in a tracker's view: its policy observation is ``body_width`` numbers of
    the body (IMU, joints, last action), then ``view * 27 + 27`` (plan relative to the
    estimate, then the estimate) and ``view * 3`` (path)."""
    rest = observation_width - body_width - SKILL
    if rest <= 0 or rest % (SKILL + 3):
        raise ValueError(f"{observation_width} - {body_width} is not a tracker's plan view")
    return rest // (SKILL + 3)


def plan_from_view(
    tail: np.ndarray, view: int, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``tail (..., view * 30 + 27)``, the plan part of a tracker's observation, in raw units.

    Returns ``(upcoming (..., view, 27), estimate (..., 27), path (..., view, 3))``; ``mean``
    and ``std`` are the generator's ``skill_norm``. A view clamped at the plan limit stays
    clamped (rare; the same clamp the tracker saw).
    """
    relative = tail[..., : view * SKILL].reshape(*tail.shape[:-1], view, SKILL)
    estimate_norm = tail[..., view * SKILL : view * SKILL + SKILL]
    path = tail[..., view * SKILL + SKILL :].reshape(*tail.shape[:-1], view, 3)
    upcoming = (relative + estimate_norm[..., None, :]) * std + mean
    return upcoming, estimate_norm * std + mean, path


class PactPolicy(nn.Module):
    """``E_b``, the anchored state encoder and the timed-plan joint decoder.

    With ``dims.anchor_inputs`` off (the no-anchor ablation) the encoder does not read the
    analytic anchors (``anchor_in`` is accepted and ignored); trained without the readout
    loss, its entity latent is then anchored by nothing.
    """

    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        if not dims.timed_plan:
            raise ValueError("PactPolicy reads a timed plan")
        self.dims = dims
        self.embodiment = EmbodimentEncoder(dims)
        self.state = StateEncoder(dims)
        self.decoder = JointDecoder(dims)

    def forward(
        self,
        bodies: BodyBatch,
        joint_obs: torch.Tensor,
        imu: torch.Tensor,
        anchor_in: torch.Tensor,
        plan: torch.Tensor,
        plan_relative: torch.Tensor,
        plan_time: torch.Tensor,
        plan_path: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``-> (learned (B, 6), targets (B, J))``.

        ``learned``: CoM velocity, height correction, the two loads (as G-A3's encoder,
        in whatever normalisation its targets were given); ``targets``: joint position
        targets, padded joints zero. ``plan`` and ``plan_relative`` are ``(B, H, 27)``.
        """
        embodiment, descriptor = self.embodiment(bodies)
        anchors = split(anchor_in) if self.dims.anchor_inputs else None
        out = self.state(bodies, descriptor, embodiment, joint_obs, imu, anchors)
        value_c, value_k = split(plan)
        delta_c, delta_k = split(plan_relative)
        targets = self.decoder(
            bodies,
            out["joints"],
            out["latent"],
            embodiment,
            torch.cat([value_c, delta_c], dim=-1),
            torch.cat([value_k, delta_k], dim=-1),
            plan_time=plan_time,
            plan_path=plan_path,
            plan_mask=plan_mask,
        )
        learned = torch.cat(
            [out["centroid"][:, 1:4], out["centroid"][:, 0:1], out["contacts"][:, :, 6]], dim=1
        )
        return learned, targets
