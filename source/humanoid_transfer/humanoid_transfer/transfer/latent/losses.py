"""The objective that factors the teacher through the shared space.

.. math::

    L = L_{BC} + \\alpha\\,\\lVert \\hat z^* - z^* \\rVert^2
        + \\gamma\\,\\bigl(\\lVert T_{z^*} - z^*_{t+1} \\rVert^2
                          + \\lVert T_\\zeta - \\mathrm{sg}(\\zeta_{t+1}) \\rVert^2\\bigr)
        + \\beta\\,\\mathrm{KL}(\\zeta)

``anchored=False`` is the ablation: the ``ẑ*`` slot becomes one more free
latent, and the dynamics term predicts the encoder's own next output instead of
the physical next state. Same architecture, same capacity -- what changes is
only whether the space has been told what its coordinates mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dataset import Transitions
from .model import BodyDecoder, BodyEncoder, ObservationLayout, SharedSkill


@dataclass(frozen=True)
class LossWeights:
    anchor: float = 1.0
    dynamics: float = 0.5
    kl: float = 1e-2
    """Strong enough to close ``ζ``. At 1e-3 it carried about 7 nats that, combined with
    ``ẑ*``, decoded the G1's joint configuration (probe R^2 0.89 against 0.43 for the
    true ``z*``) for no gain in imitation: the free part is where a body leaks into
    the shared space. ``transfer/scripts/train_latent.py`` keeps that run as the
    ``anchored_open`` ablation."""
    anchored: bool = True


def latent_losses(
    encoder: BodyEncoder,
    skill: SharedSkill,
    decoder: BodyDecoder,
    layout: ObservationLayout,
    batch: Transitions,
    weights: LossWeights,
    sample: bool = True,
) -> dict[str, torch.Tensor]:
    skill_n, mean, log_var = encoder(layout.encoder_input(batch.obs))
    free = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean) if sample else mean
    action = skill.act(skill_n, free, batch.command)
    imitation = F.mse_loss(decoder(action, layout.context_input(batch.obs)), batch.action)

    with torch.no_grad():
        next_skill_n, next_mean, _ = encoder(layout.encoder_input(batch.next_obs))
    predicted_skill, predicted_free = skill.predict(skill_n, free, action)

    if weights.anchored:
        anchor = F.mse_loss(skill_n, skill.skill_norm(batch.skill))
        skill_target = skill.skill_norm(batch.next_skill)
    else:
        anchor = torch.zeros((), device=skill_n.device)
        skill_target = next_skill_n
    dynamics = F.mse_loss(predicted_skill, skill_target) + F.mse_loss(predicted_free, next_mean)
    kl = 0.5 * (mean.pow(2) + log_var.exp() - 1.0 - log_var).sum(dim=-1).mean()

    total = imitation + weights.anchor * anchor + weights.dynamics * dynamics + weights.kl * kl
    return {
        "total": total,
        "imitation": imitation,
        "anchor": anchor,
        "dynamics": dynamics,
        "kl": kl,
    }


__all__ = ["LossWeights", "latent_losses"]
