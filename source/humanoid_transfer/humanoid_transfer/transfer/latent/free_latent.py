"""A free part of the shared latent, learned from both bodies with attention (§9, F5).

``z*`` is fixed by physics: the centroidal state in body units. What it leaves
out -- how the load sits under each foot, how close a body is to falling,
whatever a later task needs -- has to be learned, and learned so that it means
the same on every body, or it cannot carry a skill from one to another. The
design follows what transferred in the literature (``transfer/docs/related_work.md``):

* **Entity tokens, not joints.** Each body's sensors are grouped into the same
  functional entities -- trunk, left foot, right foot -- by a per-body stem, so
  the tokens line up across bodies by construction (Aberman et al.'s primal
  skeleton; URMA's per-entity encoding). A transformer shared by every body
  attends across them (Body Transformer, Sferrazza et al. 2024).
* **Anchored and bottlenecked.** ``ζ`` is read next to ``z*``, never instead of
  it, and its information is bounded by a KL budget (PULSE; the variational
  information bottleneck): what ``ζ`` carries must be what ``z*`` lacks.
* **Trained to predict physics.** A predictor shared by every body reads
  ``(z*, ζ)`` and predicts each foot's centre of pressure and ground force and
  whether the body falls soon. A body-agnostic predictor forces ``ζ`` to encode
  them the same way on every body.
* **Aligned on paired states.** The same planned gait at the same phase, on
  both bodies (pairs from the procedural family, dynamically valid on each):
  their ``ζ`` should agree (CrossBFM, SONIC's token loss, Gupta et al. 2017).
* **Not a body detector.** A classifier reading ``ζ`` must not tell the bodies
  apart better than chance (X-WBC found MSE alignment alone leaves the body's
  identity in the tokens). It is a Lagrangian constraint: its multiplier rises
  by dual ascent while the classifier beats chance by more than twice its
  standard error, and the encoder is trained against it through a reversed
  gradient (Ganin and Lempitsky 2015).

Pure torch: the per-body stems take the entities' features as a body's
captures record them; :func:`entity_features` builds them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

ENTITIES = ("trunk", "left_foot", "right_foot")


@dataclass(frozen=True)
class FreeLatentDims:
    width: int = 64
    """Each token's size."""

    heads: int = 4
    layers: int = 2
    free: int = 8
    """``ζ``."""

    skill: int = 27
    """``z*``, read next to ``ζ``."""

    balance: int = 10
    """Each foot's centre of pressure ``(x, y)`` and ground force ``(x, y, z)``."""


class BodyStems(nn.Module):
    """One body's map from each entity's own features to a token."""

    def __init__(self, sizes: dict[str, int], width: int) -> None:
        super().__init__()
        missing = set(ENTITIES) - set(sizes)
        if missing:
            raise ValueError(f"no features for {sorted(missing)}")
        self.stems = nn.ModuleDict(
            {name: nn.Sequential(nn.LayerNorm(sizes[name]), nn.Linear(sizes[name], width))
             for name in ENTITIES}
        )  # fmt: skip

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """``{entity: (N, size)} -> (N, entities, width)``."""
        return torch.stack([self.stems[name](features[name]) for name in ENTITIES], dim=1)


class FreeLatentEncoder(nn.Module):
    """Shared by every body: attention across the entity tokens, then the posterior of ``ζ``."""

    def __init__(self, dims: FreeLatentDims) -> None:
        super().__init__()
        self.dims = dims
        self.embedding = nn.Parameter(torch.zeros(len(ENTITIES), dims.width))
        layer = nn.TransformerEncoderLayer(
            dims.width, dims.heads, 4 * dims.width, dropout=0.0, batch_first=True,
            norm_first=True,
        )  # fmt: skip
        self.transformer = nn.TransformerEncoder(layer, dims.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dims.width)
        self.posterior = nn.Linear(len(ENTITIES) * dims.width, 2 * dims.free)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, entities, width) -> (mean, log_variance)``, each ``(N, free)``."""
        mixed = self.norm(self.transformer(tokens + self.embedding))
        mean, log_variance = self.posterior(mixed.flatten(1)).chunk(2, dim=-1)
        return mean, log_variance.clamp(-10.0, 5.0)


class PhysicsPredictor(nn.Module):
    """Shared by every body: from ``(z*, ζ)``, the balance and the logit of falling soon."""

    def __init__(self, dims: FreeLatentDims, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims.skill + dims.free, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dims.balance + 1),
        )
        self.log_variance = nn.Parameter(torch.zeros(dims.balance))

    def forward(self, skill: torch.Tensor, free: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([skill, free], dim=-1))
        return out[..., :-1], out[..., -1]


class BodyClassifier(nn.Module):
    """Which body a ``ζ`` came from: the constraint's witness."""

    def __init__(self, dims: FreeLatentDims, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims.free, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, free: torch.Tensor) -> torch.Tensor:
        return self.net(free)[..., 0]


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.scale * grad, None


def reverse_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity forwards; backwards, the gradient times ``-scale`` (Ganin and Lempitsky 2015)."""
    return _Reverse.apply(x, scale)


@dataclass
class DualAscent:
    """A Lagrange multiplier for ``g <= bound``, raised while the constraint is violated."""

    rate: float = 0.05
    value: float = 0.0

    def step(self, violation: float) -> float:
        self.value = max(0.0, self.value + self.rate * violation)
        return self.value


def accuracy_excess(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """How much better than chance a classifier is, and the standard error of its accuracy."""
    correct = ((logits > 0).float() == labels).float()
    accuracy = float(correct.mean())
    se = math.sqrt(max(accuracy * (1 - accuracy), 1e-6) / max(len(correct), 1))
    chance = float(max(labels.mean(), 1 - labels.mean()))
    return accuracy - chance, se


def gaussian_kl(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """``KL(N(mean, var) || N(0, I))`` per sample, summed over ``ζ``."""
    return 0.5 * (mean.pow(2) + log_variance.exp() - 1.0 - log_variance).sum(dim=-1)


def prediction_loss(
    predictor: PhysicsPredictor,
    skill: torch.Tensor,
    free: torch.Tensor,
    balance: torch.Tensor,
    falls: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gaussian NLL of the balance (learned variances), binary cross-entropy of the falls."""
    predicted, logit = predictor(skill, free)
    log_variance = predictor.log_variance.clamp(math.log(0.01), math.log(10.0))
    squared = (predicted - balance).pow(2).mean(dim=0)
    nll = (0.5 * torch.exp(-log_variance) * squared + 0.5 * log_variance).mean()
    fall = nn.functional.binary_cross_entropy_with_logits(logit, falls.float())
    return {"balance": nll, "falls": fall}


def alignment_loss(free_a: torch.Tensor, free_b: torch.Tensor) -> torch.Tensor:
    """Paired states' ``ζ`` should agree: their mean squared distance, scale-free.

    Divided by the spread of ``ζ`` over the batch, so shrinking ``ζ`` to a point does not
    satisfy it -- the KL budget keeps that spread near one.
    """
    spread = torch.cat([free_a, free_b]).var(dim=0).mean().clamp(min=1e-6)
    return (free_a - free_b).pow(2).mean() / spread


def entity_features(
    encoder_input: torch.Tensor,
    skill: torch.Tensor,
    joint_legs: tuple[Sequence[int], Sequence[int]],
    action_legs: tuple[Sequence[int], Sequence[int]],
    joints: int,
    skill_slices: dict[str, slice],
) -> dict[str, torch.Tensor]:
    """A body's sensed features, grouped by entity.

    ``encoder_input`` is ``ReferenceCommand.encoder_input``: base linear and angular
    velocity, gravity, then the ``joints`` positions and velocities, then the last
    action. The trunk reads the base and the CoM's fields of ``z*``; each foot reads
    its leg's joints and actions and its own fields of ``z*`` (position, velocity, load).

    Args:
        joint_legs: the left and the right leg's indices among the ``joints``.
        action_legs: the same legs' indices in the action.
        skill_slices: ``SKILL_LAYOUT.slice_of`` for every field.
    """
    base = encoder_input[..., :9]
    position = encoder_input[..., 9 : 9 + joints]
    velocity = encoder_input[..., 9 + joints : 9 + 2 * joints]
    action = encoder_input[..., 9 + 2 * joints :]
    trunk_fields = ("height", "com_velocity", "gravity", "base_angular_velocity",
                    "angular_momentum")  # fmt: skip
    trunk = torch.cat([base] + [skill[..., skill_slices[f]] for f in trunk_fields], dim=-1)
    features = {"trunk": trunk}
    device = encoder_input.device
    for side, leg, acted in zip(("left", "right"), joint_legs, action_legs, strict=True):
        index = torch.as_tensor(list(leg), device=device)
        driven = torch.as_tensor(list(acted), device=device)
        mine = []
        for field in ("foot_position", "foot_velocity", "foot_load"):
            both = skill[..., skill_slices[field]]
            half = both.shape[-1] // 2
            mine.append(both[..., :half] if side == "left" else both[..., half:])
        features[f"{side}_foot"] = torch.cat(
            [position[..., index], velocity[..., index], action[..., driven], *mine], dim=-1
        )
    return features


__all__ = [
    "ENTITIES",
    "BodyClassifier",
    "BodyStems",
    "DualAscent",
    "FreeLatentDims",
    "FreeLatentEncoder",
    "PhysicsPredictor",
    "accuracy_excess",
    "alignment_loss",
    "entity_features",
    "gaussian_kl",
    "prediction_loss",
    "reverse_gradient",
]
