"""Train the G1 side of the shared space, and measure what it learned.

Training is offline distillation from the teacher's rollouts. Evaluation asks
three questions, none of which the training loss answers by itself:

**Does the factored policy reproduce the teacher?** Action error on held-out
rollouts. (Whether it *walks* is a closed-loop question, answered in the
simulator by ``g1/scripts/eval_latent.py``.)

**Does the skill flow through the shared space?** The same error with the
latent action ``u`` replaced by its mean. If the decoder can still imitate the
teacher, ``u`` carries nothing and the "shared skill" is decoration.

**What does the input of the skill policy know?** Linear probes from
``(ẑ*, ζ)`` to the physical skill state (should be high: the space means
something physical) and to the G1's own joint configuration (an index of how
much of this particular body the skill sees).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .dataset import Transitions
from .losses import LossWeights, latent_losses
from .model import (
    BodyDecoder,
    BodyEncoder,
    LatentDims,
    LatentPolicy,
    ObservationLayout,
    SharedSkill,
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    batch_size: int = 4096
    learning_rate: float = 1e-3
    seed: int = 0
    weights: LossWeights = field(default_factory=LossWeights)
    dims: LatentDims = field(default_factory=LatentDims)


@dataclass
class LatentModel:
    encoder: BodyEncoder
    skill: SharedSkill
    decoder: BodyDecoder
    layout: ObservationLayout

    def modules(self) -> list[torch.nn.Module]:
        return [self.encoder, self.skill, self.decoder]

    def parameters(self):
        for module in self.modules():
            yield from module.parameters()

    def to(self, device: torch.device | str) -> LatentModel:
        for module in self.modules():
            module.to(device)
        return self

    def train(self, mode: bool = True) -> None:
        for module in self.modules():
            module.train(mode)

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "skill": self.skill.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def policy(self, velocity_unit: float, time_unit: float) -> LatentPolicy:
        return LatentPolicy(
            self.encoder, self.skill, self.decoder, self.layout, velocity_unit, time_unit
        )


def build(
    layout: ObservationLayout, actions: int, dims: LatentDims, data: Transitions
) -> LatentModel:
    """Create the modules and fit every normaliser on the training data."""
    model = LatentModel(
        encoder=BodyEncoder(layout.encoder_size, dims),
        skill=SharedSkill(dims),
        decoder=BodyDecoder(layout.context_size, actions, dims),
        layout=layout,
    )
    model.encoder.obs_norm.fit(layout.encoder_input(data.obs))
    model.decoder.context_norm.fit(layout.context_input(data.obs))
    model.skill.skill_norm.fit(data.skill)
    model.skill.command_norm.fit(data.command)
    return model


def train(
    data: Transitions, layout: ObservationLayout, config: TrainConfig, device: str = "cpu"
) -> tuple[LatentModel, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    model = build(layout, data.action.shape[-1], config.dims, data).to(device)
    data = data.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    steps_per_epoch = max(1, math.ceil(len(data) / config.batch_size))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.epochs * steps_per_epoch
    )

    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(len(data), generator=generator).to(device)
        sums: dict[str, float] = {}
        for start in range(0, len(data), config.batch_size):
            batch = data.subset(order[start : start + config.batch_size])
            losses = latent_losses(
                model.encoder, model.skill, model.decoder, layout, batch, config.weights
            )
            optimiser.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimiser.step()
            schedule.step()
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) / steps_per_epoch
        history.append({"epoch": epoch, **sums})
    model.train(False)
    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = ((prediction - target) ** 2).sum(axis=0)
    total = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 1e-12, 1.0 - residual / total, np.nan)


def linear_probe(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
) -> np.ndarray:
    """Per-output R^2 of a least-squares map from ``x`` to ``y``, fitted and tested apart."""
    design = np.hstack([train_x, np.ones((train_x.shape[0], 1))])
    weights, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    prediction = np.hstack([test_x, np.ones((test_x.shape[0], 1))]) @ weights
    return _r2(prediction, test_y)


@torch.no_grad()
def latents(model: LatentModel, data: Transitions) -> dict[str, torch.Tensor]:
    obs = data.obs.to(next(model.encoder.parameters()).device)
    command = data.command.to(obs.device)
    skill_n, mean, _ = model.encoder(model.layout.encoder_input(obs))
    action = model.skill.act(skill_n, mean, command)
    return {"skill_n": skill_n, "free": mean, "action": action, "obs": obs, "command": command}


@torch.no_grad()
def evaluate(
    model: LatentModel,
    data: Transitions,
    probe_fit: Transitions,
    body_joints: slice,
    weights: LossWeights,
) -> dict:
    """Imitation, bypass, anchor quality and the two probes, on ``data``."""
    device = next(model.encoder.parameters()).device
    out = latents(model, data)
    context = model.layout.context_input(out["obs"])
    target = data.action.to(device)

    imitation = torch.mean((model.decoder(out["action"], context) - target) ** 2)
    frozen = out["action"].mean(dim=0, keepdim=True).expand_as(out["action"])
    bypass = torch.mean((model.decoder(frozen, context) - target) ** 2)
    baseline = torch.mean((target - target.mean(dim=0)) ** 2)

    skill_hat = model.skill.skill_norm.inverse(out["skill_n"]).cpu().numpy()
    skill_true = data.skill.numpy()

    fit = latents(model, probe_fit)
    fit_input = torch.cat([fit["skill_n"], fit["free"]], dim=-1).cpu().numpy()
    test_input = torch.cat([out["skill_n"], out["free"]], dim=-1).cpu().numpy()
    joints_fit = probe_fit.obs[:, body_joints].numpy()
    joints_test = data.obs[:, body_joints].numpy()

    anchor_r2 = _r2(skill_hat, skill_true)
    anchor_rmse = np.sqrt(((skill_hat - skill_true) ** 2).mean(axis=0))
    # What the physical state itself says about the joints: the floor any
    # latent that encodes it will reach. Leakage is the excess over it.
    reference_probe = linear_probe(probe_fit.skill.numpy(), joints_fit, skill_true, joints_test)
    physical_probe = linear_probe(fit_input, probe_fit.skill.numpy(), test_input, skill_true)
    joint_probe = linear_probe(fit_input, joints_fit, test_input, joints_test)
    return {
        "samples": len(data),
        "imitation_mse": float(imitation),
        "action_variance": float(baseline),
        "imitation_r2": float(1.0 - imitation / baseline),
        "bypass_mse": float(bypass),
        "bypass_r2": float(1.0 - bypass / baseline),
        "anchor_r2": anchor_r2.tolist() if weights.anchored else None,
        "anchor_r2_median": float(np.nanmedian(anchor_r2)) if weights.anchored else None,
        "anchor_r2_min": float(np.nanmin(anchor_r2)) if weights.anchored else None,
        "anchor_rmse_body_units": anchor_rmse.tolist() if weights.anchored else None,
        "probe_skill_state_r2_median": float(np.nanmedian(physical_probe)),
        "probe_body_joints_r2_median": float(np.nanmedian(joint_probe)),
        "probe_body_joints_from_true_skill_state_r2_median": float(np.nanmedian(reference_probe)),
    }


def config_dict(config: TrainConfig) -> dict:
    return asdict(config)


__all__ = [
    "LatentModel",
    "TrainConfig",
    "build",
    "config_dict",
    "evaluate",
    "latents",
    "linear_probe",
    "train",
]
