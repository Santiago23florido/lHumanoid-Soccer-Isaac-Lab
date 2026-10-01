"""Train the memory model in two stages, and measure what it learned.

**Stage A** trains the encoder on the anchor alone and measures its error on
held-out walking, dimension by dimension. **Stage B** trains the shared
modules and the decoder on the *true* skill state plus noise of that measured
size; the encoder is not in the graph. The split is what keeps the body out
of the shared space: the encoder has no reason to hide anything in its
errors, and the skill policy never learns to read them.

Evaluation asks the memoryless model's questions on the path that actually
runs -- encoder, buffer, memory, policy, decoder -- and two more:

* **Is the memory used?** Imitation with ``φ`` held at its mean.
* **Did anything leak?** Probes for the body's joints from the skill
  policy's input, against the same probes from the *true* physical history.
  By construction the memory cannot know more about the joints than that
  history; any excess has come in through the encoder's errors.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from .limbs import LimbDecoder, g1_limbs
from .memory import (
    MemoryDims,
    MemoryPolicy,
    MemorySkill,
    StateEncoder,
    TimeGrid,
    anchor_loss,
    skill_losses,
)
from .model import BodyDecoder, LatentDims, Normaliser, ObservationLayout
from .probes import linear_probe, mlp_probe, r2
from .sequences import Streams


@dataclass(frozen=True)
class MemoryConfig:
    encoder_epochs: int = 150
    skill_epochs: int = 150
    batch_size: int = 4096
    learning_rate: float = 1e-3
    horizon: int = 8
    """``k``: steps ``T`` is rolled out, about 0.6 ``T`` on the G1."""

    dynamics: float = 0.5
    samples: int = 26
    """The memory's grid: about 2 ``T`` on the G1's control step."""

    limbs: bool = False
    """Decode by limb (:class:`~.limbs.LimbDecoder`) instead of all joints at once."""

    limb_hidden: int = 128
    seed: int = 0
    dims: MemoryDims = field(default_factory=MemoryDims)


@dataclass
class MemoryModel:
    encoder: StateEncoder
    skill: MemorySkill
    decoder: nn.Module
    layout: ObservationLayout
    sigma: torch.Tensor
    """The encoder's held-out error per dimension, normalised units: the training noise."""

    def modules(self) -> list[nn.Module]:
        return [self.encoder, self.skill, self.decoder]

    def to(self, device: torch.device | str) -> MemoryModel:
        for module in self.modules():
            module.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def eval(self) -> MemoryModel:
        for module in self.modules():
            module.eval()
        return self

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "skill": self.skill.state_dict(),
            "decoder": self.decoder.state_dict(),
            "sigma": self.sigma.cpu(),
        }

    def policy(
        self,
        velocity_unit: float,
        time_unit: float,
        dt_star: float,
        memory: torch.Tensor | None = None,
    ) -> MemoryPolicy:
        return MemoryPolicy(
            self.encoder,
            self.skill,
            self.decoder,
            self.layout,
            velocity_unit,
            time_unit,
            dt_star,
            memory=memory,
        )


def _optimise(
    parameters: list[nn.Parameter],
    steps: torch.Tensor,
    loss: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    epochs: int,
    config: MemoryConfig,
) -> list[dict[str, float]]:
    optimiser = torch.optim.Adam(parameters, lr=config.learning_rate)
    per_epoch = max(1, math.ceil(len(steps) / config.batch_size))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs * per_epoch)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = steps[torch.randperm(len(steps), generator=generator).to(steps.device)]
        sums: dict[str, float] = {}
        for start in range(0, len(order), config.batch_size):
            losses = loss(order[start : start + config.batch_size])
            optimiser.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimiser.step()
            schedule.step()
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) / per_epoch
        history.append({"epoch": epoch, **sums})
    return history


def fit_skill_normaliser(streams: Streams, steps: torch.Tensor, dims: MemoryDims) -> Normaliser:
    """The shared space's normalisation, fitted once on the training steps."""
    norm = Normaliser(dims.skill).to(streams.skill.device)
    norm.fit(streams.skill[steps])
    return norm


def new_skill(
    streams: Streams,
    steps: torch.Tensor,
    skill_norm: Normaliser,
    config: MemoryConfig,
    dt_star: float,
) -> MemorySkill:
    torch.manual_seed(config.seed)
    skill = MemorySkill(config.dims, TimeGrid(step=dt_star, samples=config.samples))
    skill = skill.to(streams.skill.device)
    skill.skill_norm.load_state_dict(skill_norm.state_dict())
    skill.command_norm.fit(streams.command[steps])
    return skill


def train_encoder(
    streams: Streams,
    steps: torch.Tensor,
    layout: ObservationLayout,
    skill: MemorySkill,
    config: MemoryConfig,
) -> tuple[StateEncoder, list[dict[str, float]]]:
    """Stage A: the encoder, on the anchor alone. ``skill`` only lends its normalisation."""
    torch.manual_seed(config.seed)
    dims = config.dims
    encoder = StateEncoder(layout.encoder_size, dims.skill, dims.hidden).to(streams.obs.device)
    encoder.obs_norm.fit(layout.encoder_input(streams.obs[steps]))

    def loss(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        anchor = anchor_loss(encoder, skill, layout, streams, batch)
        return {"total": anchor, "anchor": anchor}

    history = _optimise(list(encoder.parameters()), steps, loss, config.encoder_epochs, config)
    encoder.eval()
    return encoder, history


@torch.no_grad()
def estimates(
    encoder: StateEncoder, layout: ObservationLayout, streams: Streams, batch: int = 65536
) -> torch.Tensor:
    """The encoder's ``ẑ*`` (normalised) at every step of the streams."""
    return torch.cat(
        [
            encoder(layout.encoder_input(streams.obs[i : i + batch]))
            for i in range(0, len(streams), batch)
        ]
    )


@torch.no_grad()
def residual_sigma(
    encoder: StateEncoder,
    skill: MemorySkill,
    layout: ObservationLayout,
    streams: Streams,
    steps: torch.Tensor,
) -> torch.Tensor:
    """Per-dimension RMS error of the encoder on ``steps``, in normalised units."""
    estimate = encoder(layout.encoder_input(streams.obs[steps]))
    error = estimate - skill.skill_norm(streams.skill[steps])
    return error.pow(2).mean(dim=0).sqrt()


def new_decoder(
    streams: Streams,
    steps: torch.Tensor,
    layout: ObservationLayout,
    config: MemoryConfig,
    joint_names: list[str],
) -> nn.Module:
    torch.manual_seed(config.seed + 1)
    actions = streams.action.shape[-1]
    if config.limbs:
        decoder: nn.Module = LimbDecoder(g1_limbs(joint_names), hidden=config.limb_hidden)
    else:
        dims = LatentDims(action=config.dims.action, hidden=config.dims.hidden)
        decoder = BodyDecoder(layout.context_size, actions, dims)
    decoder = decoder.to(streams.obs.device)
    decoder.context_norm.fit(layout.context_input(streams.obs[steps]))
    return decoder


def train_skill(
    streams: Streams,
    steps: torch.Tensor,
    layout: ObservationLayout,
    skill: MemorySkill,
    decoder: nn.Module,
    sigma: torch.Tensor,
    config: MemoryConfig,
) -> list[dict[str, float]]:
    """Stage B: memory, policy, dynamics and decoder on the true ``z*`` plus noise.

    ``steps`` must keep ``config.horizon`` successors inside their episode.
    """
    parameters = list(skill.parameters()) + list(decoder.parameters())

    def loss(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return skill_losses(
            skill, decoder, layout, streams, batch, sigma, config.horizon, config.dynamics
        )

    skill.train()
    decoder.train()
    history = _optimise(parameters, steps, loss, config.skill_epochs, config)
    skill.eval()
    decoder.eval()
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def act(
    model: MemoryModel,
    streams: Streams,
    source: torch.Tensor,
    steps: torch.Tensor,
    memory: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(φ, u)`` at ``steps``, with ``source`` (normalised ``z*`` per stream step) as history."""
    if memory is None:
        memory = model.skill.remember(source[streams.windows(steps, model.skill.samples)])
    else:
        memory = memory.expand(len(steps), -1)
    return memory, model.skill.act(source[steps], memory, streams.command[steps])


def _imitation_r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    baseline = torch.mean((target - target.mean(dim=0)) ** 2)
    return float(1.0 - torch.mean((prediction - target) ** 2) / baseline)


@torch.no_grad()
def evaluate_memory(
    model: MemoryModel,
    streams: Streams,
    steps: torch.Tensor,
    estimate: torch.Tensor,
    mean_memory: torch.Tensor,
    mean_action: torch.Tensor,
) -> dict:
    """Imitation (deployed and on the true state), the two bypasses and the anchor."""
    truth = model.skill.skill_norm(streams.skill)
    context = model.layout.context_input(streams.obs[steps])
    target = streams.action[steps]

    memory, action = act(model, streams, estimate, steps)
    _, true_action = act(model, streams, truth, steps)
    _, frozen_memory = act(model, streams, estimate, steps, memory=mean_memory)

    anchor = r2(estimate[steps], truth[steps])
    body = model.skill.skill_norm.inverse(estimate[steps])
    rmse = (body - streams.skill[steps]).pow(2).mean(dim=0).sqrt()
    return {
        "samples": len(steps),
        "imitation_r2": _imitation_r2(model.decoder(action, context), target),
        "imitation_r2_true_state": _imitation_r2(model.decoder(true_action, context), target),
        "bypass_action_r2": _imitation_r2(
            model.decoder(mean_action.expand(len(steps), -1), context), target
        ),
        "bypass_memory_r2": _imitation_r2(model.decoder(frozen_memory, context), target),
        "anchor_r2": anchor.tolist(),
        "anchor_r2_median": float(np.nanmedian(anchor)),
        "anchor_r2_min": float(np.nanmin(anchor)),
        "anchor_rmse_body_units": rmse.tolist(),
        "memory_spread": float(memory.std(dim=0).mean()),
    }


def leakage_probes(
    model: MemoryModel,
    streams: Streams,
    fit: torch.Tensor,
    test: torch.Tensor,
    estimate: torch.Tensor,
    body_joints: slice,
    epochs: int = 30,
) -> dict:
    """What the skill policy's input knows about the body's joints, against the physics.

    ``policy_input`` is ``(ẑ*, φ)`` as deployed. ``physical_history`` is the
    true ``z*`` window the memory reads -- the most the memory could know about
    the joints with a perfect encoder. ``physical_state`` is the true ``z*``
    now, the memoryless model's reference. Leakage is the policy input's
    excess over the physical history.
    """
    truth = model.skill.skill_norm(streams.skill)
    joints = streams.obs[:, body_joints]
    samples = model.skill.samples

    def policy_input(idx: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            memory = model.skill.remember(estimate[streams.windows(idx, samples)])
        return torch.cat([estimate[idx], memory], dim=-1)

    def physical_history(idx: torch.Tensor) -> torch.Tensor:
        return truth[streams.windows(idx, samples)].flatten(1)

    def physical_state(idx: torch.Tensor) -> torch.Tensor:
        return truth[idx]

    rows: dict = {}
    for name, features in (
        ("policy_input", policy_input),
        ("physical_history", physical_history),
        ("physical_state", physical_state),
    ):
        linear = linear_probe(features, joints, fit, test)
        nonlinear = mlp_probe(features, joints, fit, test, epochs=epochs)
        rows[name] = {
            "linear_r2_median": float(np.nanmedian(linear)),
            "mlp_r2_median": float(np.nanmedian(nonlinear)),
        }
    rows["excess_linear"] = (
        rows["policy_input"]["linear_r2_median"] - rows["physical_history"]["linear_r2_median"]
    )
    rows["excess_mlp"] = (
        rows["policy_input"]["mlp_r2_median"] - rows["physical_history"]["mlp_r2_median"]
    )
    return rows


def config_dict(config: MemoryConfig) -> dict:
    return asdict(config)


__all__ = [
    "MemoryConfig",
    "MemoryModel",
    "act",
    "config_dict",
    "estimates",
    "evaluate_memory",
    "fit_skill_normaliser",
    "leakage_probes",
    "new_decoder",
    "new_skill",
    "residual_sigma",
    "train_encoder",
    "train_skill",
]
