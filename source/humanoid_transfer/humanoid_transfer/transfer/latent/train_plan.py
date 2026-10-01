"""Train the planned skill and ask whether its schedule is calibrated and used.

The two stages of :mod:`.train_memory` are kept: the encoder learns the
anchor alone (stage A, reused as is), and the shared modules -- memory,
planner, policy, dynamics -- and the decoder train on the true ``z*`` plus
noise of the encoder's size (stage B, here with the planner in the loop).

Evaluation adds the questions the schedule raises:

* **Is it calibrated?** Per contact, the planned probability that the change
  has come by each bin against how often it had. A sampled schedule is only
  the teacher's if this holds.
* **Is it informative?** Its likelihood against the population hazard, which
  knows the teacher's typical timing and nothing about the moment.
* **Is it used?** Imitation with the schedule that followed against the same
  with the schedule held at its mean.
* **What does running it give?** Imitation along held-out rollouts with the
  exported policy itself: drawing, holding and redrawing step by step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from ...common.skill_state import DEFAULT_CONTACT_THRESHOLD, SKILL_LAYOUT
from .events import ContactEvents
from .memory import MemorySkill, StateEncoder
from .model import ObservationLayout
from .plan import (
    ContactPlanner,
    EventClock,
    PlannedPolicy,
    calibration,
    fit_landing_normaliser,
    marginal_hazard,
    median_bin,
    plan_features,
    plan_targets,
    planned_losses,
    sample_bins,
    survival_nll,
)
from .probes import r2
from .sequences import Streams
from .train_memory import MemoryConfig, _imitation_r2, _optimise

FEET = SKILL_LAYOUT.slice_of("foot_position")
LOAD = SKILL_LAYOUT.slice_of("foot_load")
PARTS = 2
"""The contacts the skill state carries: the feet."""


@dataclass(frozen=True)
class PlanConfig(MemoryConfig):
    bins: int = 26
    """``K``: the planner's horizon in grid steps, about 2 ``T`` like the memory."""

    plan_weight: float = 1.0


def plan_config(**kwargs) -> PlanConfig:
    config = PlanConfig(**kwargs)
    return replace(config, dims=replace(config.dims, plan=3 * PARTS))


@dataclass
class PlannedModel:
    encoder: StateEncoder
    skill: MemorySkill
    planner: ContactPlanner
    decoder: nn.Module
    layout: ObservationLayout
    sigma: torch.Tensor

    def modules(self) -> list[nn.Module]:
        return [self.encoder, self.skill, self.planner, self.decoder]

    def to(self, device: torch.device | str) -> PlannedModel:
        for module in self.modules():
            module.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def eval(self) -> PlannedModel:
        for module in self.modules():
            module.eval()
        return self

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "skill": self.skill.state_dict(),
            "planner": self.planner.state_dict(),
            "decoder": self.decoder.state_dict(),
            "sigma": self.sigma.cpu(),
        }

    @property
    def clock(self) -> EventClock:
        return EventClock(step=self.planner.step, bins=self.planner.bins)

    def policy(
        self,
        velocity_unit: float,
        time_unit: float,
        dt_star: float,
        plan: torch.Tensor | None = None,
        stochastic: bool = True,
    ) -> PlannedPolicy:
        return PlannedPolicy(
            self.encoder,
            self.skill,
            self.planner,
            self.decoder,
            self.layout,
            velocity_unit,
            time_unit,
            dt_star,
            threshold=DEFAULT_CONTACT_THRESHOLD,
            load=LOAD,
            plan=plan,
            stochastic=stochastic,
        )


def new_planner(
    events: ContactEvents, steps: torch.Tensor, config: PlanConfig, dt_star: float
) -> ContactPlanner:
    torch.manual_seed(config.seed + 2)
    planner = ContactPlanner(config.dims, PARTS, EventClock(step=dt_star, bins=config.bins))
    planner = planner.to(events.contact.device)
    fit_landing_normaliser(planner, events, steps)
    return planner


def train_planned(
    streams: Streams,
    events: ContactEvents,
    hindsight: torch.Tensor,
    steps: torch.Tensor,
    layout: ObservationLayout,
    skill: MemorySkill,
    planner: ContactPlanner,
    decoder: nn.Module,
    sigma: torch.Tensor,
    config: PlanConfig,
    dt_star: float,
) -> list[dict[str, float]]:
    """Stage B with the planner. ``steps`` must keep ``config.horizon`` successors."""
    clock = EventClock(step=planner.step, bins=planner.bins)
    modules = [skill, planner, decoder]
    parameters = [p for module in modules for p in module.parameters()]

    def loss(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return planned_losses(
            skill, planner, decoder, layout, streams, events, hindsight, batch, sigma,
            config.horizon, config.dynamics, config.plan_weight, clock, dt_star, FEET,
        )  # fmt: skip

    for module in modules:
        module.train()
    history = _optimise(parameters, steps, loss, config.skill_epochs, config)
    for module in modules:
        module.eval()
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def schedule_inputs(
    model: PlannedModel, streams: Streams, source: torch.Tensor, steps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(φ, hazard logits, landing mean)`` at ``steps`` from ``source`` as the history."""
    memory = model.skill.remember(source[streams.windows(steps, model.skill.samples)])
    command = model.skill.command_norm(streams.command[steps])
    logits, mean, _ = model.planner(source[steps], memory, command)
    return memory, logits, mean


@torch.no_grad()
def hindsight_plans(
    model: PlannedModel, streams: Streams, events: ContactEvents, steps: torch.Tensor, dt_star
) -> torch.Tensor:
    return plan_targets(model.planner, streams, events, steps, model.clock, dt_star, FEET).plan


@torch.no_grad()
def population_hazard(
    model: PlannedModel, streams: Streams, events: ContactEvents, steps: torch.Tensor, dt_star
) -> torch.Tensor:
    """``(P, K)`` logits of the population hazard on ``steps``."""
    targets = plan_targets(model.planner, streams, events, steps, model.clock, dt_star, FEET)
    return marginal_hazard(targets.event_bin, targets.observed, model.planner.bins)


@torch.no_grad()
def evaluate_plan(
    model: PlannedModel,
    streams: Streams,
    events: ContactEvents,
    steps: torch.Tensor,
    estimate: torch.Tensor,
    dt_star: float,
    population: torch.Tensor,
    mean_plan: torch.Tensor,
    mean_action: torch.Tensor,
) -> dict:
    """Calibration, information and use of the schedule, and imitation through it."""
    bins = model.planner.bins
    truth = model.skill.skill_norm(streams.skill)
    targets = plan_targets(model.planner, streams, events, steps, model.clock, dt_star, FEET)
    memory, logits, mean = schedule_inputs(model, streams, estimate, steps)

    ece = calibration(logits, targets.event_bin, targets.observed)
    nll = survival_nll(logits, targets.event_bin, targets.observed).mean(dim=0)
    reference = survival_nll(
        population.expand_as(logits), targets.event_bin, targets.observed
    ).mean(dim=0)
    within = targets.event_bin <= bins
    median = median_bin(logits).float()
    timing = [
        float(
            r2(median[within[:, p], p, None], targets.event_bin[within[:, p], p, None].float())[0]
        )
        for p in range(model.planner.parts)
    ]
    known = targets.landing_known
    landing = [
        float(np.nanmedian(r2(mean[known[:, p], p], targets.landing_n[known[:, p], p])))
        for p in range(model.planner.parts)
    ]

    context = model.layout.context_input(streams.obs[steps])
    target = streams.action[steps]
    command = streams.command[steps]

    def imitation(state: torch.Tensor, phi: torch.Tensor, plan: torch.Tensor) -> float:
        action = model.skill.act(state, phi, command, plan)
        return _imitation_r2(model.decoder(action, context), target)

    drawn = plan_features(
        model.planner.remaining(
            sample_bins(logits, torch.rand(logits.shape[:-1], device=logits.device))
        ),
        mean,
        model.planner.horizon,
    )
    true_memory = model.skill.remember(truth[streams.windows(steps, model.skill.samples)])
    frozen = mean_plan.expand(len(steps), -1)
    with_plan = imitation(estimate[steps], memory, targets.plan)
    without = imitation(estimate[steps], memory, frozen)
    return {
        "samples": len(steps),
        "calibration_error": ece.tolist(),
        "hazard_nll": nll.tolist(),
        "population_hazard_nll": reference.tolist(),
        "nll_gain_nats": (reference - nll).tolist(),
        "median_bin_r2": timing,
        "landing_r2_median": landing,
        "imitation_r2_followed_plan": with_plan,
        "imitation_r2_drawn_plan": imitation(estimate[steps], memory, drawn),
        "imitation_r2_frozen_plan": without,
        "plan_use": with_plan - without,
        "imitation_r2_true_state": imitation(truth[steps], true_memory, targets.plan),
        "bypass_action_r2": _imitation_r2(
            model.decoder(mean_action.expand(len(steps), -1), context), target
        ),
    }


@torch.no_grad()
def runtime_imitation(
    policy: PlannedPolicy,
    obs: torch.Tensor,
    done: torch.Tensor,
    action: torch.Tensor,
    valid: torch.Tensor,
    seed: int = 0,
) -> float:
    """Imitation R² of the running policy along ``(T, N, ...)`` rollouts, on ``valid`` steps."""
    torch.manual_seed(seed)
    outputs = []
    for t in range(obs.shape[0]):
        outputs.append(policy(obs[t]))
        policy.reset(done[t])
    predicted = torch.stack(outputs)
    return _imitation_r2(predicted[valid], action[valid])


def column_rollouts(
    streams: Streams, capture_steps: int, envs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(obs, action, stream index)`` of some environments of the first capture, ``(T, N, ...)``,
    and whether each step is a trustworthy, walking one."""
    index = (envs[None, :] * capture_steps) + torch.arange(capture_steps, device=envs.device)[
        :, None
    ]
    trustworthy = index >= streams.start[index]
    valid = trustworthy & ~streams.falling[index]
    return streams.obs[index], streams.action[index], index, valid


__all__ = [
    "FEET",
    "LOAD",
    "PARTS",
    "PlanConfig",
    "PlannedModel",
    "column_rollouts",
    "evaluate_plan",
    "hindsight_plans",
    "new_planner",
    "plan_config",
    "population_hazard",
    "runtime_imitation",
    "schedule_inputs",
    "train_planned",
]
