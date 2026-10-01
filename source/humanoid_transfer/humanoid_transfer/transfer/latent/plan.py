"""A shared planner for the contact schedule: when each contact changes, where it lands.

The memory model walked, but its stride drifted: nothing in its objective
pins down *when* a foot leaves the ground, so speed was matched by whatever
split of stride length and cadence imitation errors drifted into. This module
makes the schedule an explicit, shared part of the skill, without assuming a
gait -- a contact is an event, not a point on a cycle::

    (ẑ*, φ, c*) ──P──► per contact: hazard over the next K bins of body time
                                    + where its next landing will be
                         │ sample once, then commit: count down in body time
                         ▼
    π(ẑ*, φ, c*, plan) ──► u ──► D_r

**When** is a discrete-time hazard: for each part and each bin ``k`` of
width ``Δ*`` (the memory's grid step), the probability that the part's
contact changes in bin ``k`` given it has not before. A distribution, not a
number, because the teacher's timing is not a clock -- the present predicts
its next landing only to R² 0.59 -- and a regression to the mean would plan
a stride the teacher never takes. Rollouts that end before the event are
censored, not discarded.

**Where** is the next landing's sole relative to the CoM, heading frame, leg
lengths, as a Gaussian.

**Commitment.** At run time a time is *sampled* and then held: it counts
down by ``dt*`` each step and is redrawn only when the contact actually
changes, when it falls due, or while no change is expected within the
horizon. The policy then has a schedule to execute instead of re-deciding the
stride every step -- which is how a 2 % imitation error became a 29 % stride
error.

**What the policy trains on.** On rollouts the teacher drove, the plan is
the schedule that actually followed (hindsight): the policy learns to execute
a schedule it is given. On rollouts a student drove (DAgger), the schedule
that followed was the student's, not the teacher's, so it would pair one
body's timing with another's actions; there the policy is given a schedule
sampled from the planner, and the planner does not learn from them.

Everything the planner reads and is trained on comes from ``z*`` and the
command, so the guarantee of :mod:`.memory` holds: nothing about the body
reaches the shared space through the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .events import ContactEvents
from .memory import MemoryDims, MemorySkill
from .model import Normaliser, ObservationLayout, mlp
from .sequences import Streams

_TOLERANCE = 1e-6
_LOG_STD_RANGE = (-5.0, 2.0)


@dataclass(frozen=True)
class EventClock:
    """``bins`` bins of ``step`` body times each, from now: the planner's horizon."""

    step: float
    bins: int

    @property
    def horizon(self) -> float:
        return self.bins * self.step

    def event_bin(self, steps: torch.Tensor, dt_star: float) -> torch.Tensor:
        """The bin, ``1..bins``, of an event ``steps`` control steps ahead; ``bins + 1`` if
        later or never."""
        finite = torch.isfinite(steps)
        bins = torch.ceil(torch.where(finite, steps, 0.0) * dt_star / self.step - _TOLERANCE)
        inside = finite & (bins <= self.bins)
        return torch.where(inside, bins, self.bins + 1).long()

    def observed_bins(self, steps: torch.Tensor, dt_star: float) -> torch.Tensor:
        """How many whole bins of the future ``steps`` control steps cover, at most ``bins``."""
        whole = torch.floor(steps * dt_star / self.step + _TOLERANCE)
        return whole.clamp(max=self.bins).long()


def plan_features(remaining: torch.Tensor, landing_n: torch.Tensor, horizon: float) -> torch.Tensor:
    """The policy's view of a plan: per part, time left as a fraction of the horizon
    (1 for "not within it"), then the normalised landing, ``(..., 3 P)``."""
    fraction = (remaining / horizon).clamp(max=1.0)
    return torch.cat([fraction, landing_n.flatten(-2)], dim=-1)


class ContactPlanner(nn.Module):
    """``P``: ``(ẑ*, φ, c*)`` in the shared normalised units, to a contact schedule.

    Returns, per part, ``bins`` hazard logits and the mean and log standard
    deviation of the next landing in normalised units (``landing_norm``).
    """

    def __init__(self, dims: MemoryDims, parts: int, clock: EventClock) -> None:
        super().__init__()
        self.parts = parts
        self.bins = clock.bins
        self.step = clock.step
        self.horizon = clock.horizon
        self.landing_norm = Normaliser(parts * 2)
        self.net = mlp(
            dims.skill + dims.memory + dims.command, parts * (clock.bins + 4), dims.hidden
        )

    @property
    def width(self) -> int:
        """Width of the plan the policy reads."""
        return 3 * self.parts

    def forward(
        self, skill_n: torch.Tensor, memory: torch.Tensor, command_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([skill_n, memory, command_n], dim=-1))
        out = out.reshape(list(out.shape[:-1]) + [self.parts, self.bins + 4])
        logits = out[..., : self.bins]
        mean = out[..., self.bins : self.bins + 2]
        log_std = out[..., self.bins + 2 :].clamp(_LOG_STD_RANGE[0], _LOG_STD_RANGE[1])
        return logits, mean, log_std

    def remaining(self, bins: torch.Tensor) -> torch.Tensor:
        """Body time to the end of each sampled bin; past the horizon for "not within it"."""
        return torch.where(bins <= self.bins, bins.float() * self.step, self.horizon + self.step)


def cdf(logits: torch.Tensor) -> torch.Tensor:
    """``(..., K)``: probability that the change has happened by the end of each bin."""
    return 1.0 - torch.exp(torch.cumsum(-F.softplus(logits), dim=-1))


def sample_bins(logits: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw of the event bin, ``1..K``, or ``K + 1`` when it falls past the horizon."""
    before = (cdf(logits) < uniform.unsqueeze(-1)).sum(dim=-1)
    return before + 1


def survival_nll(
    logits: torch.Tensor, event_bin: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of a discrete-time hazard, with censoring.

    An event in bin ``e <= K`` contributes ``log h_e + sum_{j<e} log(1 - h_j)``;
    no event seen through ``observed`` bins contributes
    ``sum_{j<=observed} log(1 - h_j)``.
    """
    bins = logits.shape[-1]
    j = torch.arange(1, bins + 1, device=logits.device)
    event = (event_bin <= bins).unsqueeze(-1)
    survived = torch.where(event, j < event_bin.unsqueeze(-1), j <= observed.unsqueeze(-1)).float()
    hit = (event & (j == event_bin.unsqueeze(-1))).float()
    log_survive = -F.softplus(logits)
    log_hazard = -F.softplus(-logits)
    return -(survived * log_survive + hit * log_hazard).sum(dim=-1)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Per point, summed over the last dimension, without the constant."""
    return (0.5 * ((target - mean) * torch.exp(-log_std)) ** 2 + log_std).sum(dim=-1)


# ---------------------------------------------------------------------------
# Labels as the planner and the policy see them
# ---------------------------------------------------------------------------


@dataclass
class PlanTargets:
    event_bin: torch.Tensor
    observed: torch.Tensor
    landing_n: torch.Tensor
    """Normalised landing; where unknown, the part's current position."""

    landing_known: torch.Tensor
    plan: torch.Tensor
    """The schedule that followed, as the policy reads it."""


def plan_targets(
    planner: ContactPlanner,
    streams: Streams,
    events: ContactEvents,
    steps: torch.Tensor,
    clock: EventClock,
    dt_star: float,
    feet: slice,
) -> PlanTargets:
    """The hindsight schedule at ``steps`` (any shape): labels for ``P``, input for ``π``."""
    to_change = events.to_change[steps]
    event_bin = clock.event_bin(to_change, dt_star)
    observed = clock.observed_bins(events.remaining[steps], dt_star).unsqueeze(-1)
    observed = observed.expand_as(event_bin)

    now = streams.skill[steps][..., feet].reshape(list(steps.shape) + [planner.parts, 3])
    known = events.landing_known[steps]
    landing = torch.where(known.unsqueeze(-1), events.landing[steps], now[..., :2])
    landing_n = planner.landing_norm(landing.flatten(-2)).reshape(landing.shape)

    remaining = torch.where(
        event_bin <= clock.bins, to_change * dt_star, torch.full_like(to_change, clock.horizon)
    )
    return PlanTargets(
        event_bin=event_bin,
        observed=observed,
        landing_n=landing_n,
        landing_known=known,
        plan=plan_features(remaining, landing_n, clock.horizon),
    )


def fit_landing_normaliser(
    planner: ContactPlanner, events: ContactEvents, steps: torch.Tensor
) -> None:
    known = events.landing_known[steps].all(dim=-1)
    planner.landing_norm.fit(events.landing[steps][known].flatten(-2))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def planned_losses(
    skill: MemorySkill,
    planner: ContactPlanner,
    decoder: nn.Module,
    layout: ObservationLayout,
    streams: Streams,
    events: ContactEvents,
    hindsight: torch.Tensor,
    steps: torch.Tensor,
    sigma: torch.Tensor,
    horizon: int,
    dynamics_weight: float,
    plan_weight: float,
    clock: EventClock,
    dt_star: float,
    feet: slice,
) -> dict[str, torch.Tensor]:
    """:func:`~.memory.skill_losses` with a contact plan in the loop.

    ``hindsight`` is per stream step: True where the teacher drove the
    rollout, so the schedule that followed is the teacher's. Only there does
    the planner learn, and only there does the policy read the schedule that
    followed; elsewhere it reads one sampled from the planner.
    """
    samples = skill.samples
    span = samples + horizon - 1
    positions = streams.windows(steps + horizon - 1, span)
    offsets = positions - (steps[:, None] - samples + 1)
    noise = torch.randn(len(steps), span, sigma.shape[0], device=sigma.device)
    noise = torch.gather(noise, 1, offsets[..., None].expand_as(noise))
    trajectory = skill.skill_norm(streams.skill[positions]) + sigma * noise

    windows = trajectory.unfold(1, samples, 1).transpose(-1, -2)
    current = trajectory[:, samples - 1 :]
    acting = steps[:, None] + torch.arange(horizon, device=steps.device)
    memory = skill.remember(windows)
    command = streams.command[acting]

    logits, mean, log_std = planner(current, memory, skill.command_norm(command))
    targets = plan_targets(planner, streams, events, acting, clock, dt_star, feet)
    teacher = hindsight[acting]
    weight = teacher.unsqueeze(-1).float()
    count = weight.sum().clamp(min=1.0) * planner.parts
    survival = (survival_nll(logits, targets.event_bin, targets.observed) * weight).sum() / count
    placed = (targets.landing_known & teacher.unsqueeze(-1)).float()
    landing = (gaussian_nll(targets.landing_n, mean, log_std) * placed).sum()
    landing = landing / placed.sum().clamp(min=1.0)

    with torch.no_grad():
        uniform = torch.rand(logits.shape[:-1], device=logits.device)
        drawn = plan_features(planner.remaining(sample_bins(logits, uniform)), mean, clock.horizon)
    plan = torch.where(teacher.unsqueeze(-1), targets.plan, drawn)

    action = skill.act(current, memory, command, plan)
    context = layout.context_input(streams.obs[acting])
    imitation = F.mse_loss(decoder(action, context), streams.action[acting])

    predicted = []
    state = current[:, 0]
    for j in range(horizon):
        state = skill.predict(state, memory[:, j], action[:, j])
        predicted.append(state)
    target = skill.skill_norm(streams.skill[acting + 1])
    dynamics = F.mse_loss(torch.stack(predicted, dim=1), target)

    return {
        "total": imitation + dynamics_weight * dynamics + plan_weight * (survival + landing),
        "imitation": imitation,
        "dynamics": dynamics,
        "survival": survival,
        "landing": landing,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def calibration(
    logits: torch.Tensor, event_bin: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    """Per part, ``mean_k |mean F_k - mean 1[e <= k]|`` over fully observed samples.

    A sample counts where its outcome is known for every bin: the change came
    within the horizon, or the horizon was seen without one.
    """
    bins = logits.shape[-1]
    k = torch.arange(1, bins + 1, device=logits.device)
    errors = []
    for part in range(logits.shape[-2]):
        e, o = event_bin[..., part].reshape(-1), observed[..., part].reshape(-1)
        known = (e <= bins) | (o >= bins)
        predicted = cdf(logits[..., part, :].reshape(-1, bins)[known]).mean(dim=0)
        happened = (e[known, None] <= k).float().mean(dim=0)
        errors.append((predicted - happened).abs().mean())
    return torch.stack(errors)


@torch.no_grad()
def median_bin(logits: torch.Tensor) -> torch.Tensor:
    """The bin by whose end the change is more likely than not; ``K + 1`` if none."""
    return sample_bins(logits, torch.full(logits.shape[:-1], 0.5, device=logits.device))


def marginal_hazard(event_bin: torch.Tensor, observed: torch.Tensor, bins: int) -> torch.Tensor:
    """Per part and bin, the population hazard: the planner's reference, as logits."""
    j = torch.arange(1, bins + 1, device=event_bin.device)
    flat_e = event_bin.reshape(-1, event_bin.shape[-1])
    flat_o = observed.reshape(-1, observed.shape[-1])
    at_risk = torch.where(
        (flat_e <= bins).unsqueeze(-1), j <= flat_e.unsqueeze(-1), j <= flat_o.unsqueeze(-1)
    ).float()
    events = (flat_e.unsqueeze(-1) == j).float()
    rate = (events.sum(0) / at_risk.sum(0).clamp(min=1.0)).clamp(1e-4, 1 - 1e-4)
    return torch.log(rate) - torch.log1p(-rate)


__all__ = [
    "ContactPlanner",
    "EventClock",
    "PlanTargets",
    "calibration",
    "cdf",
    "fit_landing_normaliser",
    "gaussian_nll",
    "marginal_hazard",
    "median_bin",
    "plan_features",
    "plan_targets",
    "planned_losses",
    "sample_bins",
    "survival_nll",
]
