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
from torch.func import functional_call

from .events import ContactEvents
from .memory import MemoryDims, MemoryPolicy, MemorySkill, StateEncoder
from .model import Normaliser, ObservationLayout, mlp
from .sequences import Streams

_TOLERANCE = 1e-6


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
        log_std = out[..., self.bins + 2 :].clamp(-5.0, 2.0)
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
# Holding the policy to its plan
# ---------------------------------------------------------------------------


def _frozen(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in module.named_parameters()}


def follow_plan(
    skill: MemorySkill,
    window: torch.Tensor,
    command: torch.Tensor,
    remaining: torch.Tensor,
    landing_n: torch.Tensor,
    steps: int,
    clock: EventClock,
    dt_star: float,
    load: slice,
    threshold: float,
    temperature: float,
    fixed_plan: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll the latent dynamics forward under the policy given a plan; ``(B, steps, P)``
    probabilities that each contact is loaded after each step.

    The rollout is closed inside the shared space: the state advances by
    ``T``, the memory reads a window that the predicted states extend, and
    the plan counts down by ``dt*`` -- once a planned change is due, the
    policy is told no further change is planned, as it would be until a
    redraw. Nothing of the recorded future enters, so the plan can differ
    from what the teacher did.

    ``Φ`` and ``T`` are used with their parameters detached: through this
    rollout only the policy can learn, so it cannot make the model of the
    world agree with it.

    With ``fixed_plan``, the policy reads that instead: what it does when the
    plan is withheld.
    """
    memory_params = _frozen(skill.memory)
    dynamics_params = _frozen(skill.dynamics)
    state = window[:, -1]
    loaded = []
    for j in range(steps):
        phi = functional_call(skill.memory, memory_params, (window,))
        left = remaining - j * dt_star
        left = torch.where(left < 0.5 * dt_star, torch.full_like(left, clock.horizon), left)
        plan = plan_features(left, landing_n, clock.horizon)
        if fixed_plan is not None:
            plan = fixed_plan.expand_as(plan)
        action = skill.act(state, phi, command, plan)
        delta = functional_call(
            skill.dynamics, dynamics_params, (torch.cat([state, phi, action], dim=-1),)
        )
        state = state + delta
        window = torch.cat([window[:, 1:], state.unsqueeze(1)], dim=1)
        share = skill.skill_norm.inverse(state)[:, load]
        loaded.append(torch.sigmoid((share - threshold) / temperature))
    return torch.stack(loaded, dim=1)


def planned_contacts(
    contact: torch.Tensor, remaining: torch.Tensor, steps: int, dt_star: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """What a plan says about the contacts after each step, ``(B, steps, P)`` each.

    The contact stays as it is now until the planned change is due and is
    flipped from then on. The plan speaks only up to that change, inclusive:
    whatever follows -- a landing three steps after a planned lift-off -- is
    the next plan's business, so those steps are not held to anything.
    """
    elapsed = torch.arange(1, steps + 1, device=remaining.device) * dt_star
    elapsed = elapsed[None, :, None]
    due = remaining.unsqueeze(1)
    wanted = contact.unsqueeze(1) ^ (elapsed >= due - 1e-6)
    spoken = elapsed < due + 0.5 * dt_star
    return wanted, spoken


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
    consistency_weight: float = 0.0,
    load: slice | None = None,
    threshold: float = 0.05,
    temperature: float = 0.05,
) -> dict[str, torch.Tensor]:
    """:func:`~.memory.skill_losses` with a contact plan in the loop.

    ``hindsight`` is per stream step: True where the teacher drove the
    rollout, so the schedule that followed is the teacher's. Only there does
    the planner learn, and only there does the policy read the schedule that
    followed; elsewhere it reads one sampled from the planner.

    With ``consistency_weight``, the policy is also held to a plan drawn from
    the planner -- on every rollout, whoever drove it -- through
    :func:`follow_plan`: the contacts the latent dynamics predict under its
    actions should change when the plan says. Imitation alone has almost no
    use for the plan; this is what makes the plan an instruction.
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

    total = imitation + dynamics_weight * dynamics + plan_weight * (survival + landing)
    losses = {
        "imitation": imitation,
        "dynamics": dynamics,
        "survival": survival,
        "landing": landing,
    }
    if consistency_weight > 0.0:
        assert load is not None
        with torch.no_grad():
            uniform = torch.rand(logits.shape[0], logits.shape[2], device=logits.device)
            remaining = planner.remaining(sample_bins(logits[:, 0], uniform))
        loaded = follow_plan(
            skill, windows[:, 0], command[:, 0], remaining, mean[:, 0].detach(), horizon,
            clock, dt_star, load, threshold, temperature,
        )  # fmt: skip
        wanted, spoken = planned_contacts(events.contact[steps], remaining, horizon, dt_star)
        weight = spoken.float()
        mismatch = F.binary_cross_entropy(loaded, wanted.float(), reduction="none")
        consistency = (mismatch * weight).sum() / weight.sum().clamp(min=1.0)
        total = total + consistency_weight * consistency
        losses["consistency"] = consistency
    return {"total": total, **losses}


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


# ---------------------------------------------------------------------------
# The policy that runs
# ---------------------------------------------------------------------------


class PlannedPolicy(MemoryPolicy):
    """:class:`~.memory.MemoryPolicy` that draws a contact schedule and holds it.

    Per environment and part it keeps the body time left to the planned
    change and the planned landing. Each step the time counts down by
    ``dt*``; the plan is redrawn when the part's contact (read off ``ẑ*``)
    changes, when the time falls due, while no change is planned within the
    horizon, and after a reset. The landing is fixed when the time is drawn.

    ``stochastic=False`` draws the median instead of a sample. With ``plan``
    given, the policy reads that fixed plan instead: the bypass test for the
    schedule.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        skill: MemorySkill,
        planner: ContactPlanner,
        decoder: nn.Module,
        layout: ObservationLayout,
        velocity_unit: float,
        time_unit: float,
        dt_star: float,
        threshold: float,
        load: slice,
        memory: torch.Tensor | None = None,
        plan: torch.Tensor | None = None,
        stochastic: bool = True,
    ) -> None:
        super().__init__(
            encoder, skill, decoder, layout, velocity_unit, time_unit, dt_star, memory=memory
        )
        self.planner = planner
        self.dt_star = dt_star
        self.threshold = threshold
        self.load_start, self.load_stop = load.start, load.stop
        self.stochastic = stochastic
        parts = planner.parts
        self.register_buffer("remaining", torch.zeros(0, parts), persistent=False)
        self.register_buffer("landing", torch.zeros(0, parts, 2), persistent=False)
        self.register_buffer("contact", torch.zeros(0, parts, dtype=torch.bool), persistent=False)
        self.frozen_plan = plan is not None
        fixed = plan if plan is not None else torch.zeros(planner.width)
        self.register_buffer("fixed_plan", fixed.detach().reshape(-1).clone())

    def schedule(
        self,
        skill_n: torch.Tensor,
        memory: torch.Tensor,
        command: torch.Tensor,
        fresh: torch.Tensor,
    ) -> torch.Tensor:
        """Advance every environment's plan by one step and return it as the policy reads it."""
        batch = skill_n.shape[0]
        load = self.skill.skill_norm.inverse(skill_n)[:, self.load_start : self.load_stop]
        contact = load > self.threshold
        if self.remaining.shape[0] != batch:
            self.remaining = torch.zeros(batch, self.planner.parts, device=skill_n.device)
            self.landing = torch.zeros(batch, self.planner.parts, 2, device=skill_n.device)
            self.contact = contact
            fresh = torch.ones(batch, dtype=torch.bool, device=skill_n.device)

        within = self.remaining <= self.planner.horizon
        remaining = torch.where(within, self.remaining - self.dt_star, self.remaining)
        redraw = (
            fresh.unsqueeze(-1)
            | (contact != self.contact)
            | (remaining < 0.5 * self.dt_star)
            | ~within
        )
        logits, mean, _ = self.planner(skill_n, memory, self.skill.command_norm(command))
        if self.stochastic:
            uniform = torch.rand(logits.shape[:-1], device=logits.device)
        else:
            uniform = torch.full(logits.shape[:-1], 0.5, device=logits.device)
        drawn = self.planner.remaining(sample_bins(logits, uniform))

        self.remaining = torch.where(redraw, drawn, remaining)
        self.landing = torch.where(redraw.unsqueeze(-1), mean, self.landing)
        self.contact = contact
        return plan_features(self.remaining, self.landing, self.planner.horizon)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        batch = observation.shape[0]
        if self.fresh.shape[0] == batch:
            fresh = self.fresh.clone()
        else:
            fresh = torch.ones(batch, dtype=torch.bool, device=observation.device)
        skill_n, memory, command = self.observe(observation)
        plan = self.schedule(skill_n, memory, command, fresh)
        if self.frozen_plan:
            plan = self.fixed_plan.unsqueeze(0).expand(batch, -1)
        action = self.skill.act(skill_n, memory, command, plan)
        return self.decoder(action, observation[:, self.context_start : self.context_stop])


__all__ = [
    "ContactPlanner",
    "EventClock",
    "PlanTargets",
    "PlannedPolicy",
    "calibration",
    "cdf",
    "follow_plan",
    "fit_landing_normaliser",
    "gaussian_nll",
    "marginal_hazard",
    "median_bin",
    "plan_features",
    "plan_targets",
    "planned_contacts",
    "planned_losses",
    "sample_bins",
    "survival_nll",
]
