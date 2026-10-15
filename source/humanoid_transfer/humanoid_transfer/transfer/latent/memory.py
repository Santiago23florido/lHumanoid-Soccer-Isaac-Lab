"""A memory for the shared skill that cannot let the body back in.

The skill state ``z*`` is instantaneous, and on a gait with a duty factor of
0.92 the present says little about where in the stride the body is. A policy
needs a memory to keep time. The danger is where that memory reads from: a
history of *observations* would carry the body's own joint trajectories into
the shared space, which a second body's encoder could never reproduce. So the
memory is arranged to make that impossible by construction, not discouraged by
a penalty::

    o_t ──E_r──► ẑ*_t ──► per-step buffer ──► grid in T ──► Φ ──► φ_t
                   │                                             │
                   └──────────────► π(ẑ*_t, φ_t, c*) ◄───────────┘
                                         │ u_t
                         D_r(u_t, q_t, q̇_t) ──► a_r        T(ẑ*, φ, u) ──► ẑ*_{t+1}

1. **The memory Φ is shared and reads only the history of ẑ*.** Whatever φ
   holds is a function of the physical trajectory, so it cannot carry anything
   about the body that the trajectory does not (the data-processing
   inequality). A new body reuses Φ frozen and only has to estimate ``z*``
   step by step.
2. **The encoder learns only the anchor.** No imitation gradient reaches it,
   so it has no incentive to hide useful facts about the G1 in its errors --
   the route a body took into the free latent of the memoryless model.
3. **The shared modules train on the true z* plus noise** of the size of the
   encoder's measured error, so the policy never learns to read the G1
   encoder's residuals and is robust to a new encoder's.

What a body itself has to remember -- actuator lag, its own joint motion --
belongs to its decoder, which sees its joints.

**The clock.** A memory every body shares has to agree on what "the last two
strides" means, and seconds are the wrong unit for it: the same gait is slower
on a longer leg. In body time ``T = sqrt(L / g)`` the G1 controls every
0.077 ``T`` and the NAO, at the same 50 Hz, every 0.126 ``T`` -- so a window of
the last 26 control steps would be 1.6 times longer, physically, on the NAO.
The memory reads a fixed grid in ``T`` instead, and each body resamples its
own history onto it (:class:`TimeGrid`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import Normaliser, ObservationLayout, mlp
from .sequences import Streams

_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TimeGrid:
    """``samples`` instants, ``step`` body times apart, ending now. Oldest first."""

    step: float
    samples: int

    @property
    def span(self) -> float:
        """Body times between the oldest sample and now."""
        return (self.samples - 1) * self.step

    def history_steps(self, dt_star: float) -> int:
        """How many of a body's control steps (``dt_star`` body times apart) the grid needs."""
        return math.ceil(self.span / dt_star - _TOLERANCE) + 1

    def interpolation(self, dt_star: float) -> np.ndarray:
        """``(samples, history_steps)``: linear interpolation of a step history onto the grid.

        The history is oldest first and ends at the current step, like the
        grid. When ``dt_star`` equals the grid step this is the identity.
        """
        steps = self.history_steps(dt_star)
        matrix = np.zeros((self.samples, steps))
        for j in range(self.samples):
            position = (steps - 1) - (self.samples - 1 - j) * self.step / dt_star
            nearest = round(position)
            if abs(position - nearest) < _TOLERANCE:
                matrix[j, nearest] = 1.0
                continue
            low = math.floor(position)
            weight = position - low
            matrix[j, low] = 1.0 - weight
            matrix[j, low + 1] = weight
        return matrix


@dataclass(frozen=True)
class MemoryDims:
    skill: int = 27
    """``ẑ*``."""

    memory: int = 16
    """``φ``."""

    action: int = 16
    """``u``: the same width as the memoryless model's, so the two compare."""

    command: int = 3
    plan: int = 0
    """Width of a contact plan the policy also reads (:mod:`.plan`); 0 for none."""

    hidden: int = 256
    channels: int = 64
    """Width of Φ's temporal convolutions."""


class StateEncoder(nn.Module):
    """``E_r``: one body's observation to its estimate of ``z*``, in shared normalised units."""

    def __init__(self, observation: int, skill: int, hidden: int) -> None:
        super().__init__()
        self.obs_norm = Normaliser(observation)
        self.net = mlp(observation, skill, hidden)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(self.obs_norm(observation))


class SkillMemory(nn.Module):
    """``Φ``: a window of ``ẑ*`` on the grid, ``(B, samples, skill)``, to ``φ``.

    Two strided temporal convolutions and a linear read-out: the same filter
    at every instant of the window, so the memory looks for *patterns in
    time* -- a landing, a weight shift -- rather than learning each lag apart.
    """

    def __init__(self, skill: int, samples: int, memory: int, channels: int) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(skill, channels, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=5, stride=2),
            nn.ELU(),
        )
        length = samples
        for _ in range(2):
            length = (length - 5) // 2 + 1
        if length < 1:
            raise ValueError(f"a grid of {samples} samples is too short for the memory")
        self.readout = nn.Linear(channels * length, memory)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        features = self.convolutions(window.transpose(1, 2))
        return self.readout(features.flatten(1))


class MemorySkill(nn.Module):
    """The shared modules: memory ``Φ``, skill policy ``π`` and latent dynamics ``T``.

    All of them read ``ẑ*`` in the normalised units of the shared space; the
    normalisation and the grid are part of the skill, fitted once and frozen
    with it.
    """

    __jit_unused_properties__ = ["grid"]

    def __init__(self, dims: MemoryDims, grid: TimeGrid) -> None:
        super().__init__()
        self.samples = grid.samples
        self.register_buffer("grid_step", torch.tensor(grid.step))
        self.skill_norm = Normaliser(dims.skill)
        self.command_norm = Normaliser(dims.command)
        self.memory = SkillMemory(dims.skill, grid.samples, dims.memory, dims.channels)
        self.policy = mlp(
            dims.skill + dims.memory + dims.command + dims.plan, dims.action, dims.hidden
        )
        self.dynamics = mlp(dims.skill + dims.memory + dims.action, dims.skill, dims.hidden)

    @property
    def grid(self) -> TimeGrid:
        return TimeGrid(step=float(self.grid_step), samples=self.samples)

    def remember(self, window: torch.Tensor) -> torch.Tensor:
        """``φ = Φ(window)``; any leading dimensions, the last two ``(samples, skill)``."""
        shape = list(window.shape[:-2]) + [-1]
        flat = window.reshape(-1, window.shape[-2], window.shape[-1])
        return self.memory(flat).reshape(shape)

    def act(
        self,
        skill_n: torch.Tensor,
        memory: torch.Tensor,
        command: torch.Tensor,
        plan: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``u = π(ẑ*, φ, c*[, plan])``; ``command`` dimensionless, not normalised."""
        inputs = [skill_n, memory, self.command_norm(command)]
        if plan is not None:
            inputs.append(plan)
        return self.policy(torch.cat(inputs, dim=-1))

    def predict(
        self, skill_n: torch.Tensor, memory: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """``ẑ*' = T(ẑ*, φ, u)``, as a residual on the current state."""
        return skill_n + self.dynamics(torch.cat([skill_n, memory, action], dim=-1))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def anchor_loss(
    encoder: StateEncoder,
    skill: MemorySkill,
    layout: ObservationLayout,
    streams: Streams,
    steps: torch.Tensor,
) -> torch.Tensor:
    """The encoder's only objective: estimate ``z*``."""
    estimate = encoder(layout.encoder_input(streams.obs[steps]))
    return F.mse_loss(estimate, skill.skill_norm(streams.skill[steps]))


def skill_losses(
    skill: MemorySkill,
    decoder: nn.Module,
    layout: ObservationLayout,
    streams: Streams,
    steps: torch.Tensor,
    sigma: torch.Tensor,
    horizon: int,
    dynamics_weight: float,
) -> dict[str, torch.Tensor]:
    """Imitation and ``horizon``-step prediction, on the true ``z*`` plus noise ``sigma``.

    For each sample ``t`` the shared modules act at ``t .. t + horizon - 1``,
    each from its own history window; the decoder imitates the teacher at all
    of them, and ``T`` is rolled out from the noisy ``ẑ*_t`` through the
    actions taken, against the clean ``z*`` that followed. Nothing here sees an
    encoder: the noise stands in for one.

    The noise is drawn per instant of the underlying trajectory, so the padded
    start of a window repeats one noisy state, as a buffer filled at a reset
    repeats one estimate.
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
    action = skill.act(current, memory, streams.command[acting])
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
        "total": imitation + dynamics_weight * dynamics,
        "imitation": imitation,
        "dynamics": dynamics,
    }


# ---------------------------------------------------------------------------
# The policy that runs
# ---------------------------------------------------------------------------


class MemoryPolicy(nn.Module):
    """One body's full policy through the shared space, with its history buffer.

    ``observation -> E_r -> buffer -> grid -> Φ -> π -> D_r -> action``. The
    buffer holds the last ``history_steps`` estimates of ``ẑ*`` per
    environment. After :meth:`reset` flags an environment, its next call
    fills the buffer with the current estimate, so no history crosses an
    episode boundary -- the same padding the training windows use.

    With ``memory`` given, ``φ`` is held at that value: the bypass test for
    the memory.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        skill: MemorySkill,
        decoder: nn.Module,
        layout: ObservationLayout,
        velocity_unit: float,
        time_unit: float,
        dt_star: float,
        memory: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.skill = skill
        self.decoder = decoder
        self.command_start, self.command_stop = layout.command
        self.context_start, self.context_stop = layout.context
        self.register_buffer(
            "command_scale", torch.tensor([1.0 / velocity_unit, 1.0 / velocity_unit, time_unit])
        )
        matrix = skill.grid.interpolation(dt_star)
        self.history_steps = matrix.shape[1]
        self.register_buffer("interpolation", torch.as_tensor(matrix, dtype=torch.float32))
        width = skill.skill_norm.mean.shape[0]
        self.register_buffer("history", torch.zeros(0, self.history_steps, width), persistent=False)
        self.register_buffer("fresh", torch.zeros(0, dtype=torch.bool), persistent=False)
        self.frozen_memory = memory is not None
        fixed = memory if memory is not None else torch.zeros(skill.memory.readout.out_features)
        self.register_buffer("fixed_memory", fixed.detach().reshape(-1).clone())

    def observe(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(ẑ*, φ, c*)`` for this step, after pushing ``ẑ*`` into the history."""
        state = torch.cat(
            [observation[:, : self.command_start], observation[:, self.command_stop :]], dim=-1
        )
        skill_n = self.encoder(state)
        batch = skill_n.shape[0]
        if self.history.shape[0] != batch:
            self.history = skill_n.unsqueeze(1).repeat(1, self.history_steps, 1)
            self.fresh = torch.zeros(batch, dtype=torch.bool, device=skill_n.device)
        else:
            refill = self.fresh.reshape(-1, 1, 1)
            self.history = torch.where(
                refill, skill_n.unsqueeze(1).expand_as(self.history), self.history
            )
            self.history = torch.cat([self.history[:, 1:], skill_n.unsqueeze(1)], dim=1)
            self.fresh = torch.zeros_like(self.fresh)

        if self.frozen_memory:
            memory = self.fixed_memory.unsqueeze(0).expand(batch, -1)
        else:
            memory = self.skill.remember(torch.matmul(self.interpolation, self.history))
        command = observation[:, self.command_start : self.command_stop] * self.command_scale
        return skill_n, memory, command

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        skill_n, memory, command = self.observe(observation)
        action = self.skill.act(skill_n, memory, command)
        return self.decoder(action, observation[:, self.context_start : self.context_stop])

    @torch.jit.export
    def reset(self, done: torch.Tensor) -> None:
        """Mark environments whose episode just ended; their history restarts on the next call."""
        if self.fresh.shape[0] == done.shape[0]:
            self.fresh = self.fresh | done.to(torch.bool)


__all__ = [
    "MemoryDims",
    "MemoryPolicy",
    "MemorySkill",
    "SkillMemory",
    "StateEncoder",
    "TimeGrid",
    "anchor_loss",
    "skill_losses",
]
