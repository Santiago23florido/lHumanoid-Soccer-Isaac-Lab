"""A latent space shared by every body, with an encoder and a decoder per body.

The architecture is the modular one cross-embodiment transfer uses (Devin et
al., 2017; Wang et al., 2024): per-body modules translate in and out of a
latent space, and the skill lives in modules that only ever see that space.
What is different here is how the space gets its meaning.

::

    o_r ──E_r──► (ẑ*, ζ) ──┐
                           ├─► π(ẑ*, ζ, c*) ──► u ──D_r(u, κ_r)──► a_r
    c* (dimensionless cmd) ┘                     │
                                   T(ẑ*, ζ, u) ──► (ẑ*', ζ')

``ẑ*``
    The **anchored** part: trained to equal the analytic skill state ``z*`` of
    :mod:`~humanoid_transfer.common.skill_state`. Because ``z*`` is defined in
    body units, it means the same thing on every humanoid, so a second body's
    encoder can be pinned to the same coordinates using only its own
    simulation -- no paired data, no adversary.
``ζ``
    A small **free** part, for whatever ``z*`` misses (the gait has no clock
    in ``z*`` other than contacts), kept small by a KL bottleneck so it cannot
    become a back door for the teacher's joint state.
``u``
    The latent action. ``T`` predicts the next ``ẑ*`` from it, which gives
    ``u`` a physical meaning: *the u that produces this change of the
    centroidal state*. A new body's decoder is held to that meaning by the
    same loss.
``κ_r``
    The body's **private** context: its joint positions and velocities.
    Deliberately nothing about the floating base -- no tilt, no
    velocity, no command -- so balance and intent can only reach the decoder
    through ``u``. Without that restriction the decoder could reproduce the
    teacher from the observation alone and ignore the shared modules.

Shared modules are :class:`SharedSkill`; per-body ones are :class:`BodyEncoder`
and :class:`BodyDecoder`. The normalisation of ``z*`` and ``c*`` lives inside
:class:`SharedSkill`: it is a fixed affine map of a space whose meaning is
body-independent, so it is part of the shared space, not of the G1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentDims:
    skill: int = 27
    """``ẑ*``: the skill state's size."""

    free: int = 8
    """``ζ``."""

    action: int = 16
    """``u``."""

    command: int = 3
    """``c*``: forward and lateral speed in ``V``, yaw rate in ``1/T``."""

    hidden: int = 256


def mlp(inputs: int, outputs: int, hidden: int, layers: int = 2) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = inputs
    for _ in range(layers):
        modules += [nn.Linear(width, hidden), nn.ELU()]
        width = hidden
    modules.append(nn.Linear(width, outputs))
    return nn.Sequential(*modules)


class Normaliser(nn.Module):
    """A fixed affine map ``(x - mean) / std``, stored with the module that uses it."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))

    def fit(self, data: torch.Tensor, floor: float = 1e-3) -> None:
        flat = data.reshape(-1, data.shape[-1])
        self.mean.copy_(flat.mean(dim=0))
        self.std.copy_(flat.std(dim=0).clamp_min(floor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class SharedSkill(nn.Module):
    """The modules every body shares: the skill policy ``π`` and the latent dynamics ``T``.

    Both take ``ẑ*`` in normalised units of the *shared* space; the
    normalisation is fitted once and then frozen with the skill.
    """

    def __init__(self, dims: LatentDims) -> None:
        super().__init__()
        self.dims = dims
        self.skill_norm = Normaliser(dims.skill)
        self.command_norm = Normaliser(dims.command)
        self.policy = mlp(dims.skill + dims.free + dims.command, dims.action, dims.hidden)
        self.dynamics = mlp(
            dims.skill + dims.free + dims.action, dims.skill + dims.free, dims.hidden
        )

    def act(self, skill_n: torch.Tensor, free: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """``u = π(ẑ*, ζ, c*)``. ``command`` is the dimensionless command, not normalised."""
        return self.policy(torch.cat([skill_n, free, self.command_norm(command)], dim=-1))

    def predict(
        self, skill_n: torch.Tensor, free: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(ẑ*', ζ') = T(ẑ*, ζ, u)``, as a residual on the current latent."""
        delta = self.dynamics(torch.cat([skill_n, free, action], dim=-1))
        return skill_n + delta[..., : self.dims.skill], free + delta[..., self.dims.skill :]


class BodyEncoder(nn.Module):
    """``E_r``: one body's observation to ``(ẑ*, ζ)``."""

    def __init__(self, observation: int, dims: LatentDims) -> None:
        super().__init__()
        self.dims = dims
        self.obs_norm = Normaliser(observation)
        self.net = mlp(observation, dims.skill + 2 * dims.free, dims.hidden)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``ẑ*`` (normalised shared units), and the mean and log-variance of ``ζ``."""
        out = self.net(self.obs_norm(observation))
        skill, free = out[..., : self.dims.skill], out[..., self.dims.skill :]
        mean, log_var = free[..., : self.dims.free], free[..., self.dims.free :]
        return skill, mean, log_var.clamp(-8.0, 4.0)


class BodyDecoder(nn.Module):
    """``D_r``: latent action and private kinematic context to this body's joint targets."""

    def __init__(self, context: int, actions: int, dims: LatentDims) -> None:
        super().__init__()
        self.context_norm = Normaliser(context)
        self.net = mlp(dims.action + context, actions, dims.hidden)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([action, self.context_norm(context)], dim=-1))


@dataclass(frozen=True)
class ObservationLayout:
    """Where a body's policy observation keeps the pieces the latent model splits apart."""

    size: int
    command: tuple[int, int]
    """``[start, stop)`` of the velocity command (vx, vy, wz) in SI units."""

    context: tuple[int, int]
    """``[start, stop)`` of the private kinematic context ``κ``."""

    @property
    def encoder_size(self) -> int:
        return self.size - (self.command[1] - self.command[0])

    @property
    def context_size(self) -> int:
        return self.context[1] - self.context[0]

    def encoder_input(self, observation: torch.Tensor) -> torch.Tensor:
        """The observation without the command: the encoder describes state, not intent.

        Anything past the layout's own width is dropped: a height scan another task's
        teacher reads (Isaac Lab's rough-terrain G1 appends 187 values) is not the
        body's proprioception.
        """
        observation = observation[..., : self.size]
        return torch.cat(
            [observation[..., : self.command[0]], observation[..., self.command[1] :]], dim=-1
        )

    def context_input(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[..., self.context[0] : self.context[1]]


G1_OBSERVATION = ObservationLayout(size=123, command=(9, 12), context=(12, 86))
"""The teacher's policy group (g1/policy/teacher_env.yaml): base linear velocity,
base angular velocity, projected gravity (0:9), command (9:12), joint positions
(12:49), joint velocities (49:86), last action (86:123).

The context is the joint positions and velocities only. The last action is
left out on purpose: a policy's consecutive actions are nearly equal, so a
decoder that sees the previous one can imitate the teacher by copying it --
the "copycat" shortcut of behaviour cloning (Wen et al., 2020) -- and the
shared latent action would carry nothing. The encoder still sees it."""


class LatentPolicy(nn.Module):
    """One body's full policy through the shared space. What gets exported and run.

    ``observation -> E_r -> π -> D_r -> action``, with ``ζ`` at its mean and the
    command converted to body units on the way in.
    """

    def __init__(
        self,
        encoder: BodyEncoder,
        skill: SharedSkill,
        decoder: BodyDecoder,
        layout: ObservationLayout,
        velocity_unit: float,
        time_unit: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.skill = skill
        self.decoder = decoder
        self.command_slice = layout.command
        self.context_slice = layout.context
        self.register_buffer(
            "command_scale", torch.tensor([1.0 / velocity_unit, 1.0 / velocity_unit, time_unit])
        )

    def dimensionless_command(self, observation: torch.Tensor) -> torch.Tensor:
        """``c* = (vx / V, vy / V, wz T)``: Froude matching, done by the units."""
        start, stop = self.command_slice
        return observation[..., start:stop] * self.command_scale

    def latent_action(self, observation: torch.Tensor) -> torch.Tensor:
        start, stop = self.command_slice
        state = torch.cat([observation[..., :start], observation[..., stop:]], dim=-1)
        skill_n, free_mean, _ = self.encoder(state)
        return self.skill.act(skill_n, free_mean, self.dimensionless_command(observation))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        start, stop = self.context_slice
        return self.decoder(self.latent_action(observation), observation[..., start:stop])


class FrozenActionPolicy(nn.Module):
    """The decoder with the latent action held at a constant: the bypass test, closed-loop.

    If a body still walks like this, its decoder has learned the gait from its
    own joints and the shared skill is decoration. Exported next to every
    policy so the test runs in the same harness.
    """

    def __init__(self, decoder: BodyDecoder, layout: ObservationLayout, action: torch.Tensor):
        super().__init__()
        self.decoder = decoder
        self.context_slice = layout.context
        self.register_buffer("action", action.detach().reshape(1, -1).clone())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        start, stop = self.context_slice
        context = observation[..., start:stop]
        return self.decoder(self.action.expand(context.shape[0], -1), context)


__all__ = [
    "BodyDecoder",
    "FrozenActionPolicy",
    "BodyEncoder",
    "G1_OBSERVATION",
    "LatentDims",
    "LatentPolicy",
    "Normaliser",
    "ObservationLayout",
    "SharedSkill",
    "mlp",
]
