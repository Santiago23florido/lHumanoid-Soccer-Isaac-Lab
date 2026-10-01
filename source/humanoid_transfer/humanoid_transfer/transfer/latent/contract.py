"""The contract between a skill and a body: the short future of ``z*``.

Everything learned so far crossed from the skill to the body as an action
code ``u``, whose meaning was whatever the G1 decoder and the latent dynamics
learned from walking data. Two measured facts rule that out for a method that
has to carry *any* task to *any* body:

* A body decoder trained on that ``u`` learns to walk, not to carry out
  ``u``: a second task would need a second code and a second decoder for
  every body.
* Imitating a deterministic teacher's actions (``a = π_T(o)``) cannot make a
  student use information about the future: once ``o`` is known nothing about
  the future adds anything about ``a``. The contact plan explained at most
  0.011 of the imitation R², and the closed-loop stride came out 16 % long,
  then 29 % short.

So the interface is fixed by physics instead of learned: **the next ``K``
values of ``z*`` on the body-time grid**. The task side is a generator that
reads only the history of ``ẑ*`` and the dimensionless command and *samples*
such a future; each body has a tracker that is rewarded for reaching it and
never sees the task (``transfer/tasks/tracking``). A new task is a new
generator, trained on its teacher's states; a new body is a new tracker.

Why a sampler, held for ``M`` steps
-----------------------------------
The future of a gait is uncertain given its past -- an MLP predicts ``z*``
eight steps ahead with R² 0.52, and the history adds nothing to the present
(``phase_probe.json``). The mean of two plausible futures is not a plausible
future: averaged over when the next lift-off happens, a foot's load is a ramp
no foot produces. The generator is therefore a conditional VAE: an intent
``ψ`` drawn from a learned prior picks one future.

At run time a plan is drawn from ``p(future | history, c*)``, held for
``M`` steps, and redrawn from the history that actually happened. By the
chain rule that is ancestral sampling of the whole trajectory: with a
calibrated generator and exact tracking, the trajectory of ``z*`` has the
teacher's distribution, so every statistic of it -- stride, period, duty
factor, or whatever a kick has instead -- matches without being named.

Leakage
-------
The generator reads only ``ẑ*`` and ``c*``, as the memory did, so whatever it
asks for is a function of the physical trajectory and cannot carry the body
that produced the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from .memory import SkillMemory, StateEncoder, TimeGrid
from .model import Normaliser, mlp


@dataclass(frozen=True)
class ContractClock:
    """The contract's instants, in body time. Counts are samples of the grid."""

    step: float
    """Body times between grid samples: the G1's control step, ``Δ* = 0.0774``."""

    history: int = 26
    """Samples the generator reads, now included: about 2 ``T``."""

    horizon: int = 13
    """Samples of future in a plan: about 1 ``T``."""

    commit: int = 6
    """Samples a plan is held before it is redrawn: about 0.46 ``T``."""

    view: int = 8
    """Samples of the held plan a tracker sees ahead of now: about 0.62 ``T``."""

    def __post_init__(self) -> None:
        if self.horizon < self.commit - 1 + self.view:
            raise ValueError("a plan must cover its commitment plus the tracker's view")

    @property
    def grid(self) -> TimeGrid:
        return TimeGrid(step=self.step, samples=self.history)


@dataclass(frozen=True)
class ContractDims:
    skill: int = 27
    command: int = 3
    memory: int = 16
    """Width of the history (and future) summaries."""

    intent: int = 8
    """``ψ``: which of the plausible futures (``cvae`` only)."""

    hidden: int = 256
    channels: int = 64
    kind: str = "flow"
    """``flow`` (flow matching) or ``cvae``. See :class:`ContractGenerator`."""

    flow_hidden: int = 512
    flow_steps: int = 16
    """Integration steps from noise to a plan."""

    flow_solver: str = "euler"
    """``euler`` or ``midpoint`` (second order: two velocity evaluations per step)."""


_FREQUENCIES = (1.0, 2.0, 4.0, 8.0)
_TIME_FEATURES = 1 + 2 * len(_FREQUENCIES)


def time_features(t: torch.Tensor) -> torch.Tensor:
    """``(B, 1)`` flow time in ``[0, 1]`` to a few sinusoids, ``(B, 9)``."""
    angles = 2.0 * torch.pi * t * torch.tensor(_FREQUENCIES, device=t.device)
    return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=-1)


class ContractGenerator(nn.Module):
    """``G``: a window of ``ẑ*`` and a command to a sampled future of ``z*``.

    Everything is in the normalised units of :attr:`skill_norm`, and the
    future is a residual on the current state. Two ways to sample it:

    ``flow``
        Flow matching (Lipman et al., 2023): a velocity field carries a
        standard normal draw to a plan in ``flow_steps`` Euler steps. A sample
        is a whole trajectory from the learned conditional, with no averaging
        anywhere -- the reason it replaced the first choice.
    ``cvae``
        An intent ``ψ`` from a learned prior, decoded to the mean future
        given ``ψ``. Measured on the walk, its futures change contact far less
        often than the data (0.28 against 0.45 within one ``T``): the timing
        ``ψ`` does not pin down is averaged, and an averaged lift-off is a
        ramp that never crosses the threshold.

    :attr:`tolerance` is how closely a body can be asked to reach a plan, per
    future sample and dimension: the generator's own conditional spread, with
    the encoder's error as a floor. It is fitted after training
    (:func:`~.train_contract.fit_tolerance`) and read by the tracking reward.
    """

    def __init__(self, dims: ContractDims, clock: ContractClock) -> None:
        super().__init__()
        if dims.kind not in ("flow", "cvae"):
            raise ValueError(f"unknown generator kind {dims.kind!r}")
        self.kind = dims.kind
        self.history = clock.history
        self.horizon = clock.horizon
        self.commit = clock.commit
        self.view = clock.view
        self.skill = dims.skill
        self.intent = dims.intent
        self.flow_steps = dims.flow_steps
        self.flow_solver = dims.flow_solver
        self.register_buffer("grid_step", torch.tensor(clock.step))
        self.register_buffer("tolerance", torch.ones(clock.horizon, dims.skill))
        self.skill_norm = Normaliser(dims.skill)
        self.command_norm = Normaliser(dims.command)
        self.residual_norm = Normaliser(clock.horizon * dims.skill)
        self.memory = SkillMemory(dims.skill, clock.history, dims.memory, dims.channels)
        condition = dims.skill + dims.memory + dims.command
        size = clock.horizon * dims.skill
        if self.kind == "flow":
            self.velocity = mlp(size + _TIME_FEATURES + condition, size, dims.flow_hidden, 3)
        else:
            self.future_memory = SkillMemory(dims.skill, clock.horizon, dims.memory, dims.channels)
            self.prior = mlp(condition, 2 * dims.intent, dims.hidden)
            self.posterior = mlp(condition + dims.memory, 2 * dims.intent, dims.hidden)
            self.decoder = mlp(condition + dims.intent, size, dims.hidden)
            self.log_scale = nn.Parameter(torch.zeros(clock.horizon, dims.skill))

    @property
    def noise_size(self) -> int:
        """Width of the standard normal draw :meth:`sample` takes."""
        return self.horizon * self.skill if self.kind == "flow" else self.intent

    def condition(self, window: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """``(B, history, skill)`` normalised window, dimensionless command -> conditioning."""
        return torch.cat([window[:, -1], self.memory(window), self.command_norm(command)], dim=-1)

    def flow_velocity(
        self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.velocity(torch.cat([x, time_features(t), condition], dim=-1))

    def prior_params(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.prior(condition).chunk(2, dim=-1)
        return mean, log_std.clamp(-6.0, 2.0)

    def posterior_params(
        self, condition: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``q(ψ | condition, future)``; ``residual`` is the future minus the current state."""
        summary = self.future_memory(residual)
        mean, log_std = self.posterior(torch.cat([condition, summary], dim=-1)).chunk(2, dim=-1)
        return mean, log_std.clamp(-6.0, 2.0)

    def decode(self, condition: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """``(B, horizon, skill)``: the future minus the current state (``cvae``)."""
        out = self.decoder(torch.cat([condition, intent], dim=-1))
        return out.reshape(-1, self.horizon, self.skill)

    def sample(
        self, window: torch.Tensor, command: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """One plan per row, ``(B, horizon, skill)`` absolute and normalised.

        ``noise`` is a standard normal ``(B, noise_size)``; it alone decides which
        plausible future is drawn.
        """
        condition = self.condition(window, command)
        if self.kind == "cvae":
            mean, log_std = self.prior_params(condition)
            residual = self.decode(condition, mean + torch.exp(log_std) * noise)
        else:
            x = noise
            step = 1.0 / self.flow_steps
            for i in range(self.flow_steps):
                t = torch.full((x.shape[0], 1), i * step, device=x.device)
                velocity = self.flow_velocity(x, t, condition)
                if self.flow_solver == "midpoint":
                    middle = x + 0.5 * step * velocity
                    velocity = self.flow_velocity(middle, t + 0.5 * step, condition)
                x = x + step * velocity
            residual = self.residual_norm.inverse(x).reshape(-1, self.horizon, self.skill)
        return window[:, -1:] + residual


def gaussian_kl(
    mean_q: torch.Tensor, log_std_q: torch.Tensor, mean_p: torch.Tensor, log_std_p: torch.Tensor
) -> torch.Tensor:
    """``KL(q || p)`` per dimension, for diagonal Gaussians."""
    var_q, var_p = torch.exp(2 * log_std_q), torch.exp(2 * log_std_p)
    return log_std_p - log_std_q + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5


def contract_loss(
    generator: ContractGenerator,
    window: torch.Tensor,
    command: torch.Tensor,
    future: torch.Tensor,
    free_bits: float,
) -> dict[str, torch.Tensor]:
    """The generator's training loss on the true future.

    ``window`` may be noisy (it stands in for an encoder); ``future`` is the
    clean ``z*`` that followed, normalised. For ``flow``, conditional flow
    matching on the straight path from noise to the standardised residual;
    for ``cvae``, the negative evidence lower bound, with ``free_bits`` a
    floor on the KL of each intent dimension so the intent cannot collapse.
    """
    condition = generator.condition(window, command)
    residual = future - window[:, -1:]
    if generator.kind == "flow":
        target = generator.residual_norm(residual.flatten(1))
        source = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)
        moved = (1.0 - t) * source + t * target
        flow = (generator.flow_velocity(moved, t, condition) - (target - source)).pow(2).mean()
        return {"total": flow, "flow": flow}
    mean_q, log_std_q = generator.posterior_params(condition, residual)
    mean_p, log_std_p = generator.prior_params(condition)
    intent = mean_q + torch.exp(log_std_q) * torch.randn_like(mean_q)
    predicted = generator.decode(condition, intent)

    log_scale = generator.log_scale
    nll = (0.5 * ((residual - predicted) * torch.exp(-log_scale)) ** 2 + log_scale).sum((1, 2))
    kl = gaussian_kl(mean_q, log_std_q, mean_p, log_std_p)
    per_dimension = kl.mean(dim=0)
    floored = per_dimension.clamp(min=free_bits).sum()
    return {
        "total": nll.mean() + floored,
        "nll": nll.mean(),
        "kl": per_dimension.sum(),
    }


@dataclass
class Contract:
    """A trained generator, the encoder of the body it was trained on, and that encoder's error."""

    generator: ContractGenerator
    encoder: StateEncoder | None
    sigma: torch.Tensor
    """The encoder's held-out RMS error per dimension, normalised units."""

    meta: dict = field(default_factory=dict)

    def to(self, device: torch.device | str) -> Contract:
        self.generator.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def eval(self) -> Contract:
        self.generator.eval()
        if self.encoder is not None:
            self.encoder.eval()
        return self


def save_contract(
    path: Path,
    contract: Contract,
    dims: ContractDims,
    clock: ContractClock,
    encoder_input: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dims": asdict(dims),
            "clock": asdict(clock),
            "generator": contract.generator.state_dict(),
            "encoder": None if contract.encoder is None else contract.encoder.state_dict(),
            "encoder_input": encoder_input,
            "sigma": contract.sigma.cpu(),
            "meta": contract.meta,
        },
        path,
    )


def load_contract(path: Path | str, device: torch.device | str = "cpu") -> Contract:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    dims = ContractDims(**saved["dims"])
    clock = ContractClock(**saved["clock"])
    generator = ContractGenerator(dims, clock)
    generator.load_state_dict(saved["generator"])
    encoder = None
    if saved["encoder"] is not None:
        encoder = StateEncoder(saved["encoder_input"], dims.skill, dims.hidden)
        encoder.load_state_dict(saved["encoder"])
    contract = Contract(generator, encoder, saved["sigma"], saved.get("meta", {}))
    return contract.to(device).eval()


__all__ = [
    "Contract",
    "ContractClock",
    "ContractDims",
    "ContractGenerator",
    "contract_loss",
    "gaussian_kl",
    "load_contract",
    "save_contract",
]
