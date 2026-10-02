"""Train a contract generator on a teacher's states, and measure what it learned.

The generator sees only the history of ``z*`` -- true, plus noise of the size
of the body encoder's measured error, so it is robust to an encoder without
ever reading one -- and learns the distribution of the ``z*`` that followed.
No action enters: what a task hands to a body is where the body should be,
never how a particular body got there.

The measurement that gates it is about time, because time is what the
action-imitation models lost. For each foot, the instant its contact next
changes is read off every *sampled* future and compared with the instant it
changed in the data. Contact is the measurement, not the method: the
generator is trained on all 27 numbers alike and never told what a contact
is, and the same check applies to whatever events another task has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ...common.gait import dominant_rhythm, gait_signature
from ...common.skill_state import DEFAULT_CONTACT_THRESHOLD, SKILL_LAYOUT
from .contract import (
    ContractClock,
    ContractDims,
    ContractGenerator,
    contract_loss,
    gaussian_kl,
    plan_integrals,
)
from .events import ContactEvents
from .model import Normaliser
from .sequences import Streams
from .train_memory import _optimise

LOAD = SKILL_LAYOUT.slice_of("foot_load")


@dataclass(frozen=True)
class ContractConfig:
    encoder_epochs: int = 150
    epochs: int = 150
    batch_size: int = 4096
    learning_rate: float = 1e-3
    free_bits: float = 0.1
    """Nats per intent dimension below which the KL is not pushed further down."""

    seed: int = 0
    dims: ContractDims = field(default_factory=ContractDims)


def contract_steps(streams: Streams, horizon: int) -> torch.Tensor:
    """Steps whose whole future plan lies in the episode, away from a fall.

    A teacher about to fall is not demonstrating the task; its last second is
    left out of what the generator learns to ask for.
    """
    steps = streams.samples(future=horizon)
    return steps[~streams.falling[steps] & ~streams.falling[steps + horizon]]


def new_generator(
    streams: Streams,
    steps: torch.Tensor,
    config: ContractConfig,
    clock: ContractClock,
    skill_norm: Normaliser | None = None,
) -> ContractGenerator:
    """A generator with its normalisations fitted on ``steps``.

    ``skill_norm`` fixes the units of ``z*`` instead: a second task reuses the
    first one's, so that a body's encoder, which estimates ``z*`` in those
    units, serves every task's generator unchanged.
    """
    torch.manual_seed(config.seed)
    generator = ContractGenerator(config.dims, clock).to(streams.skill.device)
    if skill_norm is None:
        generator.skill_norm.fit(streams.skill[steps])
    else:
        generator.skill_norm.load_state_dict(skill_norm.state_dict())
    generator.command_norm.fit(streams.command[steps])
    generator.fit_warp(streams.skill[steps])
    future = generator.warp(futures(generator, streams, steps))
    now = generator.warp(generator.skill_norm(streams.skill[steps, None]))
    generator.residual_norm.fit((future - now).flatten(1))
    return generator


def windows(
    generator: ContractGenerator,
    streams: Streams,
    steps: torch.Tensor,
    sigma: torch.Tensor | None,
) -> torch.Tensor:
    """``(B, history, skill)`` normalised history windows, with noise ``sigma`` if given.

    The noise is drawn per instant of the trajectory, so a window padded at an
    episode start repeats one noisy state, as a buffer filled at a reset does.
    """
    positions = streams.windows(steps, generator.history)
    window = generator.skill_norm(streams.skill[positions])
    if sigma is None:
        return window
    offsets = positions - (steps[:, None] - generator.history + 1)
    noise = torch.randn_like(window)
    noise = torch.gather(noise, 1, offsets[..., None].expand_as(noise))
    return window + sigma * noise


def futures(generator: ContractGenerator, streams: Streams, steps: torch.Tensor) -> torch.Tensor:
    """``(B, horizon, skill)``: the clean, normalised ``z*`` that followed each step."""
    return generator.skill_norm(streams.skill[streams.futures(steps, generator.horizon)])


def train_generator(
    streams: Streams,
    steps: torch.Tensor,
    generator: ContractGenerator,
    sigma: torch.Tensor,
    config: ContractConfig,
) -> list[dict[str, float]]:
    def loss(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return contract_loss(
            generator,
            windows(generator, streams, batch, sigma),
            streams.command[batch],
            futures(generator, streams, batch),
            config.free_bits,
        )

    generator.train()
    history = _optimise(list(generator.parameters()), steps, loss, config.epochs, config)
    generator.eval()
    return history


@torch.no_grad()
def fit_tolerance(
    generator: ContractGenerator,
    streams: Streams,
    steps: torch.Tensor,
    sigma: torch.Tensor,
    samples: int = 16,
    limit: int = 4096,
    seed: int = 0,
) -> torch.Tensor:
    """How closely a body can be asked to reach a plan, ``(horizon, skill)``, normalised units.

    The spread of the generator's own futures around each other, per future
    sample and dimension, averaged over held-out states: a body that lands on
    another plausible future instead of the drawn one is off by about this
    much. The encoder's error is the floor -- a plan cannot be checked more
    finely than the state can be estimated. The same spread of the plans'
    integrals (:func:`~.contract.plan_integrals`) goes to
    ``generator.integral_tolerance``, floored at a thousandth of a leg length.
    """
    torch.manual_seed(seed)
    chosen = steps[torch.randperm(len(steps), device=steps.device)[:limit]]
    window = windows(generator, streams, chosen, None).repeat_interleave(samples, 0)
    command = streams.command[chosen].repeat_interleave(samples, 0)
    noise = torch.randn(len(window), generator.noise_size, device=window.device)
    plans = generator.sample(window, command, noise)
    plans = plans.reshape(len(chosen), samples, generator.horizon, generator.skill)
    variance = plans.var(dim=1).mean(dim=0)
    if generator.kind == "cvae":
        variance = variance + torch.exp(2 * generator.log_scale)
    tolerance = torch.sqrt(variance + sigma**2)
    generator.tolerance.copy_(tolerance)
    integrals = plan_integrals(generator.skill_norm.inverse(plans), float(generator.grid_step))
    floor = 1e-3
    generator.integral_tolerance.copy_(torch.sqrt(integrals.var(dim=1).mean(dim=0) + floor**2))
    return tolerance


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def change_times(contact: torch.Tensor) -> torch.Tensor:
    """First ``k`` in ``1..K`` where contact differs from the sample before, else ``K + 1``.

    ``contact`` is ``(..., K + 1, P)``: the current contact, then ``K`` future ones.
    """
    horizon = contact.shape[-2] - 1
    changed = contact[..., 1:, :] != contact[..., :-1, :]
    k = torch.arange(1, horizon + 1, device=contact.device)[:, None]
    first = torch.where(changed, k, horizon + 1).amin(dim=-2)
    return first


def event_categories(events: ContactEvents, steps: torch.Tensor, horizon: int) -> torch.Tensor:
    """``(B, P)``: the data's next change, ``1..K``, or ``K + 1`` if later or never."""
    to_change = events.to_change[steps]
    return torch.where(to_change <= horizon, to_change, horizon + 1).long()


def population_scores(categories: torch.Tensor, horizon: int) -> torch.Tensor:
    """``(P, K + 1)`` marginal frequencies of each category, the no-information baseline."""
    counts = torch.stack(
        [
            torch.bincount(categories[:, p] - 1, minlength=horizon + 1)
            for p in range(categories.shape[1])
        ]
    ).float()
    return (counts + 0.5) / (counts.sum(dim=-1, keepdim=True) + 0.5 * (horizon + 1))


@torch.no_grad()
def evaluate_generator(
    generator: ContractGenerator,
    streams: Streams,
    steps: torch.Tensor,
    events: ContactEvents,
    population: torch.Tensor,
    sigma: torch.Tensor | None,
    samples: int = 32,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
    batch: int = 1024,
    seed: int = 0,
) -> dict:
    """Timing calibration, timing information and fit of sampled futures, per foot."""
    torch.manual_seed(seed)
    horizon = generator.horizon
    k = torch.arange(1, horizon + 1, device=steps.device)
    sums = {"forecast": 0.0, "observed": 0.0, "nll": 0.0, "nll_population": 0.0}
    squared = {"mean": torch.zeros(horizon), "persistence": torch.zeros(horizon)}
    first = torch.zeros(horizon, generator.skill)
    second = torch.zeros(horizon, generator.skill)
    kl_sum = torch.zeros(generator.intent)
    count = 0
    for start in range(0, len(steps), batch):
        chunk = steps[start : start + batch]
        size = len(chunk)
        window = windows(generator, streams, chunk, sigma)
        command = streams.command[chunk]
        plans = generator.sample(
            window.repeat_interleave(samples, 0),
            command.repeat_interleave(samples, 0),
            torch.randn(size * samples, generator.noise_size, device=chunk.device),
        ).reshape(size, samples, horizon, -1)

        # A plan is judged from the true present, as the data's own events are: a
        # plan whose first sample contradicts the contact the body really has is
        # asking for a change now, whatever its noisy input said.
        loads = generator.skill_norm.inverse(plans)[..., LOAD] > threshold
        now = streams.skill[chunk][:, LOAD] > threshold
        sequence = torch.cat([now[:, None, None].expand(-1, samples, 1, -1), loads], dim=2)
        forecast = change_times(sequence)  # (B, S, P)
        truth = event_categories(events, chunk, horizon)  # (B, P)

        within = (forecast[..., None] <= k).float().mean(dim=1)  # (B, P, K)
        sums["forecast"] = sums["forecast"] + within.sum(dim=0)
        sums["observed"] = sums["observed"] + (truth[..., None] <= k).float().sum(dim=0)
        hits = (forecast == truth[:, None]).float().sum(dim=1)
        likelihood = (hits + 0.5) / (samples + 0.5 * (horizon + 1))
        sums["nll"] = sums["nll"] - torch.log(likelihood).sum(dim=0)
        sums["nll_population"] = sums["nll_population"] - torch.log(
            population.gather(1, (truth - 1).T).T
        ).sum(dim=0)

        target = futures(generator, streams, chunk)
        mean_plan = plans.mean(dim=1)
        clean_now = generator.skill_norm(streams.skill[chunk])[:, None]
        squared["mean"] += ((mean_plan - target) ** 2).mean(dim=-1).sum(dim=0).cpu()
        squared["persistence"] += ((clean_now - target) ** 2).mean(dim=-1).sum(dim=0).cpu()
        first += target.sum(dim=0).cpu()
        second += (target**2).sum(dim=0).cpu()

        if generator.kind == "cvae":
            condition = generator.condition(window, command)
            mean_q, log_std_q = generator.posterior_params(condition, target - window[:, -1:])
            mean_p, log_std_p = generator.prior_params(condition)
            kl_sum += gaussian_kl(mean_q, log_std_q, mean_p, log_std_p).sum(dim=0).cpu()
        count += size

    forecast = sums["forecast"] / count  # (P, K)
    observed = sums["observed"] / count
    calibration = (forecast - observed).abs().mean(dim=-1)
    variance = (second / count - (first / count) ** 2).mean(dim=-1)
    r2_mean = 1.0 - squared["mean"] / count / variance
    r2_persistence = 1.0 - squared["persistence"] / count / variance
    nll = sums["nll"] / count
    nll_population = sums["nll_population"] / count
    return {
        "steps": count,
        "calibration_error": calibration.tolist(),
        "forecast_change_by_k": forecast.tolist(),
        "observed_change_by_k": observed.tolist(),
        "timing_nll": nll.tolist(),
        "timing_nll_population": nll_population.tolist(),
        "timing_nll_gain": (nll_population - nll).tolist(),
        "r2_sample_mean_by_k": r2_mean.tolist(),
        "r2_persistence_by_k": r2_persistence.tolist(),
        "kl_per_intent_dim": (kl_sum / count).tolist() if generator.kind == "cvae" else None,
    }


def _segments(contact: np.ndarray) -> dict[str, np.ndarray]:
    """Lengths of complete stance and swing segments, ``contact`` ``(T, N, P)``."""
    out: dict[str, list[int]] = {"stance": [], "swing": []}
    for n in range(contact.shape[1]):
        for p in range(contact.shape[2]):
            x = contact[:, n, p].astype(int)
            bounds = np.concatenate([[0], np.flatnonzero(np.diff(x)) + 1, [len(x)]])
            for a, b in zip(bounds[1:-2], bounds[2:-1], strict=True):
                out["stance" if x[a] else "swing"].append(int(b - a))
    return {k: np.array(v) for k, v in out.items()}


def _describe(contact: np.ndarray) -> dict:
    segments = _segments(contact)
    return {
        k: {
            "count": int(len(v)),
            "median": float(np.median(v)) if len(v) else None,
            "mean": float(v.mean()) if len(v) else None,
            "share_one_step": float(np.mean(v == 1)) if len(v) else None,
        }
        for k, v in segments.items()
    }


@torch.no_grad()
def chained_gait(
    generator: ContractGenerator,
    start: torch.Tensor,
    command: torch.Tensor,
    sigma: torch.Tensor,
    steps: int,
    commit: int,
    dt_star: float,
    seed: int = 0,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> dict:
    """The gait of the generator run on its own plans: what a perfect tracker would walk.

    From ``start`` -- ``(N, history, skill)`` normalised windows of real ``z*`` -- a
    plan is drawn, its first ``commit`` samples appended to the history (with the
    training noise ``sigma``, which is what the generator reads in a closed loop),
    and so on for ``steps``. Each plan is calibrated on its own; this asks whether
    their chain keeps the teacher's gait, which no single plan can show.
    """
    torch.manual_seed(seed)
    clean, out = start, []
    while len(out) * commit < steps:
        window = clean + sigma * torch.randn_like(clean)
        noise = torch.randn(len(window), generator.noise_size, device=window.device)
        chunk = generator.sample(window, command, noise)[:, :commit]
        out.append(chunk)
        clean = torch.cat([clean[:, commit:], chunk], dim=1)
    z = generator.skill_norm.inverse(torch.cat(out, dim=1)[:, :steps]).transpose(0, 1)
    z = z.cpu().numpy().astype(float)
    finite = np.isfinite(z).all(axis=(0, 2)) & (np.abs(z).max(axis=(0, 2)) < 10.0)
    z = z[:, finite]
    done = np.zeros(z.shape[:2], dtype=bool)
    signature = gait_signature(z, done, dt_star).as_dict()
    return {
        "environments_stable": int(finite.sum()),
        "environments": int(len(finite)),
        "signature": signature,
        "rhythm_period": float(dominant_rhythm(z, dt_star).period),
        "segments": _describe(z[..., LOAD] > threshold),
        "min_load": float(z[..., LOAD].min()),
    }


def gait_reference(
    z: np.ndarray, dt_star: float, threshold: float = DEFAULT_CONTACT_THRESHOLD
) -> dict:
    """The same summary for a recorded gait ``(T, N, skill)``, to compare with."""
    done = np.zeros(z.shape[:2], dtype=bool)
    return {
        "signature": gait_signature(z, done, dt_star).as_dict(),
        "rhythm_period": float(dominant_rhythm(z, dt_star).period),
        "segments": _describe(z[..., LOAD] > threshold),
        "min_load": float(z[..., LOAD].min()),
    }


__all__ = [
    "ContractConfig",
    "chained_gait",
    "gait_reference",
    "change_times",
    "contract_steps",
    "evaluate_generator",
    "event_categories",
    "fit_tolerance",
    "futures",
    "new_generator",
    "population_scores",
    "train_generator",
    "windows",
]
