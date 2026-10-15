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

import copy
import math
from collections.abc import Callable
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
    keep = ~streams.falling[steps] & ~streams.falling[steps + horizon]
    if streams.target is not None:
        keep &= streams.target[steps]
    return steps[keep]


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


@dataclass(frozen=True)
class ChainConfig:
    """Training on the generator's own chains (Data as Demonstrator)."""

    rounds: int = 0
    """At most this many rounds of chains, each aggregated with the ones before; 0 trains on
    the data alone. With a measure, rounds stop as soon as one fails to improve it beyond
    its own standard error, and the best generator is kept."""

    plans: int = 6
    """Plans per chain, each held for the commitment, as deployed."""

    starts: int = 8192
    """Chains drawn per round, from the data's own windows."""

    epochs: int = 300
    """Epochs of training after each round, on the data and every chain so far."""


def chain_starts(
    streams: Streams, steps: torch.Tensor, plans: int, commit: int, horizon: int
) -> torch.Tensor:
    """Training steps from which a chain keeps the data's real future at every plan boundary."""
    span = plans * commit + horizon
    # Inside the episode first: only then can the chain's steps be looked up at all.
    steps = steps[steps + span <= streams.end[steps]]
    boundaries = steps[:, None] + commit * torch.arange(1, plans + 1, device=steps.device)
    keep = ~streams.falling[boundaries + horizon].any(dim=1)
    if streams.target is not None:
        keep &= streams.target[boundaries].all(dim=1)
    return steps[keep]


@torch.no_grad()
def chain_from(
    generator: ContractGenerator,
    streams: Streams,
    starts: torch.Tensor,
    sigma: torch.Tensor,
    plans: int,
    commit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The generator's own chains from the data's windows.

    Returns, for every plan boundary of every chain, the clean history the
    generator wrote up to it (normalised) and the data's step at that instant,
    whose real future is where the teacher went from there.
    """
    clean = windows(generator, streams, starts, None)
    histories, steps = [], []
    for k in range(plans):
        noisy = clean + sigma * torch.randn_like(clean)
        noise = torch.randn(len(starts), generator.noise_size, device=clean.device)
        plan = generator.sample(noisy, streams.command[starts + k * commit], noise)[:, :commit]
        clean = torch.cat([clean[:, commit:], plan], dim=1)
        histories.append(clean)
        steps.append(starts + (k + 1) * commit)
    return torch.cat(histories), torch.cat(steps)


def _isolated(function: Callable):
    """Call a measurement without letting its own seeding reset the training's random state."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        return function()


@torch.no_grad()
def data_loss(
    generator: ContractGenerator,
    streams: Streams,
    steps: torch.Tensor,
    sigma: torch.Tensor,
    config: ContractConfig,
    batch: int = 1024,
) -> tuple[float, float]:
    """The generator's loss on the data at ``steps``, fixed draws: mean and standard error."""
    device = steps.device
    devices = [device.index or 0] if device.type == "cuda" else []
    values = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        for chunk in steps.split(batch):
            window = windows(generator, streams, chunk, sigma)
            future = futures(generator, streams, chunk)
            losses = contract_loss(
                generator, window, streams.command[chunk], future, config.free_bits
            )
            values.append(float(losses["total"]))
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def train_on_chains(
    streams: Streams,
    steps: torch.Tensor,
    generator: ContractGenerator,
    sigma: torch.Tensor,
    config: ContractConfig,
    chain: ChainConfig,
    measure: Callable[[], dict] | None = None,
    held_out: torch.Tensor | None = None,
    constraint: Callable[[], float] | None = None,
    limit: float = 0.0,
    check_every: int = 25,
) -> list[dict[str, float]]:
    """Teach the generator to come back from its own drift, as a constrained problem.

    With ``constraint`` (a measured quantity that must stay at or below
    ``limit``, checked every ``check_every`` epochs), the multiplier follows
    that measurement instead of the data loss's bound: ``L_data`` stays the
    restoring term, weighted up only when the measured constraint is violated.
    The plans' timing calibration against its gate is the one used: the bound
    on the raw loss at its own noise left no room to improve the chain at all.

    A generator trained on one plan of data at a time is used on a chain of its
    own plans, and its small errors compound: the rough-terrain policy's
    generator stepped 17-20 % fast on its own chain while each plan was
    calibrated. From the data's own windows the generator writes chains of plans
    as deployed, and each history it wrote is paired with the *real* future the
    teacher went on to from that instant (Data as Demonstrator; Venkatraman,
    Hebert and Bagnell, 2015). Rounds aggregate, as DAgger does.

    How much those chains weigh is not fixed. The generator minimises its loss
    on its own chains subject to not becoming measurably worse on the data:

        min  L_chain(θ)   s.t.  L_data(θ) <= L_data(θ₀) + 2 SE

    the bound being its held-out data loss before this training plus twice that
    loss's own standard error -- what the measurement cannot tell apart. The
    Lagrangian ``L_chain + μ L_data`` is minimised while ``μ`` rises by dual
    ascent when the held-out loss exceeds the bound and falls toward zero when
    it does not. A round is kept only if it improves ``measure`` by more than
    twice the best one's standard error; the best generator is returned.
    """
    commit = generator.commit
    device = steps.device
    pool = chain_starts(streams, steps, chain.plans, commit, generator.horizon)
    reference = held_out if held_out is not None else steps
    generator.eval()
    base, base_se = data_loss(generator, streams, reference, sigma, config)
    bound = base + 2.0 * base_se
    multiplier = 1.0
    best = _isolated(measure) if measure is not None else None
    best_state = copy.deepcopy(generator.state_dict())
    measured = _isolated(constraint) if constraint is not None else None
    order = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    histories: list[torch.Tensor] = []
    chained: list[torch.Tensor] = []
    log: list[dict] = [{"round": -1, "data_loss": base, "data_loss_se": base_se, "bound": bound}]
    for round_index in range(chain.rounds):
        pick = pool[torch.randperm(len(pool), generator=order)[: chain.starts].to(device)]
        generator.eval()
        history, at = chain_from(generator, streams, pick, sigma, chain.plans, commit)
        histories.append(history)
        chained.append(at)
        own_history, own_steps = torch.cat(histories), torch.cat(chained)

        optimiser = torch.optim.Adam(generator.parameters(), lr=config.learning_rate)
        per_epoch = max(1, math.ceil(len(own_steps) / config.batch_size))
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=chain.epochs * per_epoch
        )
        for epoch in range(chain.epochs):
            generator.train()
            shuffled = torch.randperm(len(own_steps), generator=order).to(device)
            for start in range(0, len(shuffled), config.batch_size):
                own = shuffled[start : start + config.batch_size]
                real = steps[torch.randint(len(steps), (len(own),), generator=order).to(device)]
                own_window = own_history[own]
                chain_term = contract_loss(
                    generator,
                    own_window + sigma * torch.randn_like(own_window),
                    streams.command[own_steps[own]],
                    futures(generator, streams, own_steps[own]),
                    config.free_bits,
                )["total"]
                data_term = contract_loss(
                    generator,
                    windows(generator, streams, real, sigma),
                    streams.command[real],
                    futures(generator, streams, real),
                    config.free_bits,
                )["total"]
                optimiser.zero_grad(set_to_none=True)
                (chain_term + multiplier * data_term).backward()
                optimiser.step()
                schedule.step()
            generator.eval()
            if constraint is None:
                held, _ = data_loss(generator, streams, reference, sigma, config)
                # Dual ascent on the bound's relative violation.
                multiplier = max(0.0, multiplier + (held - bound) / bound)
            elif (epoch + 1) % check_every == 0 or epoch + 1 == chain.epochs:
                measured = _isolated(constraint)
                # Dual ascent on the measured constraint's relative violation.
                multiplier = max(0.0, multiplier + (measured - limit) / limit)
        if constraint is None:
            held_entry = {"data_loss": held, "bound": bound}
        else:
            held_entry = {"constraint": measured, "limit": limit}
        entry = {
            "round": round_index,
            "chain_pairs": len(own_steps),
            "chain_loss": float(chain_term),
            **held_entry,
            "multiplier": multiplier,
        }
        if measure is not None:
            fidelity = _isolated(measure)
            entry["fidelity"] = fidelity
            log.append(entry)
            if fidelity["score"] < best["score"] - 2.0 * best["score_se"]:
                best, best_state = fidelity, copy.deepcopy(generator.state_dict())
            else:
                break
        else:
            log.append(entry)
    if measure is not None:
        generator.load_state_dict(best_state)
    return log


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
def chain_states(
    generator: ContractGenerator,
    start: torch.Tensor,
    command: torch.Tensor,
    sigma: torch.Tensor,
    steps: int,
    commit: int,
    seed: int = 0,
) -> np.ndarray:
    """``(steps, N, skill)`` the generator's own chain from ``start``, unstable chains dropped.

    From ``start`` -- ``(N, history, skill)`` normalised windows of real ``z*`` -- a
    plan is drawn, its first ``commit`` samples appended to the history (with the
    training noise ``sigma``, which is what the generator reads in a closed loop),
    and so on for ``steps``.
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
    return z[:, finite]


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

    Each plan is calibrated on its own; this asks whether their chain keeps the
    teacher's gait, which no single plan can show (:func:`chain_states`).
    """
    z = chain_states(generator, start, command, sigma, steps, commit, seed)
    done = np.zeros(z.shape[:2], dtype=bool)
    signature = gait_signature(z, done, dt_star).as_dict()
    return {
        "environments_stable": int(z.shape[1]),
        "environments": int(len(start)),
        "signature": signature,
        "rhythm_period": float(dominant_rhythm(z, dt_star).period),
        "segments": _describe(z[..., LOAD] > threshold),
        "min_load": float(z[..., LOAD].min()),
    }


def per_episode(z: np.ndarray, dt_star: float) -> np.ndarray:
    """``(N, 2)``: each column's stride period and stride length (``z`` is ``(T, N, skill)``)."""
    done = np.zeros((z.shape[0], 1), dtype=bool)
    rows = []
    for column in range(z.shape[1]):
        signature = gait_signature(z[:, column : column + 1], done, dt_star)
        rows.append((signature.stride_period, signature.stride_length))
    return np.asarray(rows, dtype=float)


def chain_fidelity(
    generator: ContractGenerator,
    start: torch.Tensor,
    command: torch.Tensor,
    sigma: torch.Tensor,
    steps: int,
    commit: int,
    dt_star: float,
    reference: np.ndarray,
    seed: int = 0,
    draws: int = 1000,
) -> dict:
    """How far the generator's chain walks from its teacher, and how well that is known.

    The deviation is relative, for stride period and stride length, and the
    score is the larger of the two. Its standard error comes from resampling
    the chains and the teacher's episodes (bootstrap): no tolerance is fixed
    here -- a change in the score is real when it exceeds its own error.
    """
    chain = per_episode(
        chain_states(generator, start, command, sigma, steps, commit, seed), dt_star
    )
    teacher = per_episode(reference, dt_star)
    chain, teacher = chain[np.isfinite(chain).all(1)], teacher[np.isfinite(teacher).all(1)]

    def score(c: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
        deviation = np.abs(c.mean(0) - t.mean(0)) / t.mean(0)
        return deviation, float(deviation.max())

    deviation, value = score(chain, teacher)
    rng = np.random.default_rng(seed)
    resampled = [
        score(
            chain[rng.integers(len(chain), size=len(chain))],
            teacher[rng.integers(len(teacher), size=len(teacher))],
        )[1]
        for _ in range(draws)
    ]
    return {
        "deviation": {"stride_period": float(deviation[0]), "stride_length": float(deviation[1])},
        "score": value,
        "score_se": float(np.std(resampled)),
        "chains": int(len(chain)),
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
    "ChainConfig",
    "ContractConfig",
    "chain_fidelity",
    "chain_from",
    "chain_starts",
    "data_loss",
    "chain_states",
    "chained_gait",
    "gait_reference",
    "change_times",
    "contract_steps",
    "evaluate_generator",
    "event_categories",
    "fit_tolerance",
    "futures",
    "new_generator",
    "per_episode",
    "population_scores",
    "train_generator",
    "train_on_chains",
    "windows",
]
