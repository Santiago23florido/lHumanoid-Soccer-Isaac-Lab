"""The bodies' executions of the base tasks as a generator's data (``method_revision.md`` §6).

A generator reads the history of the body that executes its plans. Trained on its teacher
alone, it has never seen a history like a second body's, and on one it continues what it
reads: under the same flat generator the NAO is asked for a duty factor of 0.550, the G1 for
0.515, the teacher walks at 0.496 (``skill_space.md`` §10.17). The body then overshoots what it
is asked, and the deviation goes round the loop body -> history -> plan -> body.

A body's capture (``capture_contract.py``) can teach a generator two things about that loop:

``teacher``
    Each history the body wrote, paired with the future its *teacher* went on to from the
    window most like it, in phase and command (:func:`.train_contract.phase_targets`, the
    retrieval the chains already use for the generator's own drift). The generator learns to
    lead each body's states back to the task's gait: DAgger in ``z*``, with no expert on the
    body.
``own``
    Each history paired with the future the body itself went on to: the body's execution as a
    demonstration. The generator learns to continue each body's deviation.

Every body capture is read in ``z*`` only. Its observations and actions are the body's own and
never reach the generator, which is why these captures are not :func:`.sequences.load_streams`
data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...common.similarity import UnitSystem
from .contract import ContractGenerator
from .dataset import dimensionless_command, falling_steps
from .sequences import Streams, _column_bounds
from .train_contract import contract_steps, futures, phase_targets, windows

TARGETS = ("teacher", "own")


def load_body_streams(paths: list[Path]) -> tuple[Streams, torch.Tensor]:
    """Every column of every body capture in ``z*``, and each step's capture index.

    Observations and actions are left empty: a body's are not the teacher's, and only its
    ``z*`` is read.
    """
    skill, command, falling, start, end, capture, env = [], [], [], [], [], [], []
    offset = 0
    for index, path in enumerate(paths):
        data = np.load(path)
        units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
        done = data["done"]
        steps, envs = done.shape
        terminated = data["terminated"] if "terminated" in data.files else np.zeros_like(done)

        def columns(array: np.ndarray) -> np.ndarray:
            return np.swapaxes(array, 0, 1).reshape(-1, *array.shape[2:])

        skill.append(columns(data["skill_state"]))
        command.append(columns(dimensionless_command(data["command"], units)))
        falling.append(columns(falling_steps(terminated)))
        capture.append(np.full(steps * envs, index))
        env.append(np.repeat(np.arange(envs), steps))
        for column in range(envs):
            first, last = _column_bounds(done[:, column])
            start.append(first + offset)
            end.append(last + offset)
            offset += steps

    def stack(parts: list[np.ndarray], dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.concatenate(parts), dtype=dtype)

    size = offset
    streams = Streams(
        obs=torch.zeros(size, 0),
        skill=stack(skill, torch.float32),
        command=stack(command, torch.float32),
        action=torch.zeros(size, 0),
        falling=stack(falling, torch.bool),
        start=stack(start, torch.long),
        end=stack(end, torch.long),
        capture=stack(capture, torch.long),
        env=stack(env, torch.long),
        target=None,
    )
    return streams, streams.capture


@dataclass
class BodyPairs:
    """Histories a body wrote and the futures a generator is trained toward from them."""

    window: torch.Tensor
    """``(N, history, skill)``, clean and normalised."""

    command: torch.Tensor
    """``(N, 3)``, dimensionless."""

    future: torch.Tensor
    """``(N, horizon, skill)``, normalised."""

    group: torch.Tensor
    """``(N,)``: which body wrote each history, as the caller numbered the captures."""

    steps: torch.Tensor
    """``(N,)``: each history's last step in the body's streams."""

    def __len__(self) -> int:
        return len(self.window)


@torch.no_grad()
def body_pairs(
    generator: ContractGenerator,
    body: Streams,
    groups: torch.Tensor,
    teacher: Streams,
    teacher_steps: torch.Tensor,
    targets: str,
    pairs: int,
    bank: int = 32768,
    seed: int = 0,
    command_from: str = "body",
) -> BodyPairs:
    """Up to ``pairs`` of the body's histories, each with its ``targets`` future.

    ``groups`` gives, per body step, the body that wrote it (one number per body). The
    histories come from steps whose whole future lies in the episode and away from a fall, as
    the teacher's do (:func:`.train_contract.contract_steps`).

    ``command_from``: the command each pair is trained with. ``body``, the one the body was
    given (it also keys the retrieval). ``target``, with ``teacher`` targets only: the
    teacher's command at the step its future was taken from. Under relabelled teacher
    streams (:mod:`.hindsight`), that is the command the target achieved.
    """
    if targets not in TARGETS:
        raise ValueError(f"targets must be one of {TARGETS}, not {targets!r}")
    if command_from not in ("body", "target"):
        raise ValueError(f"command_from must be 'body' or 'target', not {command_from!r}")
    if command_from == "target" and targets != "teacher":
        raise ValueError("command_from='target' needs teacher targets")
    order = torch.Generator(device="cpu").manual_seed(seed)
    steps = contract_steps(body, generator.horizon)
    steps = steps[torch.randperm(len(steps), generator=order)[:pairs].to(steps.device)]
    window = windows(generator, body, steps, None)
    command = body.command[steps]
    if targets == "own":
        future = futures(generator, body, steps)
    else:
        picked = torch.randperm(len(teacher_steps), generator=order)[:bank]
        future, source = phase_targets(
            generator,
            teacher,
            teacher_steps[picked.to(teacher_steps.device)],
            window,
            command,
            return_index=True,
        )
        if command_from == "target":
            command = teacher.command[source]
    return BodyPairs(window, command, future, groups[steps], steps)


class _Reverse(torch.autograd.Function):
    """The identity forward; the gradient negated and scaled backward (Ganin and Lempitsky,
    2015): what the adversary learns to read, the summary learns to stop carrying."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad, None


def summary(generator: ContractGenerator, window: torch.Tensor) -> torch.Tensor:
    """The learned part of the generator's conditioning: its summary of the history.

    The rest of what it conditions on is ``z*`` itself (the latest state), the command and
    the noise level (:meth:`.contract.ContractGenerator.condition`).
    """
    return generator.memory(generator.warp(window))


def mmd2(a: torch.Tensor, b: torch.Tensor, scales: tuple[float, ...] = (0.5, 1.0, 2.0)):
    """Unbiased squared MMD between samples ``a`` ``(n, d)`` and ``b`` ``(m, d)`` (Gretton et
    al., 2012), with a sum of Gaussian kernels whose bandwidths are ``scales`` times the median
    pairwise distance of the pooled samples (held fixed, not differentiated): scale-free, so
    the summary cannot shrink its way to a small value. Zero in expectation when ``a`` and
    ``b`` come from one distribution."""
    pooled = torch.cat([a, b])
    distance = torch.cdist(pooled, pooled).pow(2)
    with torch.no_grad():
        median = distance[distance > 0].median().clamp_min(1e-12)
    kernel = sum(torch.exp(-distance / (2.0 * s * s * median)) for s in scales)
    n, m = len(a), len(b)
    kaa, kbb, kab = kernel[:n, :n], kernel[n:, n:], kernel[:n, n:]
    within_a = (kaa.sum() - kaa.diagonal().sum()) / (n * (n - 1))
    within_b = (kbb.sum() - kbb.diagonal().sum()) / (m * (m - 1))
    return within_a + within_b - 2.0 * kab.mean()


def _balanced_ce(logits: torch.Tensor, labels: torch.Tensor, classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=classes).clamp_min(1).float()
    weight = counts.sum() / (classes * counts)
    return torch.nn.functional.cross_entropy(logits, labels, weight=weight)


def train_on_bodies(
    streams: Streams,
    steps: torch.Tensor,
    held_out: torch.Tensor,
    generator: ContractGenerator,
    sigma: torch.Tensor,
    config,
    pairs: BodyPairs,
    epochs: int,
    invariance: float = 0.0,
    seed: int = 0,
    invariance_kind: str = "adversary",
) -> list[dict[str, float]]:
    """Train on the bodies' pairs as a constrained problem, as the chains are trained.

        min  L_bodies(θ)   s.t.  L_data(θ) <= L_data(θ₀) + 2 SE

    on the teacher's held-out steps (:func:`.train_contract.train_on_chains` explains the
    bound and the dual ascent on its multiplier). With ``invariance`` > 0 the generator's
    history summary (:func:`summary`) is trained to carry no body, the teacher's windows
    counting as the G1's (group 0), in one of two ways (``invariance_kind``):

    ``adversary``
        An adversary is trained to tell the bodies apart, and the summary, through a
        reversed gradient of weight ``invariance``, to make that impossible (W16).
    ``mmd``
        A second constraint, with its own dual ascent, starting at ``invariance``:

            MMD²(other bodies, group 0)  <=  MMD²(group 0, group 0)

        the right side being the same estimate between two random halves of group 0 in the
        batch: the bodies may differ in the summary no more than group 0 differs from itself
        (:func:`mmd2`, at most ``MMD_SAMPLES`` per side). No adversary, so nothing to
        maximise (W17).
    """
    if invariance_kind not in ("adversary", "mmd"):
        raise ValueError(f"invariance_kind must be 'adversary' or 'mmd', not {invariance_kind!r}")
    from .contract import contract_loss
    from .train_contract import data_loss

    device = steps.device
    order = torch.Generator(device="cpu").manual_seed(seed + 2)
    generator.eval()
    base, base_se = data_loss(generator, streams, held_out, sigma, config)
    bound = base + 2.0 * base_se
    multiplier = 1.0
    classes = int(pairs.group.max()) + 1
    adversary = None
    use_mmd = invariance > 0.0 and invariance_kind == "mmd"
    mmd_multiplier, mmd_scale = invariance, None
    parameters = list(generator.parameters())
    if invariance > 0.0 and invariance_kind == "adversary":
        width = summary(generator, pairs.window[:2]).shape[-1]
        adversary = torch.nn.Sequential(
            torch.nn.Linear(width, 128), torch.nn.ELU(), torch.nn.Linear(128, classes)
        ).to(device)
        parameters += list(adversary.parameters())
    optimiser = torch.optim.Adam(parameters, lr=config.learning_rate)
    per_epoch = max(1, -(-len(pairs) // config.batch_size))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs * per_epoch)
    log: list[dict[str, float]] = [{"epoch": -1, "data_loss": base, "bound": bound}]
    for epoch in range(epochs):
        generator.train()
        shuffled = torch.randperm(len(pairs), generator=order).to(device)
        gaps, references = [], []
        for start in range(0, len(shuffled), config.batch_size):
            batch = shuffled[start : start + config.batch_size]
            real = steps[torch.randint(len(steps), (len(batch),), generator=order).to(device)]
            window = pairs.window[batch]
            noisy = window + sigma * torch.randn_like(window)
            body_term = contract_loss(
                generator, noisy, pairs.command[batch], pairs.future[batch], config.free_bits
            )["total"]
            teacher_window = windows(generator, streams, real, sigma)
            data_term = contract_loss(
                generator,
                teacher_window,
                streams.command[real],
                futures(generator, streams, real),
                config.free_bits,
            )["total"]
            total = body_term + multiplier * data_term
            if adversary is not None:
                features = torch.cat(
                    [summary(generator, noisy), summary(generator, teacher_window)]
                )
                labels = torch.cat([pairs.group[batch], torch.zeros_like(real)])
                logits = adversary(_Reverse.apply(features, invariance))
                total = total + _balanced_ce(logits, labels, classes)
            if use_mmd:
                gap, reference = _invariance_terms(
                    summary(generator, noisy), pairs.group[batch],
                    summary(generator, teacher_window), order,
                )  # fmt: skip
                if gap is not None:
                    total = total + mmd_multiplier * gap
                    gaps.append(float(gap))
                    references.append(reference)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            optimiser.step()
            schedule.step()
        generator.eval()
        held, _ = data_loss(generator, streams, held_out, sigma, config)
        multiplier = max(0.0, multiplier + (held - bound) / bound)
        entry = {
            "epoch": epoch,
            "body_loss": float(body_term),
            "data_loss": held,
            "multiplier": multiplier,
        }
        if adversary is not None:
            entry["adversary_loss"] = float(_balanced_ce(logits.detach(), labels, classes))
        if use_mmd and gaps:
            gap, reference = float(np.mean(gaps)), float(np.mean(references))
            mmd_scale = mmd_scale or max(abs(gap), 1e-6)
            mmd_multiplier = max(0.0, mmd_multiplier + (gap - reference) / mmd_scale)
            entry.update(mmd=gap, mmd_reference=reference, mmd_multiplier=mmd_multiplier)
        log.append(entry)
    generator.eval()
    return log


MMD_SAMPLES = 1024


def _invariance_terms(
    body: torch.Tensor, group: torch.Tensor, teacher: torch.Tensor, order: torch.Generator
) -> tuple[torch.Tensor | None, float]:
    """The MMD constraint's two sides on one batch: other bodies against group 0, and group
    0's two halves against each other (detached)."""
    other = body[group != 0]
    zero = torch.cat([body[group == 0], teacher])
    if len(other) < 2 or len(zero) < 4:
        return None, 0.0

    def take(x: torch.Tensor, count: int) -> torch.Tensor:
        pick = torch.randperm(len(x), generator=order)[:count].to(x.device)
        return x[pick]

    other = take(other, MMD_SAMPLES)
    zero = take(zero, 2 * MMD_SAMPLES)
    half = len(zero) // 2
    with torch.no_grad():
        reference = float(mmd2(zero[:half], zero[half : 2 * half]))
    return mmd2(other, zero[:half]), reference


@torch.no_grad()
def _features(generator: ContractGenerator, pairs: BodyPairs, sigma: torch.Tensor):
    window = pairs.window + sigma * torch.randn_like(pairs.window)
    return summary(generator, window)


def leakage(
    generator: ContractGenerator,
    pairs: BodyPairs,
    sigma: torch.Tensor,
    seed: int = 0,
    epochs: int = 300,
) -> dict[str, float]:
    """How well a fresh probe tells the bodies apart from the generator's history summary.

    A two-layer probe is trained on 70 % of the pairs and scored on the rest, by balanced
    accuracy; the control is the same probe on labels shuffled among the pairs. The summary
    leaks the body when the probe beats its control. Chance is ``1 / bodies``; ``se`` is the
    binomial standard error of the balanced accuracy on the test pairs.
    """
    torch.manual_seed(seed)
    features = _features(generator, pairs, sigma)
    features = (features - features.mean(0)) / features.std(0).clamp_min(1e-6)
    labels = pairs.group
    classes = int(labels.max()) + 1
    order = torch.randperm(len(labels), device=labels.device)
    cut = int(0.7 * len(labels))
    train, test = order[:cut], order[cut:]

    def probe(target: torch.Tensor) -> float:
        model = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], 64), torch.nn.ELU(), torch.nn.Linear(64, classes)
        ).to(features.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
        with torch.enable_grad():
            for _ in range(epochs):
                optimiser.zero_grad(set_to_none=True)
                _balanced_ce(model(features[train]), target[train], classes).backward()
                optimiser.step()
        predicted = model(features[test]).argmax(dim=1)
        recalls = [
            float((predicted[target[test] == c] == c).float().mean())
            for c in range(classes)
            if bool((target[test] == c).any())
        ]
        return float(np.mean(recalls))

    shuffled = labels[torch.randperm(len(labels), device=labels.device)]
    accuracy = probe(labels)
    return {
        "balanced_accuracy": accuracy,
        "control": probe(shuffled),
        "chance": 1 / classes,
        "test_pairs": len(test),
        "se": math.sqrt(max(accuracy * (1 - accuracy), 1e-6) / len(test)),
    }


__all__ = [
    "MMD_SAMPLES",
    "TARGETS",
    "BodyPairs",
    "body_pairs",
    "leakage",
    "load_body_streams",
    "mmd2",
    "summary",
    "train_on_bodies",
]
