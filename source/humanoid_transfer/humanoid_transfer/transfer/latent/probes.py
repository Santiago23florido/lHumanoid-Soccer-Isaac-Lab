"""How much one signal says about another: probes, fitted and tested apart.

A probe is a small regressor fitted on one set of steps and scored by R^2 on
held-out ones. A linear probe is the usual instrument, but it only finds what
is linearly readable, so it can miss information a downstream network uses;
the MLP probe here closes most of that gap. Inputs are given as a function of
the step indices, so wide inputs -- whole history windows -- are gathered one
batch at a time and never materialised for every step at once.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .model import mlp

Features = Callable[[torch.Tensor], torch.Tensor]
"""Step indices ``(B,)`` to the probe's input ``(B, F)``."""


def r2(prediction: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Per-output coefficient of determination; NaN where the target is constant."""
    residual = ((prediction - target) ** 2).sum(dim=0)
    total = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    score = 1.0 - residual / total
    return torch.where(total > 1e-12, score, torch.nan).cpu().numpy()


def _moments(features: Features, steps: torch.Tensor, batch: int) -> tuple[torch.Tensor, ...]:
    sample = steps[torch.randperm(len(steps), device=steps.device)[: 16 * batch]]
    x = torch.cat([features(sample[i : i + batch]) for i in range(0, len(sample), batch)])
    return x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)


@torch.no_grad()
def _predict(model, features, steps, mean, std, batch) -> torch.Tensor:
    return torch.cat(
        [model((features(steps[i : i + batch]) - mean) / std) for i in range(0, len(steps), batch)]
    )


def mlp_probe(
    features: Features,
    targets: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    epochs: int = 30,
    hidden: int = 256,
    batch: int = 4096,
    seed: int = 0,
) -> np.ndarray:
    """Per-output R^2 on ``test`` of a two-layer MLP fitted on ``train``.

    ``targets`` is indexed by step, like ``features``.
    """
    torch.manual_seed(seed)
    mean, std = _moments(features, train, batch)
    y_mean, y_std = targets[train].mean(dim=0), targets[train].std(dim=0).clamp_min(1e-6)
    model = mlp(mean.shape[0], targets.shape[1], hidden).to(targets.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps_per_epoch = max(1, len(train) // batch)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs * steps_per_epoch)
    for _ in range(epochs):
        order = train[torch.randperm(len(train), device=train.device)]
        for i in range(steps_per_epoch):
            chunk = order[i * batch : (i + 1) * batch]
            prediction = model((features(chunk) - mean) / std)
            loss = torch.mean((prediction - (targets[chunk] - y_mean) / y_std) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
    model.eval()
    prediction = _predict(model, features, test, mean, std, batch) * y_std + y_mean
    return r2(prediction, targets[test])


@torch.no_grad()
def linear_probe(
    features: Features,
    targets: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    batch: int = 8192,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Per-output R^2 on ``test`` of a least-squares map fitted on ``train``.

    The normal equations are accumulated batch by batch (in float64), with a
    small ridge on the standardised inputs so that redundant inputs -- the
    repeated padding of history windows -- do not make them singular.
    """
    mean, std = _moments(features, train, batch)
    width = mean.shape[0] + 1
    gram = torch.zeros(width, width, dtype=torch.float64, device=targets.device)
    cross = torch.zeros(width, targets.shape[1], dtype=torch.float64, device=targets.device)

    def design(steps: torch.Tensor) -> torch.Tensor:
        x = ((features(steps) - mean) / std).double()
        return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=1)

    for i in range(0, len(train), batch):
        x = design(train[i : i + batch])
        gram += x.T @ x
        cross += x.T @ targets[train[i : i + batch]].double()
    gram += ridge * len(train) * torch.eye(width, dtype=gram.dtype, device=gram.device)
    weights = torch.linalg.solve(gram, cross)
    prediction = torch.cat(
        [design(test[i : i + batch]) @ weights for i in range(0, len(test), batch)]
    )
    return r2(prediction.float(), targets[test])


__all__ = ["Features", "linear_probe", "mlp_probe", "r2"]
