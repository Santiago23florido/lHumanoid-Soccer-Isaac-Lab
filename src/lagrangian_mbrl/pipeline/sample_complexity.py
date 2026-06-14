"""Data-budget sweep for the empirical sample-complexity comparison."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lagrangian_mbrl.pipeline.data import train_val_datasets
from lagrangian_mbrl.pipeline.experiment import run_single
from lagrangian_mbrl.pipeline.hpo import tune


@dataclass
class SampleComplexityConfig:
    """Configuration for a matched data-budget sweep."""

    source: str = "two_link"
    models: tuple[str, ...] = ("delan", "mlp_ensemble")
    budgets: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
    n_val: int = 2048
    tuning_n_train: int = 256
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hpo_trials: int = 20
    hpo_epochs: int = 150
    train_epochs: int = 600
    dtype: str = "float64"
    n_bootstrap: int = 10_000
    target_mses: tuple[float, ...] = (0.1, 0.05, 0.01)
    structured_model: str = "delan"
    comparator_model: str = "mlp_ensemble"
    out_dir: str = "logs/sample_complexity"


def _bootstrap_ci(
    values: np.ndarray, n_bootstrap: int, seed: int = 0
) -> tuple[float, float]:
    if values.size <= 1:
        value = float(values.mean()) if values.size else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def first_budget_to_threshold(rows: list[dict[str, Any]], target_mse: float) -> int | None:
    """Return the first budget whose cumulative-best mean MSE reaches a target."""
    best = float("inf")
    for row in sorted(rows, key=lambda item: item["n_train"]):
        best = min(best, float(row["accel_mse_mean"]))
        if best <= target_mse:
            return int(row["n_train"])
    return None


def empirical_kappas(
    model_rows: dict[str, list[dict[str, Any]]],
    targets: tuple[float, ...],
    structured_model: str,
    comparator_model: str,
) -> list[dict[str, Any]]:
    """Compute ``N_comparator / N_structured`` at predeclared MSE targets."""
    if structured_model not in model_rows or comparator_model not in model_rows:
        return []
    output = []
    for target in targets:
        structured_n = first_budget_to_threshold(model_rows[structured_model], target)
        comparator_n = first_budget_to_threshold(model_rows[comparator_model], target)
        kappa = (
            comparator_n / structured_n
            if structured_n is not None and comparator_n is not None
            else None
        )
        output.append(
            {
                "target_mse": target,
                "structured_n": structured_n,
                "comparator_n": comparator_n,
                "kappa_empirical": kappa,
            }
        )
    return output


def run_sample_complexity(
    cfg: SampleComplexityConfig | None = None,
    *,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Tune once, retrain across budgets/seeds, and save comparable curves."""
    cfg = cfg or SampleComplexityConfig()
    dtype = getattr(torch, cfg.dtype)
    output_dir = Path(out_dir or cfg.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"config": asdict(cfg), "models": {}}

    tuning_train, tuning_val, dof = train_val_datasets(
        cfg.source,
        cfg.tuning_n_train,
        cfg.n_val,
        seed=cfg.seeds[0],
        dtype=dtype,
    )

    for model_name in cfg.models:
        hpo = tune(
            model_name,
            tuning_train,
            tuning_val,
            dof,
            n_trials=cfg.hpo_trials,
            epochs=cfg.hpo_epochs,
            seed=cfg.seeds[0],
            dtype=dtype,
        )
        rows: list[dict[str, Any]] = []
        for budget in sorted(cfg.budgets):
            per_seed = []
            for seed in cfg.seeds:
                train, val, budget_dof = train_val_datasets(
                    cfg.source, budget, cfg.n_val, seed=seed, dtype=dtype
                )
                per_seed.append(
                    run_single(
                        model_name,
                        hpo["best_hp"],
                        train,
                        val,
                        budget_dof,
                        epochs=cfg.train_epochs,
                        seed=seed,
                        dtype=dtype,
                    )
                )
            mses = np.asarray(
                [run["best_val_accel_mse"] for run in per_seed], dtype=float
            )
            ci_low, ci_high = _bootstrap_ci(
                mses, cfg.n_bootstrap, seed=budget + cfg.seeds[0]
            )
            rows.append(
                {
                    "n_train": budget,
                    "accel_mse_mean": float(mses.mean()),
                    "accel_mse_std": (
                        float(mses.std(ddof=1)) if mses.size > 1 else 0.0
                    ),
                    "accel_mse_ci95": [ci_low, ci_high],
                    "per_seed_accel_mse": mses.tolist(),
                    "mean_wall_time_s": float(
                        np.mean([run["wall_time_s"] for run in per_seed])
                    ),
                    "num_params": per_seed[0]["num_params"],
                }
            )
        results["models"][model_name] = {
            "best_hp": hpo["best_hp"],
            "hpo_backend": hpo["backend"],
            "hpo_best_value": hpo["best_value"],
            "rows": rows,
        }

    model_rows = {
        name: model_result["rows"] for name, model_result in results["models"].items()
    }
    results["empirical_kappa"] = empirical_kappas(
        model_rows,
        tuple(cfg.target_mses),
        cfg.structured_model,
        cfg.comparator_model,
    )
    _write_csv(results, output_dir / "sample_complexity.csv")
    (output_dir / "sample_complexity_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    _plot(results, output_dir / "sample_complexity.png")
    return results


def _write_csv(results: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "model",
        "n_train",
        "accel_mse_mean",
        "accel_mse_std",
        "ci95_low",
        "ci95_high",
        "num_params",
        "mean_wall_time_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_result in results["models"].items():
            for row in model_result["rows"]:
                writer.writerow(
                    {
                        "model": model_name,
                        "n_train": row["n_train"],
                        "accel_mse_mean": row["accel_mse_mean"],
                        "accel_mse_std": row["accel_mse_std"],
                        "ci95_low": row["accel_mse_ci95"][0],
                        "ci95_high": row["accel_mse_ci95"][1],
                        "num_params": row["num_params"],
                        "mean_wall_time_s": row["mean_wall_time_s"],
                    }
                )


def _plot(results: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for model_name, model_result in results["models"].items():
        rows = model_result["rows"]
        budgets = np.asarray([row["n_train"] for row in rows])
        means = np.asarray([row["accel_mse_mean"] for row in rows])
        lows = np.asarray([row["accel_mse_ci95"][0] for row in rows])
        highs = np.asarray([row["accel_mse_ci95"][1] for row in rows])
        axis.plot(budgets, means, marker="o", label=model_name)
        axis.fill_between(budgets, lows, highs, alpha=0.2)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("training transitions N")
    axis.set_ylabel("held-out one-step acceleration MSE")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
