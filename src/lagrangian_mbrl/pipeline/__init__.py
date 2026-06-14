"""Dynamics-learning benchmark pipeline: data → train → test, multi-seed + HPO.

This subpackage turns the Phase-0 offline comparison into a full, *comparable*
benchmark that you can launch once and leave running:

- :mod:`lagrangian_mbrl.pipeline.registry`   — recommended network architectures
  for learning manipulator dynamics, behind one ``build_model`` factory, plus
  each model's hyper-parameter search space.
- :mod:`lagrangian_mbrl.pipeline.data`       — a data source abstraction
  (analytic rigid-body systems now; logged Isaac Lab Franka transitions later),
  all yielding the same ``(q, q̇, τ, q̈)`` schema.
- :mod:`lagrangian_mbrl.pipeline.experiment` — train+test one (model, seed,
  hyper-parameter) configuration and return metrics.
- :mod:`lagrangian_mbrl.pipeline.hpo`        — hyper-parameter search (Optuna if
  installed, built-in random search otherwise — no extra deps required).
- :mod:`lagrangian_mbrl.pipeline.benchmark`  — the orchestrator: for every model,
  tune hyper-parameters, retrain over many seeds, and aggregate with bootstrap
  confidence intervals into a comparable results table + plots.

Nothing here needs Isaac Sim: the default analytic data source runs on CPU, so
the whole benchmark is runnable tonight; swap ``source="franka"`` once Isaac Lab
logs are available.
"""

from lagrangian_mbrl.pipeline.registry import MODEL_REGISTRY, build_model, model_hp_space
from lagrangian_mbrl.pipeline.sample_complexity import (
    SampleComplexityConfig,
    run_sample_complexity,
)

__all__ = [
    "MODEL_REGISTRY",
    "SampleComplexityConfig",
    "build_model",
    "model_hp_space",
    "run_sample_complexity",
]
