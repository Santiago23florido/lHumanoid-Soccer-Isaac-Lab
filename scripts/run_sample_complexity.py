#!/usr/bin/env python
"""Run the matched data-budget sweep used to estimate empirical kappa."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from lagrangian_mbrl.pipeline.sample_complexity import (
    SampleComplexityConfig,
    run_sample_complexity,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmark/sample_complexity.yaml",
        help="YAML configuration.",
    )
    parser.add_argument("--out", default=None, help="Output directory.")
    return parser.parse_args()


def _load_config(path: str | Path) -> SampleComplexityConfig:
    values: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fields = {field.name for field in dataclasses.fields(SampleComplexityConfig)}
    unknown = set(values) - fields
    if unknown:
        raise KeyError(f"Unknown config keys {sorted(unknown)}; valid: {sorted(fields)}")
    return SampleComplexityConfig(**values)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    output = args.out or str(
        Path(config.out_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    results = run_sample_complexity(config, out_dir=output)
    print(f"Results: {output}")
    for row in results["empirical_kappa"]:
        print(
            f"target={row['target_mse']:.3e} "
            f"N_struct={row['structured_n']} N_cmp={row['comparator_n']} "
            f"kappa={row['kappa_empirical']}"
        )


if __name__ == "__main__":
    main()
