"""Two arms over several seeds: do they master different numbers of the family's probes?

The adoption rule asks for more cells mastered by over two standard errors
(``transfer/docs/skill_space.md`` §9). With one seed per arm the count moves by
the run's own luck as much as by the change. Here each arm is several
competence reports (``evaluate_competence.py``), one per seed. The statistic is
the difference in the mean number of probes passed per seed. Its standard error
comes from a hierarchical bootstrap: the probes are resampled (the family's
draw), and within each arm the seeds are resampled too (the training's draw).
The per-check deviations are compared the same way.

Usage::

    python transfer/scripts/compare_seeds.py --group grid \
        --reference transfer/results/competence_g1_warm_r3_r8.json ... \
        --candidate transfer/results/competence_g1_warm_r3_logload.json ... \
        --output transfer/results/compare_seeds_logload_grid.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEVIATIONS = ("speed", "stride_period", "duty_left", "duty_right")


def load(paths: list[Path], group: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(probes, passes (seeds, probes), deviations (seeds, probes, checks))``.

    A path may name several reports of one seed joined by commas (the map and its corners),
    and ``group`` several groups joined by commas: their probes are then pooled."""
    groups = set(group.split(","))
    names, passes, deviations = None, [], []
    for path in paths:
        probes = [p for part in str(path).split(",")
                  for p in json.loads(Path(part).read_text(encoding="utf-8"))["probes"]
                  if p["group"] in groups]  # fmt: skip
        these = [p["name"] for p in probes]
        if names is None:
            names = these
        elif these != names:
            raise ValueError(f"{path}: its probes differ from the first report's")
        passes.append([bool(p["passes"]) for p in probes])
        deviations.append([[float(p["deviation"].get(k, np.nan)) for k in DEVIATIONS]
                           for p in probes])  # fmt: skip
    return names, np.array(passes, dtype=float), np.array(deviations, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--group", default="grid")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    names, ref_pass, ref_dev = load(args.reference, args.group)
    cand_names, cand_pass, cand_dev = load(args.candidate, args.group)
    if cand_names != names:
        raise ValueError("the two arms were measured on different probes")
    rng = np.random.default_rng(args.seed)
    count = len(names)

    def statistic(probes, ref_seeds, cand_seeds):
        passed = (
            cand_pass[cand_seeds][:, probes].sum(1).mean()
            - ref_pass[ref_seeds][:, probes].sum(1).mean()
        )
        dev = np.nanmean(cand_dev[cand_seeds][:, probes], axis=(0, 1)) - np.nanmean(
            ref_dev[ref_seeds][:, probes], axis=(0, 1)
        )
        return passed, dev

    observed_passed, observed_dev = statistic(
        np.arange(count), np.arange(len(ref_pass)), np.arange(len(cand_pass))
    )
    draws_passed, draws_dev = [], []
    for _ in range(args.draws):
        probes = rng.integers(0, count, count)
        ref_seeds = rng.integers(0, len(ref_pass), len(ref_pass))
        cand_seeds = rng.integers(0, len(cand_pass), len(cand_pass))
        p, d = statistic(probes, ref_seeds, cand_seeds)
        draws_passed.append(p)
        draws_dev.append(d)
    se_passed = float(np.std(draws_passed))
    se_dev = np.std(np.array(draws_dev), axis=0)
    report = {
        "group": args.group,
        "probes": count,
        "reference": [p.as_posix() for p in args.reference],
        "candidate": [p.as_posix() for p in args.candidate],
        "passed_per_seed": {
            "reference": ref_pass.sum(1).tolist(),
            "candidate": cand_pass.sum(1).tolist(),
        },
        "passed_difference": float(observed_passed),
        "passed_se": se_passed,
        "clears_two_se": bool(observed_passed > 2 * se_passed),
        "deviation_difference": dict(zip(DEVIATIONS, observed_dev.round(4).tolist(), strict=True)),
        "deviation_se": dict(zip(DEVIATIONS, se_dev.round(4).tolist(), strict=True)),
    }
    print(f"{args.group}: reference {report['passed_per_seed']['reference']}, candidate "
          f"{report['passed_per_seed']['candidate']} -> {observed_passed:+.2f} ± {se_passed:.2f} "
          f"probes ({'clears' if report['clears_two_se'] else 'does not clear'} 2 SE)")  # fmt: skip
    for k in DEVIATIONS:
        print(
            f"  {k:14s} {report['deviation_difference'][k]:+.4f} ± {report['deviation_se'][k]:.4f}"
        )
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
