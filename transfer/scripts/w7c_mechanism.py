"""Where W7c's gait went: contact timing and swing height of the three NAO groups at gain 2.

W7c trained S's NAO recipe at ``HUMANOID_NAO_LEG_GAIN=2`` and
compared it with S's seeds frozen at gain 2 and at the derived gains. Its criterion reads
gate C. This reads how each group's feet move, on the same L4 captures of the deployed
path (both skills x three commands), with ``w6_mechanism.condition``: the stance level and
absolute swing peak, the clearance above the stance, the swing's air time against the
plan's, the duty, and the lift-off and touchdown lags (``common.contact``; positive is
later than the plan, in ``T``).

Per seed, the mean over the six captures; then the seed mean and its standard error.

    python transfer/scripts/w7c_mechanism.py --output transfer/results/w7c_mechanism.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from w6_mechanism import COMMANDS, SKILLS, condition, mean  # noqa: E402

SEEDS = (31, 32, 33)
GROUPS = {
    "derived": "warm_nao_r1_s{seed}_sym",
    "frozen_g2": "warm_nao_r1_s{seed}_sym_g2",
    "arm": "warm_nao_g2_s{seed}_sym",
}
KEYS = (
    "stance_level_body",
    "peak_body",
    "clearance_body",
    "air_time_ratio",
    "duty_body",
    "duty_asked",
    "liftoff_lag_mean",
    "touchdown_lag_mean",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {}
    for name, pattern in GROUPS.items():
        seeds = {}
        for seed in SEEDS:
            tag = pattern.format(seed=seed)
            rows = [condition(tag, s, c) for s in SKILLS for c in COMMANDS]
            seeds[seed] = {k: v for k, v in mean(rows).items() if k in KEYS}
        summary = {}
        for key in KEYS:
            values = np.array([seeds[s][key] for s in SEEDS])
            summary[key] = {
                "mean": float(values.mean()),
                "se": float(values.std(ddof=1) / np.sqrt(len(values))),
            }
        report[name] = {"seeds": seeds, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, entry in report.items():
        print(
            f"{name:10s} "
            + "  ".join(
                f"{k} {v['mean']:.4f} +/- {v['se']:.4f}" for k, v in entry["summary"].items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
