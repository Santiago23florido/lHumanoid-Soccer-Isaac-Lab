"""When the bodies fall on sidestep, and what the plan was asking then (``skill_space.md``
§10.21, read after the screen; instrumentation, not a criterion).

From the L4 captures on sidestep under one held-out generator, per body and tracker arm
(``<prefix>_<body>_s<seed>_sym``), over the seeds on disk:

- **age at fall**: steps since the episode's reset (the capture's settle steps count for the
  episodes that began before the recording), its 10th, 50th and 90th percentiles, and the share
  of falls within ``EARLY`` steps of a reset;
- **stalled starts**: in the autonomous loop a reset fills the generator's history with the
  body's state at rest, and the generator then reads its own plans. A start is *stalled* when
  its plan asks for double support (both loads above 0.1 of the weight) on more than
  ``STALLED`` of the window's steps:
  - episodes that began before the recording: window = the first 20 recorded steps (ages
    100-120), fall = a termination within the first ``EARLY`` recorded steps;
  - episodes that began inside it: window = ages 40-80, fall = a termination before age
    ``EARLY + 100``; those that fall before age 80, or end unobserved before then, are left out;

  with the share of stalled starts and the fall rate of stalled and walking starts;
- **runs completed**: the environments (32 per command and seed) that never fall over the
  recorded 900 steps, and their share.

Before the first resynchronisation the plans do not depend on the tracker, so the share of
stalled starts is the generator's, and the fall rate given a stalled start is the tracker's.

    python transfer/scripts/w20_starts.py --output transfer/results/w20_starts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
SEEDS = {"g1": (41, 42, 43), "nao": (31, 32, 33)}
COMMANDS = ("left", "left_fast", "right")
LOADED = 0.1
STALLED = 0.4
EARLY = 200


def ages(done: np.ndarray, settle: int) -> np.ndarray:
    """``(T, E)`` steps since each env's reset, ``settle`` for the first recorded step."""
    age = np.zeros(done.shape, dtype=int)
    current = np.full(done.shape[1], settle)
    for t in range(done.shape[0]):
        if t > 0:
            current = np.where(done[t - 1], 0, current + 1)
        age[t] = current
    return age


def starts(data) -> tuple[list[tuple[str, float, bool]], list[int]]:
    """Each observed start as (kind, plan's double-support share, fell), and every fall's age."""
    done, terminated = data["done"], data["terminated"]
    plan = data["contract_due"]
    double = (plan[..., 22:24] > LOADED).all(-1)
    age = ages(done, int(data["settle_steps"]))
    steps, envs = done.shape
    rows = []
    for e in range(envs):
        ends = np.flatnonzero(done[:, e])
        fell = ends.size > 0 and bool(terminated[ends[0], e]) and ends[0] < EARLY
        rows.append(("first", float(double[:20, e].mean()), fell))
        for k, end in enumerate(ends):
            begin = end + 1
            following = ends[k + 1] if k + 1 < len(ends) else None
            if begin + 80 >= steps or (following is not None and following < begin + 80):
                continue
            if following is None and steps - begin < EARLY + 100:
                continue
            fell = (
                following is not None
                and bool(terminated[following, e])
                and following - begin < EARLY + 100
            )
            rows.append(("restart", float(double[begin + 40 : begin + 80, e].mean()), fell))
    return rows, age[terminated].tolist()


def summary(rows: list[tuple[str, float, bool]], kind: str) -> dict:
    mine = [r for r in rows if r[0] == kind]
    double = np.array([r[1] for r in mine])
    fell = np.array([r[2] for r in mine])
    stalled = double > STALLED

    def rate(mask):
        return float(fell[mask].mean()) if mask.any() else None

    return {
        "starts": len(mine),
        "stalled_share": float(stalled.mean()) if mine else None,
        "stalled": int(stalled.sum()),
        "fall_given_stalled": rate(stalled),
        "fall_given_walking": rate(~stalled),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prefixes", nargs="+", default=["mt", "xf", "xs"])
    parser.add_argument("--generator", default="mt18_no_sidestep")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict = {"generator": args.generator, "stalled": STALLED, "early": EARLY, "bodies": {}}
    for body, seeds in SEEDS.items():
        report["bodies"][body] = {}
        for prefix in args.prefixes:
            rows, fall_ages, used = [], [], []
            runs = completed = 0
            for seed in seeds:
                paths = [
                    REPO_ROOT
                    / body
                    / "rollouts"
                    / (
                        f"zs_{prefix}_{body}_s{seed}_sym_{args.generator}_own_auto_kin_"
                        f"{command}_encoder.npz"
                    )
                    for command in COMMANDS
                ]
                if not all(p.exists() for p in paths):
                    continue
                used.append(seed)
                for path in paths:
                    data = np.load(path)
                    r, a = starts(data)
                    rows += r
                    fall_ages += a
                    runs += data["terminated"].shape[1]
                    completed += int((~data["terminated"].any(0)).sum())
            if not used:
                continue
            a = np.array(fall_ages)
            entry = {
                "seeds": used,
                "falls": int(a.size),
                "runs": runs,
                "runs_completed": completed,
                "completed_share": completed / runs,
                "fall_age_p10_p50_p90": np.percentile(a, [10, 50, 90]).tolist() if a.size else None,
                "falls_early_share": float((a <= EARLY).mean()) if a.size else None,
                "first": summary(rows, "first"),
                "restart": summary(rows, "restart"),
            }
            report["bodies"][body][prefix] = entry
            f = entry["first"]
            print(
                f"{body} {prefix} seeds {used} falls {entry['falls']:4d}"
                f" completed {completed}/{runs}"
                f" early {entry['falls_early_share'] or 0:.2f}"
                f"  first: stalled {f['stalled_share']:.2f}"
                f" P(fall|stalled) {f['fall_given_stalled'] if f['stalled'] else float('nan'):.2f}"
                f" P(fall|walking) {f['fall_given_walking']:.2f}"
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
