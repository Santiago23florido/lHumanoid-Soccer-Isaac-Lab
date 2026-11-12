"""W7c's reading: NAO trackers trained on faster drives.

The arm (``warm_nao_g2_s<seed>``: r1_3498 continued 1000 iterations as S's NAO seeds were,
with ``HUMANOID_NAO_LEG_GAIN=2`` in training and evaluation) against two controls, S's NAO
seeds 31-33, all on the L4 captures of the deployed path, both skills x three commands:

* **frozen at gain 2** -- S's seeds deployed on the same faster drives without training on
  them (W7b's captures; seed 31's from ``w7c_train.sh``). This is the comparison the
  criterion reads: what training on the faster body adds to deploying on it.
* **derived gains** -- S's seeds as recorded. Reported beside it: the whole change against
  the system as it stands.

Criterion, against the frozen control: the largest gate C deviation (the paper's reading:
per command the largest of the stride, period and duty deviations; mean over both skills and
three commands) of the arm's seed mean below the control's by more than twice the standard
error of the difference of seed means, ``sqrt(s_a^2/3 + s_c^2/3)``. No regression: falls
(summed, at most two more), the speed ratio at least 0.9 at every command, S0's walk
passing, and the lift (``w7_compare``) not lower by more than 2 SE.

    python transfer/scripts/w7c_compare.py --output transfer/results/w7c_trained.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
RESULTS = REPO_ROOT / "transfer" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w7_compare import COMMANDS, SKILLS  # noqa: E402
from w7_compare import features as contact_features  # noqa: E402

SEEDS = (31, 32, 33)
ARM = {seed: f"warm_nao_g2_s{seed}_sym" for seed in SEEDS}
FROZEN = {seed: f"warm_nao_r1_s{seed}_sym_g2" for seed in SEEDS}
DERIVED = {seed: f"warm_nao_r1_s{seed}_sym" for seed in SEEDS}
SIGNATURE = ("stride_length", "stride_period", "duty_factor_0", "duty_factor_1")
KEYS = ("largest", "lift", "duty", "duty_body", "passes", "falls")


def largest(tag: str) -> float | None:
    values = []
    for skill in SKILLS:
        for command in COMMANDS:
            path = RESULTS / f"zero_shot_nao_{tag}_{skill}_auto_kin_l4_{command}.json"
            if not path.exists():
                return None
            run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
            dev = run["verdict"]["signature_deviation"]
            values.append(max(dev[k] for k in SIGNATURE))
    return float(np.mean(values))


def features(tag: str) -> dict | None:
    contact = contact_features(tag)
    peak = largest(tag)
    if contact is None or peak is None:
        return None
    return {"largest": peak, **contact}


def compare(arm: dict, control: dict) -> dict:
    out = {}
    for key in KEYS:
        a = np.array([f[key] for f in arm.values()], dtype=float)
        c = np.array([f[key] for f in control.values()], dtype=float)
        se = float(np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)))
        out[key] = {"arm": float(a.mean()), "control": float(c.mean()), "se": se}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = {
        name: {s: f for s, t in tags.items() if (f := features(t)) is not None}
        for name, tags in (("arm", ARM), ("frozen_g2", FROZEN), ("derived", DERIVED))
    }
    report: dict = {"groups": groups}
    arm, frozen = groups["arm"], groups["frozen_g2"]
    if len(arm) == 3 and len(frozen) == 3:
        stats = compare(arm, frozen)
        diff = {k: v["arm"] - v["control"] for k, v in stats.items()}
        regressions = {
            "falls": sum(f["falls"] for f in arm.values())
            <= sum(f["falls"] for f in frozen.values()) + 2,
            "speed": all(f["speed_ratio_min"] >= 0.9 for f in arm.values()),
            "walk": all(
                f["walk"] is not None
                and f["walk"]["speed_ratio"] >= 0.9
                and f["walk"]["fall_rate_excess"] <= 0.02
                for f in arm.values()
            ),
            "lift": diff["lift"] >= -2 * stats["lift"]["se"],
        }
        target = diff["largest"] < -2 * stats["largest"]["se"]
        report.update(
            against_frozen=stats,
            target_met=bool(target),
            no_regression=regressions,
            passes=bool(target and all(regressions.values())),
        )
        if len(groups["derived"]) == 3:
            report["against_derived"] = compare(arm, groups["derived"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, group in groups.items():
        for seed, f in sorted(group.items()):
            print(
                f"{name:9s} s{seed}: largest {f['largest']:.3f}  lift {f['lift']:.3f}  "
                f"duty {f['duty']:.3f}  passes {f['passes']}  falls {f['falls']}"
            )
    for key in ("target_met", "no_regression", "passes"):
        if key in report:
            print(f"{key}: {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
