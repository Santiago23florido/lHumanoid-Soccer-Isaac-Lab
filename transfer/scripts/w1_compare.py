"""W1's verdict, read from the L4 results it produced.

For each frozen NAO tracker, the linear captures (recorded before) against the ones with
``HUMANOID_RESAMPLE=generator`` (``w1_resample.sh``), on the deployed path:

* **target** -- the mean over the three commands, both skills and both feet of the duty
  factors' relative deviation from the teacher's own grid; the generator space must lower
  it by more than twice the standard error of the three trackers' paired differences;
* **no regression** -- falls, the speed ratio (at least 0.9 at every command), stride length
  and period (not worse by twice their standard error), S0's walk (speed ratio at least
  0.9, fall excess at most 0.02).

The control re-run (the incumbent, linear, under this code) must reproduce the recorded
results; its largest difference is reported first.

    python transfer/scripts/w1_compare.py --output transfer/results/w1_resample.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
RESULTS = REPO_ROOT / "transfer" / "results"
PAIRS = [
    ("r1_3498_sym_v3", "r1_3498_sym_v3_gen"),
    ("warm_nao_r1_s32_sym", "warm_nao_r1_s32_sym_gen"),
    ("warm_nao_r1_s33_sym", "warm_nao_r1_s33_sym_gen"),
]
SKILLS = ("flat_dagger2", "rough_b4")
COMMANDS = ("forward", "fast", "turning")


def l4(tag: str, generator: str, command: str) -> dict | None:
    path = RESULTS / f"zero_shot_nao_{tag}_{generator}_auto_kin_l4_{command}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]


def features(tag: str) -> dict | None:
    rows = []
    for generator in SKILLS:
        for command in COMMANDS:
            run = l4(tag, generator, command)
            if run is None:
                return None
            dev = run["verdict"]["signature_deviation"]
            rows.append(
                {
                    "duty": (dev["duty_factor_0"] + dev["duty_factor_1"]) / 2,
                    "stride_length": dev["stride_length"],
                    "stride_period": dev["stride_period"],
                    "speed_ratio": run["verdict"]["speed_ratio"],
                    "falls": run["falls"],
                    "episodes": run["episodes"],
                    "passes": run["verdict"]["passes"],
                    "passes_unbiased_rate": run.get("verdict_at_body_rate_unbiased", {}).get(
                        "passes"
                    ),
                }
            )
    walk_path = RESULTS / f"zero_shot_nao_{tag}_walk_auto_kin.json"
    walk = None
    if walk_path.exists():
        w = json.loads(walk_path.read_text(encoding="utf-8"))["runs"]["encoder"]["verdict"]
        walk = {"speed_ratio": w["speed_ratio"], "fall_rate_excess": w["fall_rate_excess"]}
    return {
        "duty": float(np.mean([r["duty"] for r in rows])),
        "stride_length": float(np.mean([r["stride_length"] for r in rows])),
        "stride_period": float(np.mean([r["stride_period"] for r in rows])),
        "speed_ratio_min": float(min(r["speed_ratio"] for r in rows)),
        "falls": int(sum(r["falls"] for r in rows)),
        "episodes": int(sum(r["episodes"] for r in rows)),
        "passes": int(sum(r["passes"] for r in rows)),
        "walk": walk,
    }


def reproduction() -> dict:
    """The control re-run against the recorded incumbent: largest signature difference."""
    worst = 0.0
    found = 0
    for generator in SKILLS:
        for command in COMMANDS:
            old, new = (
                l4("r1_3498_sym_v3", generator, command),
                l4("r1_3498_sym_v3_linrep", generator, command),
            )
            if old is None or new is None:
                continue
            found += 1
            for key in ("stride_length", "stride_period"):
                worst = max(worst, abs(old["signature"][key] - new["signature"][key]))
            pairs = zip(
                old["signature"]["duty_factor"], new["signature"]["duty_factor"], strict=True
            )
            for a, b in pairs:
                worst = max(worst, abs(a - b))
    return {"commands": found, "largest_signature_difference": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = []
    for old, new in PAIRS:
        a, b = features(old), features(new)
        if a is None or b is None:
            print(f"{old} / {new}: results missing")
            continue
        pairs.append({"linear": old, "generator": new, "before": a, "after": b})
    report = {"reproduction": reproduction(), "pairs": pairs}
    if len(pairs) >= 2:
        diff = {
            k: np.array([p["after"][k] - p["before"][k] for p in pairs])
            for k in ("duty", "stride_length", "stride_period")
        }
        se = {k: float(v.std(ddof=1) / np.sqrt(len(v))) for k, v in diff.items()}
        mean = {k: float(v.mean()) for k, v in diff.items()}
        target = mean["duty"] < -2 * se["duty"]
        regressions = {
            "speed": all(p["after"]["speed_ratio_min"] >= 0.9 for p in pairs),
            "falls": sum(p["after"]["falls"] for p in pairs)
            <= sum(p["before"]["falls"] for p in pairs) + 2,
            "stride_length": mean["stride_length"] <= 2 * se["stride_length"],
            "stride_period": mean["stride_period"] <= 2 * se["stride_period"],
            "walk": all(
                p["after"]["walk"] is not None
                and p["after"]["walk"]["speed_ratio"] >= 0.9
                and p["after"]["walk"]["fall_rate_excess"] <= 0.02
                for p in pairs
            ),
        }
        report["difference"] = {"mean": mean, "se": se, "trackers": len(pairs)}
        report["target_met"] = bool(target)
        report["no_regression"] = regressions
        report["passes"] = bool(target and all(regressions.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=1))
    for p in pairs:
        print(
            f"{p['linear']:<22} duty {p['before']['duty']:.3f} -> {p['after']['duty']:.3f}  "
            f"stride {p['before']['stride_length']:.3f} -> {p['after']['stride_length']:.3f}  "
            f"period {p['before']['stride_period']:.3f} -> {p['after']['stride_period']:.3f}  "
            f"passes {p['before']['passes']} -> {p['after']['passes']}  "
            f"falls {p['before']['falls']} -> {p['after']['falls']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
