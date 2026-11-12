"""W7's reading: faster NAO leg drives, frozen trackers.

For each frozen NAO tracker, its linear captures under the derived gains (the incumbent's
W1 control re-run; S's seeds' records) against the same tracker with
``HUMANOID_NAO_LEG_GAIN=4`` (``w7_drives.sh``), on the deployed path, over both skills and
three commands:

* **lift** -- the clearance a body reaches over the clearance its plans ask
  (``common/contact.py``: the median over swings of each, per foot), mean over feet and
  conditions; it must rise by more than twice the standard error of the three trackers'
  paired differences;
* **duty** -- W1's target, the duty factors' relative deviation from the teacher's own
  grid; it must fall by more than twice its standard error;
* **no regression** -- falls (summed, at most two more), the speed ratio (at least 0.9 at
  every command), S0's walk (speed ratio at least 0.9, fall excess at most 0.02).

The reading is ``both`` (the drive's lag is a cause of both gaps and the lift sets the
duty), ``lift only``, ``duty only`` or ``neither`` (a frozen tracker under a faster leg is
not informative either way). The incumbent at gain 2 is reported beside it as a dose.

W7b (``w7b_drives.sh``) reads the same criteria at gain 2 over the same three trackers.

    python transfer/scripts/w7_compare.py --output transfer/results/w7_drives.json
    python transfer/scripts/w7_compare.py --gain 2 --output transfer/results/w7b_drives.json
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

from contact_fidelity import score  # noqa: E402

BASES = (
    ("r1_3498_sym_v3_linrep", "r1_3498_sym_v3"),
    ("warm_nao_r1_s32_sym", "warm_nao_r1_s32_sym"),
    ("warm_nao_r1_s33_sym", "warm_nao_r1_s33_sym"),
)
"""Each tracker's capture under the derived gains, and the stem its faster captures carry."""


def pairs_at(gain: int) -> list[tuple[str, str]]:
    return [(old, f"{stem}_g{gain}") for old, stem in BASES]


DOSE = ("r1_3498_sym_v3_linrep", "r1_3498_sym_v3_g2", "r1_3498_sym_v3_g4")
SKILLS = ("flat_dagger2", "rough_b4")
COMMANDS = ("forward", "fast", "turning")


def condition(tag: str, skill: str, command: str) -> dict | None:
    path = RESULTS / f"zero_shot_nao_{tag}_{skill}_auto_kin_l4_{command}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
    contact = score(path)["runs"]["encoder"]
    dev = run["verdict"]["signature_deviation"]
    lift = [
        b / a for b, a in zip(contact["clearance_body"], contact["clearance_asked"], strict=True)
    ]
    return {
        "lift": float(np.mean(lift)),
        "duty": (dev["duty_factor_0"] + dev["duty_factor_1"]) / 2,
        "duty_body": float(np.mean(contact["duty_body"])),
        "touchdown_lag": float(np.mean(contact["touchdown_lag_mean"])),
        "graze": float(np.mean(contact["graze_in_swing"])),
        "speed_ratio": run["verdict"]["speed_ratio"],
        "falls": run["falls"],
        "episodes": run["episodes"],
        "passes": run["verdict"]["passes"],
    }


def features(tag: str, skills=SKILLS) -> dict | None:
    rows = []
    for skill in skills:
        for command in COMMANDS:
            row = condition(tag, skill, command)
            if row is None:
                return None
            rows.append(row)
    walk_path = RESULTS / f"zero_shot_nao_{tag}_walk_auto_kin.json"
    walk = None
    if walk_path.exists():
        w = json.loads(walk_path.read_text(encoding="utf-8"))["runs"]["encoder"]["verdict"]
        walk = {"speed_ratio": w["speed_ratio"], "fall_rate_excess": w["fall_rate_excess"]}
    mean = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("lift", "duty", "duty_body", "touchdown_lag", "graze")
    }
    return {
        **mean,
        "speed_ratio_min": float(min(r["speed_ratio"] for r in rows)),
        "falls": int(sum(r["falls"] for r in rows)),
        "episodes": int(sum(r["episodes"] for r in rows)),
        "passes": int(sum(r["passes"] for r in rows)),
        "walk": walk,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gain", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    pairs = []
    for old, new in pairs_at(args.gain):
        a, b = features(old), features(new)
        if a is None or b is None:
            print(f"{old} / {new}: results missing")
            continue
        pairs.append({"derived": old, "faster": new, "before": a, "after": b})
    report: dict = {"gain": float(args.gain), "pairs": pairs}
    keys = ("lift", "duty", "duty_body", "touchdown_lag", "graze")
    if len(pairs) >= 2:
        diff = {k: np.array([p["after"][k] - p["before"][k] for p in pairs]) for k in keys}
        mean = {k: float(v.mean()) for k, v in diff.items()}
        se = {k: float(v.std(ddof=1) / np.sqrt(len(v))) for k, v in diff.items()}
        lift = mean["lift"] > 2 * se["lift"]
        duty = mean["duty"] < -2 * se["duty"]
        regressions = {
            "falls": sum(p["after"]["falls"] for p in pairs)
            <= sum(p["before"]["falls"] for p in pairs) + 2,
            "speed": all(p["after"]["speed_ratio_min"] >= 0.9 for p in pairs),
            "walk": all(
                p["after"]["walk"] is not None
                and p["after"]["walk"]["speed_ratio"] >= 0.9
                and p["after"]["walk"]["fall_rate_excess"] <= 0.02
                for p in pairs
            ),
        }
        reading = {
            (True, True): "both",
            (True, False): "lift only",
            (False, True): "duty only",
            (False, False): "neither",
        }[(bool(lift), bool(duty))]
        report.update(
            difference={"mean": mean, "se": se, "trackers": len(pairs)},
            lift_rises=bool(lift),
            duty_falls=bool(duty),
            no_regression=regressions,
            reading=reading,
            opens_rl_arm=bool(lift and duty and all(regressions.values())),
        )
    dose = {}
    for tag in DOSE:
        f = features(tag)
        if f is not None:
            dose[tag] = {k: f[k] for k in (*keys, "falls", "speed_ratio_min")}
    report["dose_incumbent"] = dose
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
