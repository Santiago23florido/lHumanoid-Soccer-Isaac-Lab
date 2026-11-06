"""W8's pre-registered reading (skill_space.md §10.10): the plans' loaded feet put on the ground.

For each body's three frozen trackers, the captures of the plans as drawn against the same
tracker with ``HUMANOID_PLAN_CONTACT=ground`` (``w8_plan_contact.sh``), deployed path, L4
of both skills at three commands:

* **G1 target** -- L4's largest gate C deviation on rough (the paper's reading: per
  command, the largest of the stride, period and duty deviations; mean over the three
  commands) falls by more than twice the standard error of the three paired seed
  differences, and flat's rises by no more than that;
* **NAO target** -- W1's: the duty factors' relative deviation from the teacher's own grid,
  mean over both skills and three commands, falls by more than twice its standard error;
* **no regression** -- falls (summed per body, at most two more), the speed ratio (at least
  0.9 at every command), S0's walk (speed ratio at least 0.9, fall excess at most 0.02).

Contact metrics (``common/contact.py``) are reported beside them.

W8n (``w8_sole.sh``) reads the NAO's half on W6's trackers, whose sole point rests at the
contact. With one tracker there is no standard error, and the paired differences are
reported as a screen (``single``).

    python transfer/scripts/w8_compare.py --output transfer/results/w8_plan_contact.json
    python transfer/scripts/w8_compare.py --bodies nao --output transfer/results/w8n_sole.json
        --nao-trackers warm_nao_sole_s31_sym  # (one command line)
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

TRACKERS = {
    "g1": ("warm_r3_logload_sym", "warm_r3_s42_log_sym", "warm_r3_s43_log_sym"),
    "nao": ("r1_3498_sym_v3", "warm_nao_r1_s32_sym", "warm_nao_r1_s33_sym"),
}
SKILLS = ("flat_dagger2", "rough_b4")
COMMANDS = ("forward", "fast", "turning")
SIGNATURE = ("stride_length", "stride_period", "duty_factor_0", "duty_factor_1")
CONTACT = ("airborne_when_light", "liftoff_lag_mean", "touchdown_lag_mean", "duty_body")


def condition(body: str, tag: str, skill: str, command: str) -> dict | None:
    path = RESULTS / f"zero_shot_{body}_{tag}_{skill}_auto_kin_l4_{command}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
    dev = run["verdict"]["signature_deviation"]
    contact = score(path)["runs"]["encoder"]
    return {
        "largest": max(dev[k] for k in SIGNATURE),
        "duty": (dev["duty_factor_0"] + dev["duty_factor_1"]) / 2,
        **{k: float(np.mean(contact[k])) for k in CONTACT},
        "speed_ratio": run["verdict"]["speed_ratio"],
        "falls": run["falls"],
        "passes": run["verdict"]["passes"],
    }


def features(body: str, tag: str) -> dict | None:
    out = {}
    for skill in SKILLS:
        rows = [condition(body, tag, skill, c) for c in COMMANDS]
        if any(r is None for r in rows):
            return None
        out[skill] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[skill]["speed_ratio_min"] = float(min(r["speed_ratio"] for r in rows))
        out[skill]["falls"] = int(sum(r["falls"] for r in rows))
        out[skill]["passes"] = int(sum(r["passes"] for r in rows))
    walk_path = RESULTS / f"zero_shot_{body}_{tag}_walk_auto_kin.json"
    out["walk"] = None
    if walk_path.exists():
        w = json.loads(walk_path.read_text(encoding="utf-8"))["runs"]["encoder"]["verdict"]
        out["walk"] = {"speed_ratio": w["speed_ratio"], "fall_rate_excess": w["fall_rate_excess"]}
    return out


def paired(pairs: list[dict], skill: str, key: str) -> dict:
    diff = np.array([p["after"][skill][key] - p["before"][skill][key] for p in pairs])
    return {"mean": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bodies", nargs="+", choices=tuple(TRACKERS), default=list(TRACKERS))
    parser.add_argument("--nao-trackers", nargs="+", default=list(TRACKERS["nao"]))
    args = parser.parse_args()
    trackers = {**TRACKERS, "nao": tuple(args.nao_trackers)}
    report: dict = {}
    for body in args.bodies:
        tags = trackers[body]
        pairs = []
        for tag in tags:
            a, b = features(body, tag), features(body, f"{tag}_ground")
            if a is None or b is None:
                print(f"{body} {tag}: results missing")
                continue
            pairs.append({"as_drawn": tag, "before": a, "after": b})
        entry: dict = {"pairs": pairs}
        if len(pairs) == 1:
            entry["single"] = {
                skill: {
                    k: pairs[0]["after"][skill][k] - pairs[0]["before"][skill][k]
                    for k in ("largest", "duty", *CONTACT)
                }
                for skill in SKILLS
            }
        if len(pairs) >= 2:
            diff = {
                skill: {k: paired(pairs, skill, k) for k in ("largest", "duty", *CONTACT)}
                for skill in SKILLS
            }
            for p in pairs:
                for when in ("before", "after"):
                    p[when]["both"] = {
                        "duty": (p[when]["flat_dagger2"]["duty"] + p[when]["rough_b4"]["duty"]) / 2
                    }
            both = paired(pairs, "both", "duty")
            if body == "g1":
                rough, flat = diff["rough_b4"]["largest"], diff["flat_dagger2"]["largest"]
                target = rough["mean"] < -2 * rough["se"] and flat["mean"] <= 2 * flat["se"]
            else:
                target = both["mean"] < -2 * both["se"]
            regressions = {
                "falls": sum(p["after"][s]["falls"] for p in pairs for s in SKILLS)
                <= sum(p["before"][s]["falls"] for p in pairs for s in SKILLS) + 2,
                "speed": all(
                    p["after"][s]["speed_ratio_min"] >= 0.9 for p in pairs for s in SKILLS
                ),
                "walk": all(
                    p["after"]["walk"] is not None
                    and p["after"]["walk"]["speed_ratio"] >= 0.9
                    and p["after"]["walk"]["fall_rate_excess"] <= 0.02
                    for p in pairs
                ),
            }
            entry.update(
                difference=diff,
                duty_both_skills=both,
                target_met=bool(target),
                no_regression=regressions,
                passes=bool(target and all(regressions.values())),
            )
        report[body] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {b: {k: v for k, v in e.items() if k != "pairs"} for b, e in report.items()}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
