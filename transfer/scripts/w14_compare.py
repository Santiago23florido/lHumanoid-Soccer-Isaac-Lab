"""W14's pre-registered reading (``skill_space.md`` §10.15): the NAO's trackers in their own units.

Each NAO tracker's L4 with every ``z*`` it observes mapped into the units of the walk contract
it trained under (``_own``), against the same tracker's L4 in the task's units, paired by
tracker, as mean ± SE over three:

1. **published** -- on S's trackers, per skill (``flat_dagger2``, ``rough_b4``): the change in
   the largest gate C deviation and in the duty deviation, each the mean over the three
   commands. A change beyond 2 SE in either corrects that skill's published NAO figures.
2. **regression** -- only if W13's NAO flat regression holds in the task's units (W13 - S,
   paired, flat's largest deviation above 2 SE): ``units`` if, in their own units, that
   difference is no longer above 2 SE and W13's body duty on flat falls by more than 2 SE;
   ``learned`` if it stays above 2 SE; ``neither`` otherwise.
3. **jog** -- W13's NAO "executes" on jog, re-read in their own units with W13's rule (at most
   one fall; the CoM's speed along the commanded axis over the expert's, 0.8-1.2).

    python transfer/scripts/w14_compare.py --output transfer/results/w14_units.json
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

from w8_compare import COMMANDS, condition  # noqa: E402
from w13_compare import MAX_FALLS, SEEDS, SPEED, S, axis_speed, paired  # noqa: E402

S_NAO = S["nao"]
W13_NAO = tuple(f"mt_nao_s{seed}_sym" for seed in SEEDS["nao"])
SKILLS = ("flat_dagger2", "rough_b4")
JOG = "base_jog_b4"


def skill_of(generator: str, own: bool) -> str:
    return f"{generator}_own" if own else generator


def over_commands(tag: str, generator: str, own: bool) -> dict | None:
    """L4 at the three commands: the largest deviation, the duty deviation, the body's duty."""
    rows = [condition("nao", tag, skill_of(generator, own), c) for c in COMMANDS]
    if any(r is None for r in rows):
        return None
    return {
        **{k: float(np.mean([r[k] for r in rows])) for k in ("largest", "duty", "duty_body")},
        "falls": int(sum(r["falls"] for r in rows)),
        "speed_ratio_min": float(min(r["speed_ratio"] for r in rows)),
    }


def jog(tag: str, own: bool) -> dict | None:
    """Jog's L4 at its middle command, with W13's speed along the commanded axis."""
    skill = skill_of(JOG, own)
    row = condition("nao", tag, skill, "mid")
    if row is None:
        return None
    path = RESULTS / f"zero_shot_nao_{tag}_{skill}_auto_kin_l4_mid.json"
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    run, expert = axis_speed(runs["encoder"]["file"], 0), axis_speed(runs["teacher"]["file"], 0)
    row["axis_speed_ratio"] = 0.0 if run is None or not expert else run / expert
    return row


def beyond(stat: dict) -> bool:
    return abs(stat["mean"]) > 2 * stat["se"]


def published() -> dict:
    out = {}
    for generator in SKILLS:
        rows = [
            (over_commands(t, generator, False), over_commands(t, generator, True)) for t in S_NAO
        ]
        if any(a is None or b is None for a, b in rows):
            out[generator] = None
            continue
        largest = paired([b["largest"] - a["largest"] for a, b in rows])
        duty = paired([b["duty"] - a["duty"] for a, b in rows])
        out[generator] = {
            "rows": [
                {"tag": t, "task_units": a, "own_units": b}
                for t, (a, b) in zip(S_NAO, rows, strict=True)
            ],
            "largest": largest,
            "duty": duty,
            "corrected": bool(beyond(largest) or beyond(duty)),
        }
    return out


def regression() -> dict | None:
    task = [
        (over_commands(s, "flat_dagger2", False), over_commands(m, "flat_dagger2", False))
        for s, m in zip(S_NAO, W13_NAO, strict=True)
    ]
    own = [
        (over_commands(s, "flat_dagger2", True), over_commands(m, "flat_dagger2", True))
        for s, m in zip(S_NAO, W13_NAO, strict=True)
    ]
    if any(x is None for pair in task + own for x in pair):
        return None
    in_task = paired([m["largest"] - s["largest"] for s, m in task])
    in_own = paired([m["largest"] - s["largest"] for s, m in own])
    duty = paired([o[1]["duty_body"] - t[1]["duty_body"] for t, o in zip(task, own, strict=True)])
    if not in_task["mean"] > 2 * in_task["se"]:
        verdict = "not applicable"
    elif in_own["mean"] > 2 * in_own["se"]:
        verdict = "learned"
    elif duty["mean"] < -2 * duty["se"]:
        verdict = "units"
    else:
        verdict = "neither"
    return {
        "w13_minus_s_task_units": in_task,
        "w13_minus_s_own_units": in_own,
        "w13_body_duty_own_minus_task": duty,
        "verdict": verdict,
        "rows": [
            {
                "s": s,
                "w13": m,
                "task_units": {"s": t[0], "w13": t[1]},
                "own_units": {"s": o[0], "w13": o[1]},
            }
            for s, m, t, o in zip(S_NAO, W13_NAO, task, own, strict=True)
        ],
    }


def jog_reading() -> dict | None:
    rows = {
        tag: {"task_units": jog(tag, False), "own_units": jog(tag, True)} for tag in S_NAO + W13_NAO
    }
    if any(r is None for row in rows.values() for r in row.values()):
        return None
    executes = all(
        rows[m]["own_units"]["falls"] <= MAX_FALLS
        and SPEED[0] <= rows[m]["own_units"]["axis_speed_ratio"] <= SPEED[1]
        for m in W13_NAO
    )
    target = paired(
        [
            rows[m]["own_units"]["largest"] - rows[s]["own_units"]["largest"]
            for s, m in zip(S_NAO, W13_NAO, strict=True)
        ]
    )
    return {
        "rows": rows,
        "executes_own_units": bool(executes),
        "w13_minus_s_largest_own_units": target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"published": published(), "regression": regression(), "jog": jog_reading()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    parts = []
    for generator, entry in report["published"].items():
        if entry is None:
            parts.append(f"{generator} incomplete")
        else:
            parts.append(f"{generator} {'corrected' if entry['corrected'] else 'unchanged'}")
    reg, jogs = report["regression"], report["jog"]
    parts.append(f"regression {'incomplete' if reg is None else reg['verdict']}")
    parts.append(f"jog executes {'incomplete' if jogs is None else jogs['executes_own_units']}")
    print("; ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
