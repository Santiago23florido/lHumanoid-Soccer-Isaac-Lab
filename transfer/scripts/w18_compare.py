"""W18's pre-registered reading (``skill_space.md`` §10.19): leave one task out where no other
task defines its commands.

Sidestep is held out of every gradient step. Arms, each a generator (``logs/contract/<arm>``)
deployed on W13's six trackers (G1 s41-43, NAO s31-33), sidestep's L4 at ``left``,
``left_fast`` and ``right`` on the deployed path, in each tracker's own units:

- ``mt18_no_sidestep``: the walk, flat, backward and jog experts, commands relabelled (W17);
- ``mt18_no_sidestep_m``: from it, plus the bodies' histories with the teacher's futures;
- ``mt17_c`` and ``mt17_m``: the same recipes with sidestep's expert, the references.

Readings, per body, paired by tracker (mean ± SE over three), on sidestep's largest gate C
deviation (the mean over its three commands):

1. **proxy calibration** -- from the state after a reset (``held_out_chains_w18_reset_<body>``):
   the chain orders ``mt16_c``, ``mt17_c``, ``mt17_m`` and ``rough_b4`` as their rough L4 does;
   on the G1 also, the chain's duty factor is within 0.05 of L4's asked one in at least 10 of
   the 12 arm x command cells;
2. **cost** -- ``mt18_no_sidestep - mt17_c``: "costs" above +2 SE, else "no detectable cost";
3. **open objective**, per arm: every command's directed speed ratio (the CoM velocity along the
   commanded direction, body over teacher, after 100 settle steps) at least 0.9, and at most two
   more falls than ``mt17_c`` over the three trackers;
4. **bodies** -- ``mt18_no_sidestep_m - mt18_no_sidestep``: "help" below -2 SE, "hurt" above
   +2 SE;
5. **offline**, per base task with teacher captures: the leave-one-out generator's mean chain
   deviation minus ``mt17_c``'s on that task, from the teacher's history and from the reset.

    python transfer/scripts/w18_compare.py --output transfer/results/w18_loto.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from w8_compare import RESULTS  # noqa: E402
from w16_compare import COMMANDS as ROUGH  # noqa: E402
from w16_compare import (  # noqa: E402
    FALLS_MORE,
    SPEED_MIN,
    TRACKERS,
    difference,
    generator_report,
    sign,
)
from w16_compare import reading as rough_reading  # noqa: E402

ARMS = ("mt18_no_sidestep", "mt18_no_sidestep_m", "mt17_c", "mt17_m")
GENERATORS = (
    "mt18_no_sidestep",
    "mt18_no_sidestep_m",
    "mt18_no_walk",
    "mt18_no_flat",
    "mt18_no_backward",
    "mt18_no_jog",
)
COMMANDS = ("left", "left_fast", "right")
CALIBRATION = ("mt16_c", "mt17_c", "mt17_m", "rough_b4")
DUTY_TOLERANCE = 0.05
DUTY_CELLS = 10
OFFLINE = ("flat", "backward", "sidestep", "jog")
SETTLE = 100


def directed(path: Path) -> float:
    """The mean CoM velocity along the commanded direction, in ``V``, after the settle steps."""
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    command = data["command"][SETTLE:, :, :2] / units.velocity
    norm = np.linalg.norm(command, axis=-1, keepdims=True)
    direction = command / np.where(norm > 0, norm, 1.0)
    velocity = data["skill_state"][SETTLE:, :, 1:3]
    return float((velocity * direction).sum(-1).mean())


def teacher_directed(command: str) -> float:
    return directed(REPO_ROOT / f"g1/rollouts/base_sidestep_play_{command}.npz")


def sidestep_reading(body: str, tag: str, arm: str, teacher: dict) -> dict | None:
    rows = []
    for command in COMMANDS:
        path = RESULTS / f"zero_shot_{body}_{tag}_{arm}_own_auto_kin_l4_{command}.json"
        capture = (
            REPO_ROOT / body / "rollouts" / f"zs_{tag}_{arm}_own_auto_kin_{command}_encoder.npz"
        )
        if not path.exists() or not capture.exists():
            return None
        run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
        rows.append(
            {
                "largest": max(run["verdict"]["signature_deviation"].values()),
                "falls": run["falls"],
                "directed_ratio": directed(capture) / teacher[command],
                "asked_duty": float(np.mean(run["asked"]["signature"]["duty_factor"])),
                "body_duty": float(np.mean(run["signature"]["duty_factor"])),
            }
        )
    return {
        "largest": float(np.mean([r["largest"] for r in rows])),
        "falls": int(sum(r["falls"] for r in rows)),
        "directed_ratio_min": float(min(r["directed_ratio"] for r in rows)),
        "per_command": dict(zip(COMMANDS, rows, strict=True)),
    }


def calibration(body: str) -> dict | None:
    path = RESULTS / f"held_out_chains_w18_reset_{body}.json"
    if not path.exists():
        return None
    chains = json.loads(path.read_text(encoding="utf-8"))["generators"]
    l4: dict = {}
    for arm in CALIBRATION:
        rows = [rough_reading(body, tag, arm) for tag in TRACKERS[body]]
        if any(r is None for r in rows):
            return None
        l4[arm] = float(np.mean([r["largest"] for r in rows]))
    chain = {arm: chains[arm]["largest_mean"] for arm in CALIBRATION}
    same_order = sorted(CALIBRATION, key=chain.get) == sorted(CALIBRATION, key=l4.get)
    cells = []
    for arm in CALIBRATION:
        for command in ROUGH:
            asked = []
            for tag in TRACKERS[body]:
                name = f"zero_shot_{body}_{tag}_{arm}_own_auto_kin_l4_{command}.json"
                run = json.loads((RESULTS / name).read_text(encoding="utf-8"))["runs"]["encoder"]
                asked.append(float(np.mean(run["asked"]["signature"]["duty_factor"])))
            sig = chains[arm][command]["chain"]
            duty = (sig["duty_factor_0"] + sig["duty_factor_1"]) / 2
            cells.append(
                {"arm": arm, "command": command, "chain": duty, "asked": float(np.mean(asked))}
            )
    within = sum(abs(c["chain"] - c["asked"]) <= DUTY_TOLERANCE for c in cells)
    passes = same_order and (body != "g1" or within >= DUTY_CELLS)
    return {
        "chain": chain,
        "l4": l4,
        "same_order": same_order,
        "duty_cells_within": int(within),
        "cells": cells,
        "calibrated": bool(passes),
    }


def offline(task: str) -> dict:
    out = {}
    for start in ("teacher", "reset"):
        path = RESULTS / f"held_out_chains_w18_{task}_{start}.json"
        if not path.exists():
            continue
        chains = json.loads(path.read_text(encoding="utf-8"))["generators"]
        held, seen = chains.get(f"mt18_no_{task}"), chains.get("mt17_c")
        if held is None or seen is None:
            continue
        out[start] = {
            "held_out": held["largest_mean"],
            "seen": seen["largest_mean"],
            "difference": held["largest_mean"] - seen["largest_mean"],
            "speed_held_out": {
                c: held[c]["speed_ratio"]
                for c in held
                if isinstance(held[c], dict) and "speed_ratio" in held[c]
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    report: dict = {
        "generators": {g: generator_report(g) for g in GENERATORS},
        "teacher_directed": teacher,
        "calibration": {body: calibration(body) for body in TRACKERS},
        "offline": {task: offline(task) for task in OFFLINE},
        "bodies": {},
    }
    summary = []
    for body, tags in TRACKERS.items():
        rows = {t: {arm: sidestep_reading(body, t, arm, teacher) for arm in ARMS} for t in tags}
        cost = difference(rows, "mt18_no_sidestep", "mt17_c")
        bodies = difference(rows, "mt18_no_sidestep_m", "mt18_no_sidestep")
        objective = {}
        for arm in ARMS:
            if any(rows[t][arm] is None or rows[t]["mt17_c"] is None for t in tags):
                continue
            falls = sum(rows[t][arm]["falls"] for t in tags)
            control = sum(rows[t]["mt17_c"]["falls"] for t in tags)
            speed = min(rows[t][arm]["directed_ratio_min"] for t in tags)
            objective[arm] = {
                "falls": falls,
                "falls_control": control,
                "directed_ratio_min": speed,
                "passes": bool(falls <= control + FALLS_MORE and speed >= SPEED_MIN),
            }
        entry = {
            "rows": rows,
            "cost": None
            if cost is None
            else {
                **cost,
                "verdict": "costs" if cost["mean"] > 2 * cost["se"] else "no detectable cost",
            },
            "bodies": None if bodies is None else {**bodies, "verdict": sign(bodies)},
            "open_objective": objective,
        }
        report["bodies"][body] = entry
        part = [f"{body}: " + (entry["cost"]["verdict"] if entry["cost"] else "incomplete")]
        if "mt18_no_sidestep" in objective:
            part.append(
                "objective " + ("passes" if objective["mt18_no_sidestep"]["passes"] else "fails")
            )
        if entry["bodies"]:
            part.append(f"bodies {entry['bodies']['verdict']}")
        cal = report["calibration"][body]
        if cal is not None:
            part.append("proxy " + ("calibrated" if cal["calibrated"] else "not calibrated"))
        summary.append(", ".join(part))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print("; ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
