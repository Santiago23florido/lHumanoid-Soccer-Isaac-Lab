"""W19's reading: decoders that also practised
cross-fitted plans.

Arms, per seed, both W13's tracker continued 300 iterations in W13's conditions:

- ``xf``: with ``base_tasks_xf.json``, where flat and jog are also drawn by generators that never
  saw them (``mt19_xf_flat``, ``mt19_xf_jog``);
- ``ctl``: with W13's ``base_tasks.json``, the same continuation without them.

Both are deployed with their W13 tracker's refitted encoder. Sidestep and rough are out of
every gradient step.

Readings, per body:

1. **signal** (instrumentation): W13's G1 s41 tracker under the cross-fitted generators at flat's
   and jog's commands, beside its falls under the in-distribution ones (W16's captures);
2. **falls** on sidestep under ``mt18_no_sidestep``, summed over the seeds, ``xf`` against
   ``ctl``, and paired per seed;
3. **open objective** (W18's reading 3), per arm and held-out generator: every command's
   directed speed ratio at least 0.9, and falls at most two more than ``mt17_c``'s on W13's
   trackers of the same seeds (W18's runs);
4. **deviation**: sidestep's largest, ``xf - ctl`` paired, "help" below -2 SE, "hurt" above;
5. **no regression**: rough under ``rough_b4``, ``xf - ctl`` paired ("regresses" above +2 SE),
   and ``xf``'s falls at most two more than ``ctl``'s.

The screen (``--screen``) reads seeds 41 and 31 alone and prints ``continue`` when, on either
body, ``ctl`` falls more than twice on sidestep under ``mt18_no_sidestep`` and ``xf`` at most half
as often.

    python transfer/scripts/w19_compare.py --output transfer/results/w19_crossfit.json
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

from w8_compare import RESULTS  # noqa: E402
from w16_compare import FALLS_MORE, SPEED_MIN, paired, sign  # noqa: E402
from w16_compare import reading as rough_reading  # noqa: E402
from w18_compare import COMMANDS, sidestep_reading, teacher_directed  # noqa: E402

SEEDS = {"g1": (41, 42, 43), "nao": (31, 32, 33)}
SCREEN = {"g1": (41,), "nao": (31,)}
ARMS = ("xf", "ctl")
HELD_OUT = ("mt18_no_sidestep", "mt18_no_sidestep_m")
MAIN = "mt18_no_sidestep"
SIGNAL = {
    "flat": ("mt19_xf_flat", ("forward", "fast", "turning")),
    "jog": ("mt19_xf_jog", ("slow", "mid", "fast")),
}


def tag(arm: str, body: str, seed: int) -> str:
    return f"{arm}_{body}_s{seed}_sym"


def signal() -> dict:
    out = {}
    for task, (generator, commands) in SIGNAL.items():
        rows = {}
        for command in commands:
            name = f"zero_shot_g1_mt_g1_s41_sym_{generator}_own_auto_kin_l4_{command}.json"
            path = RESULTS / name
            seen = REPO_ROOT / f"g1/rollouts/w16_mt_g1_s41_sym_{task}_{command}.npz"
            if not path.exists() or not seen.exists():
                continue
            run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
            rows[command] = {
                "crossfit_falls": run["falls"],
                "crossfit_episodes": run["episodes"],
                "in_distribution_falls": int(np.load(seen)["terminated"].sum()),
            }
        out[task] = rows
    return out


def w18_bar(body: str, seeds: tuple[int, ...]) -> int | None:
    """``mt17_c``'s sidestep falls on W13's trackers of these seeds (W18)."""
    total = 0
    for seed in seeds:
        for command in COMMANDS:
            name = f"zero_shot_{body}_mt_{body}_s{seed}_sym_mt17_c_own_auto_kin_l4_{command}.json"
            path = RESULTS / name
            if not path.exists():
                return None
            total += json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]["falls"]
    return total


def rows_for(body: str, seeds: tuple[int, ...], teacher: dict) -> dict:
    return {
        seed: {
            arm: {g: sidestep_reading(body, tag(arm, body, seed), g, teacher) for g in HELD_OUT}
            for arm in ARMS
        }
        for seed in seeds
    }


def complete(rows: dict, generator: str) -> list[int]:
    return [s for s in rows if all(rows[s][a][generator] is not None for a in ARMS)]


def screen(teacher: dict) -> str:
    parts, go = [], False
    for body, seeds in SCREEN.items():
        rows = rows_for(body, seeds, teacher)
        if not complete(rows, MAIN):
            return "stop: incomplete"
        xf = sum(rows[s]["xf"][MAIN]["falls"] for s in seeds)
        ctl = sum(rows[s]["ctl"][MAIN]["falls"] for s in seeds)
        go = go or (ctl > FALLS_MORE and 2 * xf <= ctl)
        parts.append(f"{body} xf {xf} ctl {ctl}")
    return ("continue: " if go else "stop: ") + ", ".join(parts)


def body_reading(body: str, teacher: dict) -> dict:
    rows = rows_for(body, SEEDS[body], teacher)
    seeds = tuple(complete(rows, MAIN))
    bar = w18_bar(body, seeds) if seeds else None
    entry: dict = {"seeds": list(seeds), "bar_mt17_c_falls": bar, "rows": rows}
    if not seeds or bar is None:
        entry["verdict"] = "incomplete"
        return entry
    objective = {}
    for arm in ARMS:
        for g in HELD_OUT:
            if any(rows[s][arm][g] is None for s in seeds):
                continue
            falls = sum(rows[s][arm][g]["falls"] for s in seeds)
            speed = min(rows[s][arm][g]["directed_ratio_min"] for s in seeds)
            objective[f"{arm}/{g}"] = {
                "falls": falls,
                "directed_ratio_min": speed,
                "passes": bool(falls <= bar + FALLS_MORE and speed >= SPEED_MIN),
            }
    entry["open_objective"] = objective
    falls = [rows[s]["xf"][MAIN]["falls"] - rows[s]["ctl"][MAIN]["falls"] for s in seeds]
    deviation = {
        g: [rows[s]["xf"][g]["largest"] - rows[s]["ctl"][g]["largest"] for s in seeds]
        for g in HELD_OUT
        if all(rows[s]["xf"][g] is not None and rows[s]["ctl"][g] is not None for s in seeds)
    }
    if len(seeds) > 1:
        entry["falls_paired"] = paired(falls)
        entry["deviation_paired"] = {
            g: {**paired(v), "verdict": sign(paired(v))} for g, v in deviation.items()
        }
    else:
        entry["falls_paired"] = {"values": falls}
        entry["deviation_paired"] = {g: {"values": v} for g, v in deviation.items()}
    regression = {}
    rough = {s: {a: rough_reading(body, tag(a, body, s), "rough_b4") for a in ARMS} for s in seeds}
    if all(rough[s][a] is not None for s in seeds for a in ARMS):
        values = [rough[s]["xf"]["largest"] - rough[s]["ctl"]["largest"] for s in seeds]
        xf_falls = sum(rough[s]["xf"]["falls"] for s in seeds)
        ctl_falls = sum(rough[s]["ctl"]["falls"] for s in seeds)
        stat = paired(values) if len(seeds) > 1 else {"values": values}
        regresses = len(seeds) > 1 and stat["mean"] > 2 * stat["se"]
        regression = {
            "largest": stat,
            "falls": {"xf": xf_falls, "ctl": ctl_falls},
            "regresses": bool(regresses or xf_falls > ctl_falls + FALLS_MORE),
        }
    entry["rough_regression"] = regression or None
    if objective.get(f"xf/{MAIN}", {}).get("passes"):
        verdict = "fixes the falls"
    elif len(seeds) > 1 and entry["falls_paired"]["mean"] < -2 * entry["falls_paired"]["se"]:
        verdict = "reduces the falls"
    else:
        verdict = "no effect on the falls"
    entry["verdict"] = verdict
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    if args.screen:
        print(screen(teacher))
        return 0
    report = {"signal": signal(), "bodies": {b: body_reading(b, teacher) for b in SEEDS}}
    summary = []
    for body, entry in report["bodies"].items():
        part = f"{body}: {entry['verdict']}"
        dev = entry.get("deviation_paired", {}).get(MAIN, {})
        if "verdict" in dev:
            part += f", deviation {dev['verdict']}"
        if entry.get("rough_regression"):
            regresses = entry["rough_regression"]["regresses"]
            part += ", rough " + ("regresses" if regresses else "holds")
        summary.append(part)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2, default=str) + "\n").encode("utf-8"))
    print("; ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
