"""W20's reading: W13's decoder recipe with W19's
cross-fitted plans, trained once from S.

Arms, per seed, both S's incumbent continued 1000 iterations in W13's conditions:

- ``xs``: with ``base_tasks_xf.json`` (W19's task set);
- ``mt``: W13's trackers, with ``base_tasks.json``. They are the paired control, already on
  disk.

Each is deployed with its own refitted encoder. Sidestep and rough are out of every gradient step.

Readings, per body, over the seeds run:

1. **falls** on sidestep under ``mt18_no_sidestep``: ``xs`` against ``mt``, summed and paired
   per seed;
2. **open objective** (W18's reading 3), per arm and held-out generator: every command's
   directed speed ratio is at least 0.9, and falls are at most two more than ``mt17_c``'s on
   W13's trackers of the same seeds;
3. **deviation**: sidestep's largest, ``xs - mt`` paired. "Help" below -2 SE, "hurt" above
   +2 SE;
4. **no regression** on rough under ``rough_b4``: ``xs - mt`` paired ("regresses" above
   +2 SE), and ``xs``'s falls at most two more than ``mt``'s;
5. **no regression on the base tasks**:
   - flat's L4 (mean over its three commands) and jog's at its middle command, ``xs - mt``
     paired, "regresses" above +2 SE;
   - ``xs``'s falls at most two more than ``mt``'s;
   - S0's walk passes on every ``xs`` tracker.

   The G1 is read in the task's units, as W13 read it. The NAO is read in its own units, as
   W14 read it (``paper_tables.OBSERVED``).

Beside the readings, as instrumentation: W19's ``xf`` continuation of the same seeds, so the
length and the start can be told apart.

The screen (``--screen``) reads seeds 41 and 31 alone. It prints ``continue`` when, on either
body, ``mt`` falls more than twice on sidestep under ``mt18_no_sidestep`` and ``xs`` at most half
as often.

    python transfer/scripts/w20_compare.py --output transfer/results/w20_from_s.json
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

from w8_compare import COMMANDS as FLAT_COMMANDS  # noqa: E402
from w8_compare import condition  # noqa: E402
from w13_compare import walk  # noqa: E402
from w16_compare import FALLS_MORE, SPEED_MIN, paired, sign  # noqa: E402
from w16_compare import reading as rough_reading  # noqa: E402
from w18_compare import COMMANDS, sidestep_reading, teacher_directed  # noqa: E402
from w19_compare import HELD_OUT, MAIN, w18_bar  # noqa: E402

SEEDS = {"g1": (41, 42, 43), "nao": (31, 32, 33)}
SCREEN = {"g1": (41,), "nao": (31,)}
ARMS = ("xs", "mt")
OBSERVED = {"g1": "", "nao": "_own"}


def tag(arm: str, body: str, seed: int) -> str:
    return f"{arm}_{body}_s{seed}_sym"


def rows_for(body: str, seeds: tuple[int, ...], teacher: dict, arms=ARMS) -> dict:
    return {
        seed: {
            arm: {g: sidestep_reading(body, tag(arm, body, seed), g, teacher) for g in HELD_OUT}
            for arm in arms
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
        xs = sum(rows[s]["xs"][MAIN]["falls"] for s in seeds)
        mt = sum(rows[s]["mt"][MAIN]["falls"] for s in seeds)
        go = go or (mt > FALLS_MORE and 2 * xs <= mt)
        parts.append(f"{body} xs {xs} mt {mt}")
    return ("continue: " if go else "stop: ") + ", ".join(parts)


def base_tasks(body: str, tag_: str) -> dict | None:
    """Flat's L4 over its three commands and jog's at its middle one, on the body's path."""
    units = OBSERVED[body]
    flat = [condition(body, tag_, f"flat_dagger2{units}", c) for c in FLAT_COMMANDS]
    jog = condition(body, tag_, f"base_jog_b4{units}", "mid")
    if any(r is None for r in flat) or jog is None:
        return None
    return {
        "flat": float(np.mean([r["largest"] for r in flat])),
        "flat_falls": int(sum(r["falls"] for r in flat)),
        "jog": jog["largest"],
        "jog_falls": int(jog["falls"]),
        "walk": walk(body, tag_),
    }


def regression(values: list[float], falls_new: int, falls_old: int) -> dict:
    stat = paired(values) if len(values) > 1 else {"values": values}
    worse = len(values) > 1 and stat["mean"] > 2 * stat["se"]
    return {
        "largest": stat,
        "falls": {"xs": falls_new, "mt": falls_old},
        "regresses": bool(worse or falls_new > falls_old + FALLS_MORE),
    }


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
    falls = [rows[s]["xs"][MAIN]["falls"] - rows[s]["mt"][MAIN]["falls"] for s in seeds]
    deviation = {
        g: [rows[s]["xs"][g]["largest"] - rows[s]["mt"][g]["largest"] for s in seeds]
        for g in HELD_OUT
        if all(rows[s]["xs"][g] is not None and rows[s]["mt"][g] is not None for s in seeds)
    }
    if len(seeds) > 1:
        entry["falls_paired"] = paired(falls)
        entry["deviation_paired"] = {
            g: {**paired(v), "verdict": sign(paired(v))} for g, v in deviation.items()
        }
    else:
        entry["falls_paired"] = {"values": falls}
        entry["deviation_paired"] = {g: {"values": v} for g, v in deviation.items()}
    rough = {s: {a: rough_reading(body, tag(a, body, s), "rough_b4") for a in ARMS} for s in seeds}
    entry["rough_regression"] = None
    if all(rough[s][a] is not None for s in seeds for a in ARMS):
        entry["rough_regression"] = regression(
            [rough[s]["xs"]["largest"] - rough[s]["mt"]["largest"] for s in seeds],
            sum(rough[s]["xs"]["falls"] for s in seeds),
            sum(rough[s]["mt"]["falls"] for s in seeds),
        )
    base = {s: {a: base_tasks(body, tag(a, body, s)) for a in ARMS} for s in seeds}
    entry["base_tasks"] = None
    if all(base[s][a] is not None for s in seeds for a in ARMS):
        walks = [base[s]["xs"]["walk"] for s in seeds]
        entry["base_tasks"] = {
            task: regression(
                [base[s]["xs"][task] - base[s]["mt"][task] for s in seeds],
                sum(base[s]["xs"][f"{task}_falls"] for s in seeds),
                sum(base[s]["mt"][f"{task}_falls"] for s in seeds),
            )
            for task in ("flat", "jog")
        }
        entry["base_tasks"]["walk_passes"] = bool(all(w is not None and w["passes"] for w in walks))
        entry["base_tasks"]["rows"] = base
    # Instrumentation: W19's continuation of the same seeds.
    continuation = rows_for(body, seeds, teacher, arms=("xf",))
    entry["w19_xf_falls"] = {
        s: continuation[s]["xf"][MAIN]["falls"] if continuation[s]["xf"][MAIN] else None
        for s in seeds
    }
    if objective.get(f"xs/{MAIN}", {}).get("passes"):
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
    report = {"bodies": {b: body_reading(b, teacher) for b in SEEDS}}
    summary = []
    for body, entry in report["bodies"].items():
        part = f"{body}: {entry['verdict']}"
        dev = entry.get("deviation_paired", {}).get(MAIN, {})
        if "verdict" in dev:
            part += f", deviation {dev['verdict']}"
        if entry.get("rough_regression"):
            part += ", rough " + (
                "regresses" if entry["rough_regression"]["regresses"] else "holds"
            )
        base = entry.get("base_tasks")
        if base:
            held = (
                not (base["flat"]["regresses"] or base["jog"]["regresses"]) and base["walk_passes"]
            )
            part += ", base tasks " + ("hold" if held else "regress")
        summary.append(part)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2, default=str) + "\n").encode("utf-8"))
    print("; ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
