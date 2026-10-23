"""The warm study's arms side by side, from their ladders' results (skill_space.md §9.1).

For each arm tag and each published task, over L4's three commands, on the deployed path
(the kinematic estimator, ``encoder`` run): falls over episodes, how many commands pass
the teacher's signature, and the speed ratio; then the walk (L1b) on the same path, and
the competence map and its corners.

Usage::

    python transfer/scripts/warm_table.py warm_r3_r8 warm_v2t_r8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "transfer" / "results"
TASKS = {"rough": "rough_b4", "flat": "flat_dagger2"}
COMMANDS = ("forward", "fast", "turning")


def ladder(tag: str, task: str) -> dict:
    falls = episodes = passes = 0
    ratios = []
    for command in COMMANDS:
        path = RESULTS / f"zero_shot_g1_{tag}_{TASKS[task]}_auto_kin_l4_{command}.json"
        run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
        falls += run["falls"]
        episodes += run["episodes"]
        passes += bool(run["verdict"]["passes"])
        ratios.append(run["verdict"]["speed_ratio"])
    return {"falls": falls, "episodes": episodes, "passes": passes, "speed_ratio": ratios}


def walk(tag: str) -> dict:
    run = json.loads((RESULTS / f"zero_shot_g1_{tag}_walk_kin.json").read_text(encoding="utf-8"))
    encoder = run["runs"]["encoder"]
    return {"falls": encoder["falls"], "episodes": encoder["episodes"],
            "speed_ratio": encoder["verdict"]["speed_ratio"],
            "passes": bool(encoder["verdict"]["passes"])}  # fmt: skip


def competence(tag: str, suffix: str, group: str) -> dict:
    """One group of the competence report's probes: how many pass, of how many."""
    path = RESULTS / f"competence_g1_{tag}{suffix}.json"
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"][group]
    return {"passed": summary["passed"], "probes": summary["probes"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--results", type=Path, action="append", default=[],
        help="Another results directory to read tags from (a worktree's); this repository's is "
        "always read first.",
    )  # fmt: skip
    args = parser.parse_args()
    global RESULTS
    directories = [RESULTS, *args.results]
    table = {}
    for tag in args.tags:
        RESULTS = next(d for d in directories if any(d.glob(f"zero_shot_g1_{tag}_*")))
        table[tag] = {
            **{task: ladder(tag, task) for task in TASKS},
            "walk": walk(tag),
            "map": competence(tag, "", "grid"),
            "corners": competence(tag, "_corners", "corner"),
        }
    for tag, row in table.items():
        cells = [f"{tag:14s}"]
        for task in TASKS:
            r = row[task]
            ratios = "/".join(f"{x:.2f}" for x in r["speed_ratio"])
            cells.append(f"{task} {r['falls']}/{r['episodes']} passes {r['passes']} ({ratios})")
        w = row["walk"]
        cells.append(f"walk {w['falls']}/{w['episodes']} ratio {w['speed_ratio']:.2f}")
        cells.append(f"map {row['map']['passed']}/{row['map']['probes']}")
        cells.append(f"corners {row['corners']['passed']}/{row['corners']['probes']}")
        print(" | ".join(cells))
    if args.output is not None:
        args.output.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
