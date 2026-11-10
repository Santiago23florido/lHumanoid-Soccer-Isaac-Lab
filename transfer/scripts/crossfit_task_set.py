"""W19's task set (``skill_space.md`` §10.20): W13's base tasks, each also drawn by a generator
that never saw it.

Every task in ``transfer/results/base_tasks.json`` (flat and jog; the walk is the decoder's own
task 0) gets a second entry, ``<task>_xf``. It has the same ranges and starts, and its plans
come from ``logs/contract/mt19_xf_<task>``, trained on every base expert except that task's and
sidestep's. A decoder then practises the plans a generator composes for commands it has not
seen, on tasks it may train on.

The file is refused when an ``_xf`` generator's own report lists its task's capture or any
held-out capture among its training data.

    python transfer/scripts/crossfit_task_set.py --output transfer/results/base_tasks_xf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1].parent
RESULTS = REPO_ROOT / "transfer" / "results"
HELD_OUT_CAPTURES = ("base_sidestep_dart.npz", "rough")


def crossfit(base: dict) -> dict:
    tasks = list(base["tasks"])
    for task in base["tasks"]:
        name = f"mt19_xf_{task['name']}"
        report = RESULTS / f"contract_{name}.json"
        if not report.exists():
            raise FileNotFoundError(f"{report} is missing: train {name} first")
        captures = json.loads(report.read_text(encoding="utf-8"))["data"]["training_captures"]
        leaked = [
            c for c in captures if task["dart"] in c or any(h in c for h in HELD_OUT_CAPTURES)
        ]
        if leaked:
            raise ValueError(f"{name} trained on {leaked}")
        tasks.append(
            {
                **task,
                "name": f"{task['name']}_xf",
                "contract": f"logs/contract/{name}/generator.pt",
                "crossfit_training_captures": captures,
            }
        )
    return {
        **base,
        "tasks": tasks,
        "crossfit": "each base task also drawn by a generator trained without it and sidestep",
        "excluded_everywhere": ["rough", "sidestep"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, default=RESULTS / "base_tasks.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = crossfit(json.loads(args.base.read_text(encoding="utf-8")))
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for task in report["tasks"]:
        print(task["name"], task["contract"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
