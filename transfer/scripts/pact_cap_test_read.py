"""Do the task plans' load spikes end the teachers' episodes? Capped against uncapped.

Each teacher's training-task capture (``pact_teacher_audit.sh``, stage ``train``) is run twice
with the same seed and conditions: as is, and with ``PACT_PLAN_MAX_LOAD`` capping the load a
generator's plan asks of one foot. Terminations are split by what the reference was doing:

* ``task``: a task episode (a generator's plan);
* ``biped_fallen``: a procedural episode whose biped fell in the 2 s before (it then holds
  still and the episode ends on the lag, by design);
* ``procedural``: any other procedural episode.

The registered read is the task-episode terminations summed over the bodies, capped against
uncapped, with a Poisson SE of the difference (``sqrt(a + b)``).

    python transfer/scripts/pact_cap_test_read.py --pair g1:contact --pair nao:contact \\
        --pair h1:contact --pair cassie:contact --cap 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TEACHERS = REPO / "logs/pact/teachers"
_LOAD = slice(22, 24)
WINDOW = 100
"""Control steps before a termination read for its reference (2 s at 50 Hz)."""


def classify(path: Path) -> dict:
    data = np.load(path)
    terminated, procedural = data["terminated"], data["procedural"]
    fallen = data["biped_fallen"]
    loads = data["contract_due"][..., _LOAD].max(-1)
    counts = {"task": 0, "biped_fallen": 0, "procedural": 0}
    window_load = []
    for step, env in zip(*np.nonzero(terminated)):
        recent = slice(max(0, step - WINDOW), step + 1)
        if not procedural[step, env]:
            counts["task"] += 1
            window_load.append(float(loads[recent, env].max()))
        elif fallen[recent, env].any():
            counts["biped_fallen"] += 1
        else:
            counts["procedural"] += 1
    task = ~procedural
    return {
        "terminations": int(terminated.sum()),
        **counts,
        "task_window_max_load": window_load,
        "task_plan_load_over_4": float((loads[task] > 4.0).mean()),
        "envs": int(terminated.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pair", action="append", required=True, help="body:variant")
    parser.add_argument("--cap", default="3")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_teacher_cap_test.json")
    args = parser.parse_args()

    report: dict = {"cap": float(args.cap), "window_steps": WINDOW, "bodies": {}}
    sums = {"uncapped": 0, "capped": 0}
    for pair in args.pair:
        body, variant = pair.split(":")
        base = TEACHERS / f"{body}_{variant}_train.npz"
        capped = TEACHERS / f"{body}_{variant}_cap{args.cap}_train.npz"
        if not (base.exists() and capped.exists()):
            print(f"{body} {variant}: missing {base.name if not base.exists() else capped.name}")
            continue
        entry = {"variant": variant, "uncapped": classify(base), "capped": classify(capped)}
        report["bodies"][body] = entry
        sums["uncapped"] += entry["uncapped"]["task"]
        sums["capped"] += entry["capped"]["task"]
        u, c = entry["uncapped"], entry["capped"]
        print(
            f"{body:6s} {variant:12s} task {u['task']} -> {c['task']} | biped fallen "
            f"{u['biped_fallen']} -> {c['biped_fallen']} | procedural {u['procedural']} -> "
            f"{c['procedural']} | plans > 4 W {u['task_plan_load_over_4']:.3f} -> "
            f"{c['task_plan_load_over_4']:.3f}"
        )
    difference = sums["uncapped"] - sums["capped"]
    se = math.sqrt(sums["uncapped"] + sums["capped"]) or 1.0
    report["task_terminations"] = {**sums, "difference": difference, "se": se, "z": difference / se}
    report["spikes_cause_falls"] = difference > 2 * se
    print(
        f"task-episode terminations: uncapped {sums['uncapped']}, capped {sums['capped']}, "
        f"difference {difference} ({difference / se:.1f} SE) -> "
        f"{'the spikes end episodes' if report['spikes_cause_falls'] else 'not shown'}"
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
