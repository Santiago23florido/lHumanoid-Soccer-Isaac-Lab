"""W13's pre-registered reading (``skill_space.md`` §10.14): decoders trained on the base tasks.

Each body's W13 trackers (``mt_<body>_s<seed>_sym``) against S's (the same continuation of the
same incumbent without the tasks), paired by seed, on the deployed path:

* **target** -- over the new base tasks at their middle command, the largest gate C deviation
  falls by more than 2 SE of the paired differences;
* **executes** -- per new task, every W13 tracker at its middle command: at most one fall and
  a speed ratio between 0.8 and 1.2 (W11's expert criterion, without the duty term). The
  ratio is the CoM's dimensionless speed along the task's commanded axis over the expert's
  own in its play rollout at that command, both once settled as L4 reads speed. (Amended
  before any W13 run: L4's own ``speed_ratio`` is forward-only, undefined for sidestep, and
  W11's experts do not all reach their command; §10.14.)
* **no regression** -- flat's L4 largest deviation (three commands) rises by no more than
  2 SE, falls rise by at most two in total, and S0's walk passes on every W13 tracker.

Rough is not read here: it is held out until the multi-task generator is evaluated.

Beside the criteria (``g1_gated_out``): the tasks W11 accepted whose W12 generator failed its
gate, on the G1's W13 trackers at their middle command, with the same "executes" reading. A
G1 decoder that executes their plans is the precondition W12 names for their DAgger round.

    python transfer/scripts/w13_compare.py --output transfer/results/w13_decoders.json
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
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from evaluate_latent import _settled  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from w8_compare import COMMANDS, condition  # noqa: E402

S = {
    "g1": ("warm_r3_logload_sym", "warm_r3_s42_log_sym", "warm_r3_s43_log_sym"),
    "nao": ("r1_3498_sym_v3", "warm_nao_r1_s32_sym", "warm_nao_r1_s33_sym"),
}
SEEDS = {"g1": (41, 42, 43), "nao": (31, 32, 33)}
MIDDLE = {"backward": "mid", "sidestep": "left", "jog": "mid"}
AXIS = {"backward": 0, "sidestep": 1, "jog": 0}
SPEED = (0.8, 1.2)
MAX_FALLS = 1


def new_tasks() -> list[str]:
    report = json.loads((RESULTS / "base_tasks.json").read_text(encoding="utf-8"))
    return [t["name"] for t in report["tasks"] if t["name"] != "flat"]


def axis_speed(path: str, axis: int) -> float | None:
    """Mean dimensionless CoM speed along ``axis`` (heading frame), once settled."""
    data = np.load(REPO_ROOT / path)
    running = int(data["settle_steps"]) if "settle_steps" in data.files else 0
    settled = _settled(data["done"], already_running=running)
    velocity = SKILL_LAYOUT.unpack(data["skill_state"].astype(float))["com_velocity"][..., axis]
    return float(velocity[settled].mean()) if settled.any() else None


def axis_ratio(body: str, tag: str, task: str) -> float:
    """The tracker's speed along the task's axis over its expert's, at the middle command."""
    path = RESULTS / f"zero_shot_{body}_{tag}_base_{task}_b4_auto_kin_l4_{MIDDLE[task]}.json"
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    run = axis_speed(runs["encoder"]["file"], AXIS[task])
    expert = axis_speed(runs["teacher"]["file"], AXIS[task])
    return 0.0 if run is None or not expert else run / expert


def gated_out() -> list[str]:
    """Tasks W11 accepted (calibrated reading) whose W12 generator exists and fails its gate."""
    accepted = json.loads((RESULTS / "w11_experts.json").read_text(encoding="utf-8"))
    out = []
    for task in accepted["accepted_calibrated"]:
        path = RESULTS / f"contract_base_{task}_b4.json"
        if path.exists() and not json.loads(path.read_text(encoding="utf-8"))["gate"]["passes"]:
            out.append(task)
    return out


def aside(tasks: list[str]) -> dict:
    """The gated-out tasks on the G1's W13 trackers: falls, speed along the axis, deviation."""
    report = {}
    for task in tasks:
        rows = []
        for seed in SEEDS["g1"]:
            tag = f"mt_g1_s{seed}_sym"
            row = condition("g1", tag, f"base_{task}_b4", MIDDLE[task])
            if row is None:
                continue
            row["axis_speed_ratio"] = axis_ratio("g1", tag, task)
            rows.append({"seed": seed, **row})
        report[task] = {
            "rows": rows,
            "executes": bool(
                len(rows) == len(SEEDS["g1"])
                and all(
                    r["falls"] <= MAX_FALLS and SPEED[0] <= r["axis_speed_ratio"] <= SPEED[1]
                    for r in rows
                )
            ),
        }
    return report


def walk(body: str, tag: str) -> dict | None:
    path = RESULTS / f"zero_shot_{body}_{tag}_walk_auto_kin.json"
    if not path.exists():
        return None
    verdict = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]["verdict"]
    return {"passes": bool(verdict["passes"]), "speed_ratio": verdict["speed_ratio"]}


def tracker(body: str, tag: str, tasks: list[str]) -> dict | None:
    flat = [condition(body, tag, "flat_dagger2", c) for c in COMMANDS]
    rows = {t: condition(body, tag, f"base_{t}_b4", MIDDLE[t]) for t in tasks}
    if any(r is None for r in flat) or any(r is None for r in rows.values()):
        return None
    for t, row in rows.items():
        row["axis_speed_ratio"] = axis_ratio(body, tag, t)
    return {
        "flat": {
            "largest": float(np.mean([r["largest"] for r in flat])),
            "falls": int(sum(r["falls"] for r in flat)),
            "passes": int(sum(r["passes"] for r in flat)),
        },
        "tasks": rows,
        "tasks_largest": float(np.mean([r["largest"] for r in rows.values()])),
        "walk": walk(body, tag),
    }


def paired(values: list[float]) -> dict:
    d = np.array(values)
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gated-out", action="store_true", help="print those tasks and stop")
    args = parser.parse_args()
    if args.gated_out:
        print(" ".join(gated_out()))
        return 0
    if args.output is None:
        parser.error("--output is required")
    tasks = new_tasks()
    report: dict = {"tasks": tasks, "middle": {t: MIDDLE[t] for t in tasks}}
    decision = []
    for body in ("g1", "nao"):
        pairs = []
        for seed, control in zip(SEEDS[body], S[body], strict=True):
            mt_tag = f"mt_{body}_s{seed}_sym"
            # S's flat and walk exist from the paper; its rows on the new tasks are W13's.
            before = tracker(body, control, tasks)
            after = tracker(body, mt_tag, tasks)
            if before is None or after is None:
                print(f"{body} seed {seed}: results missing")
                continue
            pairs.append(
                {"seed": seed, "s": control, "mt": mt_tag, "before": before, "after": after}
            )
        entry: dict = {"pairs": pairs}
        if len(pairs) >= 2 and tasks:
            target = paired(
                [p["after"]["tasks_largest"] - p["before"]["tasks_largest"] for p in pairs]
            )
            flat = paired(
                [p["after"]["flat"]["largest"] - p["before"]["flat"]["largest"] for p in pairs]
            )
            per_task = {
                t: paired(
                    [
                        p["after"]["tasks"][t]["largest"] - p["before"]["tasks"][t]["largest"]
                        for p in pairs
                    ]
                )
                for t in tasks
            }
            executes = {
                t: all(
                    p["after"]["tasks"][t]["falls"] <= MAX_FALLS
                    and SPEED[0] <= p["after"]["tasks"][t]["axis_speed_ratio"] <= SPEED[1]
                    for p in pairs
                )
                for t in tasks
            }
            falls = sum(p["after"]["flat"]["falls"] - p["before"]["flat"]["falls"] for p in pairs)
            no_regression = {
                "flat": flat["mean"] <= 2 * flat["se"],
                "falls": falls <= 2,
                "walk": all(p["after"]["walk"] and p["after"]["walk"]["passes"] for p in pairs),
            }
            entry.update(
                tasks_largest=target,
                per_task=per_task,
                flat_largest=flat,
                target_met=bool(target["mean"] < -2 * target["se"]),
                executes=executes,
                no_regression=no_regression,
                passes=bool(target["mean"] < -2 * target["se"] and all(no_regression.values())),
            )
            accepted = [t for t, ok in executes.items() if ok]
            decision.append(
                f"{body} target {entry['target_met']} executes {' '.join(accepted) or 'none'}"
            )
        report[body] = entry
    report["g1_gated_out"] = aside(gated_out())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print("; ".join(decision) or "incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
