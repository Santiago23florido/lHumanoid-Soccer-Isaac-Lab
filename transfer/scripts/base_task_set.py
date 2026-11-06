"""The base tasks a decoder trains on, beside its walk (``method_revision.md`` §6, step 3).

Each entry is what ``ReferenceCommandCfg.tasks`` reads: the task's generator, its command
ranges **dimensionless** (velocities over the G1's ``V``, yaw rates times its ``T``), its
heading range, and its expert's DART capture (starts on the G1, windows on the NAO).

* **flat**: Isaac Lab's published G1 flat policy and its deployed generator, ``flat_dagger2``
  (L4's). Ranges from Isaac Lab's source: ``G1FlatEnvCfg`` for the velocities, the base
  ``CommandsCfg`` for the heading.
* **backward, sidestep, jog**: W11's experts, if accepted (``w11_experts.json``'s calibrated
  reading, the deviation declared in ``skill_space.md`` §10.12), with their
  W12 generators, if gated (``contract_base_<task>_b4.json``). Ranges from the run's saved
  ``env.yaml``, what the expert trained on.

The walk is task 0, the decoder's own contract (``HUMANOID_CONTRACT``), and is not listed.
**Rough is held out**: it is in no entry, and the script refuses any path that names it.

Every declared range is checked against the commands its DART capture recorded: each must
lie inside it.

    python transfer/scripts/base_task_set.py --output transfer/results/base_tasks.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402

ISAACLAB = Path("C:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion")
FLAT_CFG = ISAACLAB / "velocity/config/g1/flat_env_cfg.py"
BASE_CFG = ISAACLAB / "velocity/velocity_env_cfg.py"
NEW_TASKS = ("backward", "sidestep", "jog")
HELD_OUT = "rough"
KEYS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def _literal(node: ast.AST) -> object:
    """A literal, with ``math.pi`` allowed (the heading range is written with it)."""
    source = ast.unparse(node).replace("math.pi", repr(math.pi))
    return ast.literal_eval(source)


def flat_ranges() -> dict:
    ranges = {}
    for node in ast.walk(ast.parse(FLAT_CFG.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute):
            target = ast.unparse(node.targets[0])
            for key in KEYS:
                if target.endswith(f"base_velocity.ranges.{key}"):
                    ranges[key] = list(_literal(node.value))
    for node in ast.walk(ast.parse(BASE_CFG.read_text(encoding="utf-8"))):
        if isinstance(node, ast.keyword) and node.arg == "heading":
            ranges["heading"] = list(_literal(node.value))
    if set(ranges) != {*KEYS, "heading"}:
        raise RuntimeError(f"flat ranges incomplete: {ranges}")
    return ranges


def run_ranges(task: str) -> dict:
    runs = sorted((REPO_ROOT / "logs/rsl_rl/g1_flat").glob(f"*_base_{task}"))
    if not runs:
        raise FileNotFoundError(f"no run of base_{task}")
    env = yaml.unsafe_load((runs[-1] / "params/env.yaml").read_text(encoding="utf-8"))
    ranges = env["commands"]["base_velocity"]["ranges"]
    return {key: list(ranges[key]) for key in (*KEYS, "heading")}


def capture_check(dart: Path, ranges: dict) -> dict:
    command = np.load(dart)["command"].reshape(-1, 3)
    low = np.array([ranges[k][0] for k in KEYS])
    high = np.array([ranges[k][1] for k in KEYS])
    inside = bool(np.all((command >= low - 1e-6) & (command <= high + 1e-6)))
    return {"inside": inside, "min": command.min(0).tolist(), "max": command.max(0).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card = json.loads((REPO_ROOT / "g1/body/body_card.json").read_text(encoding="utf-8"))
    units = UnitSystem(float(card["leg_length"]), float(card["total_mass"]))
    scale = {"lin_vel_x": units.velocity, "lin_vel_y": units.velocity, "ang_vel_z": 1 / units.time}

    candidates = [("flat", flat_ranges(), "logs/contract/flat_dagger2/generator.pt", "flat_dart")]
    accepted = []
    accept_path = REPO_ROOT / "transfer/results/w11_experts.json"
    if accept_path.exists():
        accepted = json.loads(accept_path.read_text(encoding="utf-8"))["accepted_calibrated"]
    excluded = {}
    for task in NEW_TASKS:
        gate_path = REPO_ROOT / f"transfer/results/contract_base_{task}_b4.json"
        if task not in accepted:
            excluded[task] = "expert not accepted by W11 (calibrated reading)"
        elif not gate_path.exists():
            excluded[task] = "no W12 generator"
        elif not json.loads(gate_path.read_text(encoding="utf-8"))["gate"]["passes"]:
            excluded[task] = "W12 generator fails its gate"
        else:
            contract = f"logs/contract/base_{task}_b4/generator.pt"
            candidates.append((task, run_ranges(task), contract, f"base_{task}_dart"))

    tasks = []
    for name, physical, contract, capture in candidates:
        dart = f"g1/rollouts/{capture}.npz"
        if HELD_OUT in name or HELD_OUT in contract or HELD_OUT in dart:
            raise RuntimeError(f"{name}: the held-out task may not enter")
        check = capture_check(REPO_ROOT / dart, physical)
        if not check["inside"]:
            raise RuntimeError(f"{name}: its capture's commands leave the declared ranges")
        ranges = {k: [v / scale[k] for v in physical[k]] for k in KEYS}
        ranges["heading"] = physical["heading"]
        tasks.append(
            {
                "name": name,
                "contract": contract,
                "dart": dart,
                "ranges": ranges,
                "ranges_physical_g1": physical,
                "capture_commands": check,
            }
        )
    report = {
        "task_0": "the decoder's own walk contract (HUMANOID_CONTRACT), with its own ranges",
        "held_out": HELD_OUT,
        "units_g1": {"velocity": units.velocity, "time": units.time},
        "tasks": tasks,
        "excluded": excluded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for task in tasks:
        r = task["ranges"]
        print(
            f"{task['name']:9s} x {r['lin_vel_x'][0]:+.3f}..{r['lin_vel_x'][1]:+.3f}"
            f"  y {r['lin_vel_y'][0]:+.3f}..{r['lin_vel_y'][1]:+.3f}"
            f"  yaw {r['ang_vel_z'][0]:+.3f}..{r['ang_vel_z'][1]:+.3f} (dimensionless)"
        )
    for task, why in excluded.items():
        print(f"{task:9s} excluded: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
