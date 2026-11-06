"""W11's pre-registered acceptance check (skill_space.md §10.12): does each new expert walk?

For each base task (``g1/tasks/g1_base``), its play rollouts at the three fixed commands of
``PLAY_COMMANDS`` (``g1/rollouts/base_<task>_play_<name>.npz``):

* **falls** -- terminated episodes in the capture (32 environments x 900 steps), at most 1;
* **speed** -- the CoM's mean velocity along the commanded axis (``z*``'s ``com_velocity``,
  heading frame, times ``V``) over the commanded speed, between 0.8 and 1.2;
* **stepping** -- each foot's duty factor (``common.contact``) at most 0.9: the expert lifts
  its feet.

A task is accepted if every one of its commands passes all three. The checks select which
experts enter the generator's task set; none is a training condition. A rejected task is
reported with its figures and left out.

    python transfer/scripts/w11_accept.py --output transfer/results/w11_experts.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import contact_fidelity  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402

CFG = REPO_ROOT / "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_base/g1_base_env_cfg.py"
AXIS = {"backward": 0, "sidestep": 1, "jog": 0}
MAX_FALLS = 1
SPEED_RATIO = (0.8, 1.2)
MAX_DUTY = 0.9


def play_commands() -> dict:
    """``PLAY_COMMANDS`` read from the configuration's source (it imports Isaac Lab)."""
    for node in ast.parse(CFG.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PLAY_COMMANDS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("PLAY_COMMANDS not found")


def check(path: Path, command: tuple[float, float, float], axis: int) -> dict:
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    z = data["skill_state"].astype(float)
    done = data["done"].astype(bool)
    first = np.zeros_like(done)
    first[0] = True
    first[1:] = done[:-1]
    valid = ~first
    velocity = SKILL_LAYOUT.unpack(z)["com_velocity"][..., axis] * units.velocity
    achieved = float(velocity[valid].mean())
    ratio = achieved / command[axis]
    duty = contact_fidelity(z, done, float(data["control_dt"]) / units.time).duty_body
    falls = int(data["terminated"].sum())
    return {
        "command": list(command),
        "achieved_speed": achieved,
        "speed_ratio": ratio,
        "duty": list(duty),
        "falls": falls,
        "passes": bool(
            falls <= MAX_FALLS
            and SPEED_RATIO[0] <= ratio <= SPEED_RATIO[1]
            and max(duty) <= MAX_DUTY
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {
        "criteria": {"max_falls": MAX_FALLS, "speed_ratio": SPEED_RATIO, "max_duty": MAX_DUTY},
        "tasks": {},
    }
    for task, commands in play_commands().items():
        rows = {}
        for name, command in commands.items():
            path = REPO_ROOT / "g1" / "rollouts" / f"base_{task}_play_{name}.npz"
            rows[name] = check(path, command, AXIS[task]) if path.exists() else None
        complete = all(r is not None for r in rows.values())
        report["tasks"][task] = {
            "commands": rows,
            "accepted": bool(complete and all(r["passes"] for r in rows.values())),
        }
    report["accepted"] = [t for t, e in report["tasks"].items() if e["accepted"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for task, entry in report["tasks"].items():
        for name, r in entry["commands"].items():
            if r is None:
                print(f"{task:9s} {name:10s} missing")
                continue
            verdict = "pass" if r["passes"] else "FAIL"
            print(
                f"{task:9s} {name:10s} speed {r['achieved_speed']:+.3f}"
                f" (ratio {r['speed_ratio']:.2f})  duty {max(r['duty']):.2f}"
                f"  falls {r['falls']}  {verdict}"
            )
    print("accepted:", " ".join(report["accepted"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
