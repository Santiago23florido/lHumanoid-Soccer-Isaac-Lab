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

**Calibrated reading** (a deviation declared in §10.12 before jog's play rollouts existed).
The same check, applied to the published flat expert that is already in the task set
(``g1/rollouts/flat_play_<forward|fast>.npz``: the same play variant, action noise 0, 32 x
900), gives a speed ratio of 0.73 at 0.25 m/s: the 0.8 bound rejects a member of the set it
selects for. ``accepted_calibrated`` lowers the bound to the reference's smallest ratio and
keeps every other criterion; ``accepted`` is the registered reading. The later stages read
``accepted_calibrated``.

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
# The published flat expert's single-axis play rollouts (``turning`` combines three axes).
REFERENCE = ("forward", "fast")


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
    row = {
        "command": list(command),
        "achieved_speed": achieved,
        "speed_ratio": ratio,
        "duty": list(duty),
        "falls": falls,
    }
    row["passes"] = passes(row, SPEED_RATIO[0])
    return row


def passes(row: dict, low: float) -> bool:
    """Every criterion, with ``low`` as the speed ratio's lower bound."""
    return bool(
        row["falls"] <= MAX_FALLS
        and low <= row["speed_ratio"] <= SPEED_RATIO[1]
        and max(row["duty"]) <= MAX_DUTY
    )


def dart_tracking(task: str) -> dict | None:
    """Diagnostic, not a criterion: the expert's DART capture, binned by commanded speed.

    Bins of 0.1 m/s along the task's axis; commands within 0.05 m/s of zero (the capture's
    standing share) are left out. Speed is the CoM's along the axis, as in ``check``.
    """
    path = REPO_ROOT / "g1" / "rollouts" / f"base_{task}_dart.npz"
    if not path.exists():
        return None
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    done = data["done"].astype(bool)
    valid = np.ones_like(done)
    valid[0] = False
    valid[1:] = ~done[:-1]
    axis = AXIS[task]
    velocity = SKILL_LAYOUT.unpack(data["skill_state"].astype(float))["com_velocity"][..., axis]
    velocity = velocity * units.velocity
    command = data["command"][..., axis]
    bins = []
    for low in np.arange(-0.5, 1.5, 0.1):
        inside = valid & (command >= low) & (command < low + 0.1) & (np.abs(command) > 0.05)
        if inside.sum() < 1000:
            continue
        achieved, commanded = float(velocity[inside].mean()), float(command[inside].mean())
        bins.append(
            {
                "commanded": commanded,
                "achieved": achieved,
                "ratio": achieved / commanded,
                "samples": int(inside.sum()),
            }
        )
    return {"falls": int(data["terminated"].sum()), "shape": list(done.shape), "bins": bins}


def calibration() -> dict:
    """The published flat expert, already in the task set, read with the same check."""
    rows = {}
    for name in REFERENCE:
        path = REPO_ROOT / "g1" / "rollouts" / f"flat_play_{name}.npz"
        command = tuple(float(c) for c in np.load(path)["command"].reshape(-1, 3)[0])
        rows[name] = check(path, command, 0)
    return {
        "reference": rows,
        "speed_ratio_low": min(r["speed_ratio"] for r in rows.values()),
        # The bound is the reference's only if the reference meets the other criteria.
        "reference_meets_the_rest": all(passes(r, 0.0) for r in rows.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = calibration()
    if not reference["reference_meets_the_rest"]:
        raise RuntimeError("the flat reference fails a criterion other than speed")
    low = reference["speed_ratio_low"]
    report: dict = {
        "criteria": {"max_falls": MAX_FALLS, "speed_ratio": SPEED_RATIO, "max_duty": MAX_DUTY},
        "calibration": reference,
        "tasks": {},
    }
    for task, commands in play_commands().items():
        rows = {}
        for name, command in commands.items():
            path = REPO_ROOT / "g1" / "rollouts" / f"base_{task}_play_{name}.npz"
            rows[name] = check(path, command, AXIS[task]) if path.exists() else None
            if rows[name] is not None:
                rows[name]["passes_calibrated"] = passes(rows[name], low)
        complete = all(r is not None for r in rows.values())
        report["tasks"][task] = {
            "commands": rows,
            "dart_tracking": dart_tracking(task),
            "accepted": bool(complete and all(r["passes"] for r in rows.values())),
            "accepted_calibrated": bool(
                complete and all(r["passes_calibrated"] for r in rows.values())
            ),
        }
    report["accepted"] = [t for t, e in report["tasks"].items() if e["accepted"]]
    report["accepted_calibrated"] = [
        t for t, e in report["tasks"].items() if e["accepted_calibrated"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for task, entry in report["tasks"].items():
        for name, r in entry["commands"].items():
            if r is None:
                print(f"{task:9s} {name:10s} missing")
                continue
            verdict = "pass" if r["passes"] else "FAIL"
            verdict += " / calibrated " + ("pass" if r["passes_calibrated"] else "FAIL")
            print(
                f"{task:9s} {name:10s} speed {r['achieved_speed']:+.3f}"
                f" (ratio {r['speed_ratio']:.2f})  duty {max(r['duty']):.2f}"
                f"  falls {r['falls']}  {verdict}"
            )
    for task, entry in report["tasks"].items():
        dart = entry["dart_tracking"]
        if dart is None:
            continue
        bins = "  ".join(f"{b['commanded']:+.2f}->{b['achieved']:+.3f}" for b in dart["bins"])
        print(f"{task:9s} dart falls {dart['falls']} in {dart['shape']}: {bins}")
    print(f"calibrated speed bound {low:.3f} (flat {', '.join(REFERENCE)})")
    print("accepted:", " ".join(report["accepted"]) or "none")
    print("accepted (calibrated):", " ".join(report["accepted_calibrated"]) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
