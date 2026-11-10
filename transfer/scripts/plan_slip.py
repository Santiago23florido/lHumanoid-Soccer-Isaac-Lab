"""Do the plans' loaded feet stay still? The sticking half of contact consistency.

``z*`` carries each foot's load, its sole position and its sole velocity (the world
velocity, in the heading frame) as separate fields. On rigid ground a loaded foot rests on
the ground (``plan_contact.py``, W8) and does not move. This reads the second condition:
the speed of every foot carrying more than the contact threshold, horizontal and vertical,
in ``V``, split into firm load (above ``LIGHT_LOAD``) and light contact. Groups:

* the teachers' captures (flat, rough, and the walk teacher);
* the procedural biped's plans, as the G1 incumbent was given them at L1 -- the plans the
  trackers practise on;
* each body's plans (``contract_due``, aligned as ``capture_fidelity`` does) and the bodies
  themselves, S's three trackers per body, deployed path, both skills at three commands,
  and S0's walk;
* the same plans with W8's ``HUMANOID_PLAN_CONTACT=ground``, which sets only the height.

The loaded feet's sole height is reported beside it.

    python transfer/scripts/plan_slip.py --output transfer/results/plan_slip.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanoid_transfer.common.contact import LIGHT_LOAD, foot_height  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)
from plan_contact import COMMANDS, PROCEDURAL, SKILLS, TEACHERS, TRACKERS, aligned  # noqa: E402

WALK_TEACHER = "g1/rollouts/closed_loop_teacher.npz"


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p90": None, "n": 0}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "n": int(values.size),
    }


def loaded_motion(z: np.ndarray) -> dict:
    """Speed and sole height of loaded feet in ``z`` ``(..., 27)``: all, firm and light."""
    fields = SKILL_LAYOUT.unpack(z)
    load = fields["foot_load"]
    velocity = fields["foot_velocity"]
    horizontal = np.linalg.norm(velocity[..., :2], axis=-1)
    vertical = np.abs(velocity[..., 2])
    height = foot_height(z)
    loaded = load > DEFAULT_CONTACT_THRESHOLD
    masks = {"loaded": loaded, "firm": load > LIGHT_LOAD, "light": loaded & (load <= LIGHT_LOAD)}
    out = {}
    for name, mask in masks.items():
        out[name] = {
            "horizontal_speed": _stats(horizontal[mask]),
            "vertical_speed": _stats(vertical[mask]),
        }
    out["loaded"]["sole_height"] = _stats(height[loaded])
    return out


def capture_paths(body: str, tag: str, skill: str) -> list[Path]:
    root = REPO_ROOT / body / "rollouts"
    if skill == "walk":
        return [root / f"zs_{tag}_walk_auto_kin_encoder.npz"]
    return [root / f"zs_{tag}_{SKILLS[skill]}_auto_kin_{c}_encoder.npz" for c in COMMANDS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"teachers": {}, "bodies": {}}
    for skill, files in TEACHERS.items():
        z = np.concatenate(
            [np.load(REPO_ROOT / f)["skill_state"].astype(float) for f in files], axis=1
        )
        report["teachers"][skill] = loaded_motion(z)
    walk = np.load(REPO_ROOT / WALK_TEACHER)["skill_state"].astype(float)
    report["teachers"]["walk"] = loaded_motion(walk)
    procedural = np.load(REPO_ROOT / PROCEDURAL)["contract_due"][:-1].astype(float)
    report["procedural_plans"] = loaded_motion(procedural)
    for body, tags in TRACKERS.items():
        for skill in (*SKILLS, "walk"):
            for contact in ("as_drawn", "ground"):
                plans, bodies = [], []
                for tag in tags:
                    name = tag if contact == "as_drawn" else f"{tag}_ground"
                    for path in capture_paths(body, name, skill):
                        executed, due, valid, _ = aligned(path)
                        plans.append(due[valid])
                        bodies.append(executed[valid])
                report["bodies"][f"{body} {skill} {contact}"] = {
                    "plans": loaded_motion(np.concatenate(plans)),
                    "bodies": loaded_motion(np.concatenate(bodies)),
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def row(name: str, entry: dict) -> None:
        firm = entry["firm"]["horizontal_speed"]["median"]
        print(
            f"{name:30s} loaded horizontal {entry['loaded']['horizontal_speed']['median']:.4f}"
            f"  vertical {entry['loaded']['vertical_speed']['median']:.4f}"
            f"  firm horizontal {firm:.4f}"
            f"  height p90 {entry['loaded']['sole_height']['p90']:.4f}"
        )

    for skill, entry in report["teachers"].items():
        row(f"teacher {skill}", entry)
    row("procedural plans", report["procedural_plans"])
    for name, entry in report["bodies"].items():
        row(f"{name} plans", entry["plans"])
        row(f"{name} bodies", entry["bodies"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
