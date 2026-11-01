"""Write the pre-registered gait probes: the family's grid, a few manoeuvres, and the twins.

A tracker's competence is scored gait by gait (``evaluate_competence.py``):
every environment of a capture follows one fixed gait of the procedural
family. The probes are fixed here, before any tracker is trained on the
family (``transfer/docs/skill_space.md`` §8.3):

* **grid** -- forward speed {0, 0.08, 0.16, 0.25, 0.35} ``V`` × stride period
  {0.8, 1.2, 1.8, 2.6} ``T`` × duty {0.4, 0.55, 0.7, 0.85}, at a middling
  width, height, clearance and lean, kept where the family's constraints hold
  for the G1's control step (the NAO's are checked when scoring);
* **manoeuvres** -- turning both ways, stepping sideways both ways, walking
  backwards, turning in place;
* **twins** -- the gaits of the walk teacher and of the published policy under
  its three test commands, as ``procedural_coverage.py`` measured them: the
  same gait written by the biped, to tell a tracker that cannot do a gait from
  one that cannot follow a task's plans of it.

With ``--corners``, the family's corners instead (``transfer/docs/skill_space.md`` §9, F4),
fixed before the physics trackers are trained: CoM height {0.80, 0.85, 0.92} ``L`` × swing
clearance {0.06, 0.10, 0.14} ``L`` × duty {0.55, 0.75} × forward speed {0.15, 0.30} ``V``,
stride period 1.0 ``T``. The lowest height and the highest clearance lie beyond the family's
ranges (0.85-1.00, 0.01-0.12), where Isaac Lab's rough-terrain policy walks (0.843, 0.145):
a tracker has to extrapolate there.

Usage::

    python transfer/scripts/make_gait_probes.py --output transfer/probes/gait_probes.json
    python transfer/scripts/make_gait_probes.py --corners \
        --output transfer/probes/corner_probes.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.procedural import (  # noqa: E402
    SWING_STEPS,
    GaitRanges,
    feasible,
)

G1_STEP = 0.07742576129865368
MIDDLE = {
    "lateral": 0.0,
    "turn": 0.0,
    "width": 0.38,
    "height": 0.92,
    "clearance": 0.06,
    "lean": 0.08,
}


def gait(group: str, name: str, **values) -> dict:
    probe = {**MIDDLE, **values}
    probe["duty"] = list(values["duty"])
    return {"group": group, "name": name, **probe}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--coverage", type=Path, default=REPO_ROOT / "transfer/results/procedural_coverage.json"
    )
    parser.add_argument("--corners", action="store_true", help="The family's corners only.")
    args = parser.parse_args()
    ranges = GaitRanges()
    min_swing = max(ranges.min_swing, SWING_STEPS * G1_STEP)
    if args.corners:
        return write(args.output, corners(ranges, min_swing))
    probes = []
    for forward, period, duty in itertools.product(
        (0.0, 0.08, 0.16, 0.25, 0.35), (0.8, 1.2, 1.8, 2.6), (0.4, 0.55, 0.7, 0.85)
    ):
        t = torch.tensor
        ok = feasible(
            t([period]), t([[duty, duty]]), t([forward]), t([0.0]), t([MIDDLE["width"]]),
            ranges, min_swing,
        )  # fmt: skip
        if bool(ok[0]):
            probes.append(
                gait(
                    "grid",
                    f"v{forward}_P{period}_d{duty}",
                    forward=forward,
                    period=period,
                    duty=(duty, duty),
                )  # fmt: skip
            )
    manoeuvres = [
        ("turn_left", dict(forward=0.1, turn=0.15)),
        ("turn_right", dict(forward=0.1, turn=-0.15)),
        ("side_left", dict(forward=0.0, lateral=0.08)),
        ("side_right", dict(forward=0.0, lateral=-0.08)),
        ("backward", dict(forward=-0.06)),
        ("turn_in_place", dict(forward=0.0, turn=0.2)),
    ]
    for name, values in manoeuvres:
        probes.append(gait("manoeuvre", name, period=1.2, duty=(0.65, 0.65), **values))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    for name, task in coverage["tasks"].items():
        measured = task["parameters"]
        probes.append(
            {
                "group": "twin",
                "name": name,
                **{
                    k: measured[k]
                    for k in (
                        "forward",
                        "lateral",
                        "turn",
                        "period",
                        "width",
                        "height",
                        "clearance",
                        "lean",
                    )
                },  # fmt: skip
                "duty": measured["duty"],
            }
        )
    return write(args.output, probes)


def corners(ranges: GaitRanges, min_swing: float) -> list[dict]:
    """Crouched, high-swing, long-stance gaits, some beyond the family's ranges."""
    probes = []
    for height, clearance, duty, forward in itertools.product(
        (0.80, 0.85, 0.92), (0.06, 0.10, 0.14), (0.55, 0.75), (0.15, 0.30)
    ):
        t = torch.tensor
        ok = feasible(
            t([1.0]), t([[duty, duty]]), t([forward]), t([0.0]), t([MIDDLE["width"]]),
            ranges, min_swing,
        )  # fmt: skip
        if bool(ok[0]):
            name = f"h{height}_c{clearance}_d{duty}_v{forward}"
            probes.append(
                gait("corner", name, forward=forward, period=1.0, duty=(duty, duty),
                     height=height, clearance=clearance)
            )  # fmt: skip
    return probes


def write(output: Path, probes: list[dict]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(probes, indent=1) + "\n", encoding="utf-8")
    groups = sorted({p["group"] for p in probes})
    counts = {g: sum(p["group"] == g for p in probes) for g in groups}
    print(f"{len(probes)} probes {counts}; written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
