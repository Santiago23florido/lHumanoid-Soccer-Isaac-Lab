"""Check the drive loads (``pact/anchors.py``, ``DriveLoads``) against the true loads of captures.

``DriveLoads`` estimates each foot's load from the card and the drives alone: every driven
joint's PD torque (the recorded gains and targets) balanced, in a serial statics, by the leg's
weight and the ground's wrench at its sole. A body whose legs are closed chains (Digit's rods
carry load past its knees) is outside that model, so before such a body enters PACT its
per-foot accuracy is read here on a capture of its teacher and compared with the four bodies
the model was built on (registered rule: per-foot R^2 at least 0.83, the lowest of them).

Captures need the joint states, targets and drive gains (``capture_contract.py --joints``): the
co-executions' map stage (``pact_coexecute.sh``) or the teacher audit's probes
(``pact_teacher_audit.sh``), the same 71 gait probes. Every 7th step from step 50 is read,
steps where an episode ends are dropped. Per foot: R^2 against the true load's variance, the
mean error (bias) and the mean absolute error, in body weights.

Usage::

    python transfer/scripts/pact_drive_loads_check.py --bodies g1,nao,h1,cassie
    python transfer/scripts/pact_drive_loads_check.py --bodies digit \\
        --captures "logs/pact/teachers/{body}_prec_probes.npz"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import torch  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors  # noqa: E402
from pact_check_anchors import sole_override  # noqa: E402

LOAD = SKILL_LAYOUT.slice_of("foot_load")
STEPS = slice(50, 900, 7)


def read(body: str, entry: dict, capture: Path) -> dict:
    card = json.loads((REPO / entry["card"]).read_text("utf-8"))
    d = np.load(capture)
    if tuple(str(n) for n in d["joint_names"]) != tuple(card["joints"]["names"]):
        raise SystemExit(f"{capture}: joint order differs from the card")
    anchors = KinematicAnchors(card, entry.get("leg_length_unit"), sole_override(body)).double()
    drive = DriveLoads(card, anchors).double()
    drive.set_drives(d["drive_stiffness"], d["drive_damping"], d["drive_effort_limit"])
    drive.double()

    def rows(key: str, width: int | None = None) -> torch.Tensor:
        x = d[key][STEPS]
        return torch.as_tensor(x.reshape(-1, width or x.shape[-1]), dtype=torch.float64)

    with torch.no_grad():
        estimate = drive(
            rows("joint_pos"), rows("joint_vel"), rows("projected_gravity"),
            target=rows("joint_pos_target"),
        )["loads"].numpy()
    true = d["skill_state"][STEPS].reshape(-1, SKILL_LAYOUT.size)[:, LOAD]
    kept = ~d["done"][STEPS].reshape(-1)
    feet = {}
    for f, side in enumerate(("left", "right")):
        e, t = estimate[kept, f], true[kept, f]
        feet[side] = {
            "r2": float(1.0 - ((e - t) ** 2).mean() / t.var()),
            "bias": float(np.mean(e - t)),
            "mae": float(np.abs(e - t).mean()),
        }
    return {"capture": str(capture.relative_to(REPO)), "rows": int(kept.sum()), "feet": feet}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--captures", default="logs/pact/coexec/{body}_map.npz")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    report = {"captures": args.captures, "steps": [STEPS.start, STEPS.stop, STEPS.step], "bodies": {}}
    for body in [b for b in args.bodies.split(",") if b]:
        entry = read(body, registry[body], REPO / args.captures.format(body=body))
        report["bodies"][body] = entry
        line = " | ".join(
            f"{side} R2 {v['r2']:.3f} bias {v['bias']:+.3f} mae {v['mae']:.3f}"
            for side, v in entry["feet"].items()
        )
        print(f"{body:7s} {line}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
