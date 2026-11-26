"""Add a body's loop closures to its card: the USD's joints excluded from the articulation.

A leg with closed chains (Digit's achilles and toe rods) is simulated as a tree whose cut joints
PhysX enforces as constraints; the card's tree does not see them. Each such joint becomes a
``loops`` entry: the two bodies it joins and the point on each, in that body's own frame (the
joint's ``localPos0`` / ``localPos1``), which ``DriveLoads`` balances as an unknown force.

Reads the USD with ``pxr`` (Isaac Sim's, or the ``usd-core`` package outside Isaac); the asset is
the one the card's provenance names, read from a local copy.

    python transfer/scripts/card_loops.py --card transfer/bodies/digit/body_card.json \\
        --usd <local copy of the card's provenance asset>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Usd, UsdPhysics


def loops(usd: Path, bodies: list[str]) -> list[dict]:
    stage = Usd.Stage.Open(str(usd))
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint.GetExcludeFromArticulationAttr().Get():
            continue
        (b0,), (b1,) = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        names = [b0.name, b1.name]
        missing = [n for n in names if n not in bodies]
        if missing:
            raise SystemExit(f"{prim.GetPath()}: bodies {missing} are not the card's")
        out.append(
            {
                "joint": prim.GetName(),
                "kind": prim.GetTypeName(),
                "body0": names[0],
                "pos0": [float(v) for v in joint.GetLocalPos0Attr().Get()],
                "body1": names[1],
                "pos1": [float(v) for v in joint.GetLocalPos1Attr().Get()],
            }
        )
    return sorted(out, key=lambda e: e["joint"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--usd", type=Path, required=True, help="local copy of the card's asset")
    args = parser.parse_args()
    card = json.loads(args.card.read_text("utf-8"))
    found = loops(args.usd, list(card["bodies"]["names"]))
    card["loops"] = found
    card.setdefault("provenance", {})["loops"] = "transfer/scripts/card_loops.py (the asset's excluded joints)"
    args.card.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")  # as measure_body.py
    print(f"{args.card}: {len(found)} loop closures")
    for e in found:
        print(f"  {e['joint']}: {e['body0']} {e['pos0']} -- {e['body1']} {e['pos1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
