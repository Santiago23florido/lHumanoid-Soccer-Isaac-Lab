"""Compare body cards' sole points: each foot's sole in its link frame, the soles in the default
posture, their left-right asymmetry, and the leg length they give.

A mirror-symmetric body standing in its symmetric default posture has soles mirror images of
each other: ``y_left + y_right`` is zero and the heights agree. ``--reference`` names a card
the others must agree with (the sole points within ``--tolerance`` metres, in the world, in
the default posture), and the exit status says whether they do.

    python transfer/scripts/card_soles.py transfer/bodies/cassie/body_card.json new_cassie.json
    python transfer/scripts/card_soles.py new_h1.json --reference transfer/bodies/h1/body_card.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def soles(card: dict) -> np.ndarray:
    return np.array(card["postures"]["default"]["soles"], dtype=float)


def describe(path: Path) -> dict:
    card = json.loads(path.read_text(encoding="utf-8"))
    s = soles(card)
    return {
        "card": str(path),
        "sole_point_link": {f: card["foot"][f]["sole_point_link"] for f in card["foot_bodies"]},
        "sole_rule": {f: card["foot"][f].get("sole_rule", "bbox_bottom") for f in card["foot_bodies"]},
        "default_soles": s.round(4).tolist(),
        "lateral_asymmetry": float(s[0, 1] + s[1, 1]),
        "height_asymmetry": float(s[0, 2] - s[1, 2]),
        "leg_length": float(card["leg_length"]),
        "com_height_above_soles": float(card["postures"]["default"]["com_height_above_soles"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("cards", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    agree = True
    reference = soles(json.loads(args.reference.read_text("utf-8"))) if args.reference else None
    for path in args.cards:
        d = describe(path)
        print(f"{path}: rule {sorted(set(d['sole_rule'].values()))}")
        for foot, point in d["sole_point_link"].items():
            print(f"  {foot:<24} sole in link {np.round(point, 4).tolist()}")
        print(
            f"  default soles {d['default_soles']}; y_left + y_right {d['lateral_asymmetry']:+.4f} m,"
            f" height difference {d['height_asymmetry']:+.4f} m; leg length {d['leg_length']:.4f} m,"
            f" CoM above soles {d['com_height_above_soles']:.4f} m"
        )
        if reference is not None:
            gap = float(np.abs(soles(json.loads(path.read_text("utf-8"))) - reference).max())
            agree &= gap <= args.tolerance
            print(f"  largest sole difference from {args.reference}: {gap:.4f} m")
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
