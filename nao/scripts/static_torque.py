"""Static leg torques of the NAO in a crouch, against what its motors can hold.

A tracker following plans of ``z*`` stands as low, in leg lengths, as the task's
body walked, and the NAO bends its knees for that. The knee then carries the
body's weight on a long horizontal lever. This script computes, from the URDF,
the torque each sagittal leg joint holds at rest in double support (half the
weight on each sole) and in single support (the whole weight on one), with the
centre of pressure under the centre of mass, and compares it with the motor's
ratings at the joint (``nao/body/motors.json``: rotor torque times reduction,
with an ideal gearbox, so the motor-side requirement is at least this).

The crouches are the sagittal family the tracker's posture is drawn from (hip
and ankle at minus half the knee, torso upright, feet flat), over a sweep of
knee angles; ``--env-yaml`` adds the posture a trained tracker actually used.
No dynamics: walking adds inertial torques on top, so this is a lower bound.

Usage::

    python nao/scripts/static_torque.py \
        --env-yaml logs/rsl_rl/nao_track/<run>/params/env.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.transfer.embodiment import NAO  # noqa: E402

SAGITTAL = ("HipPitch", "KneePitch", "AnklePitch")
KNEES = (0.0, 0.4, 0.8, 1.0, 1.2, 1.4)


def crouch(knee: float) -> dict[str, float]:
    posture = dict(nk.NOMINAL_STAND_JOINT_POS)
    for side in ("L", "R"):
        posture[f"{side}HipPitch"] = -knee / 2
        posture[f"{side}KneePitch"] = knee
        posture[f"{side}AnklePitch"] = -knee / 2
    return posture


def posture_from(env_yaml: Path) -> dict[str, float]:
    import yaml

    env = yaml.unsafe_load(env_yaml.read_text(encoding="utf-8"))
    return {**nk.NOMINAL_STAND_JOINT_POS, **env["scene"]["robot"]["init_state"]["joint_pos"]}


def ratings(motors: dict) -> dict[str, dict[str, float]]:
    """Each joint's torque ratings at its output, N m, and no-load speed, rad/s."""
    out = {}
    for joint, (kind, reduction) in motors["joints"].items():
        motor = motors["motors"][kind]
        ratio = motor["reduction"][reduction]
        out[joint] = {
            "nominal": motor["nominal_torque_mNm"] * 1e-3 * ratio,
            "stall": motor["stall_torque_mNm"] * 1e-3 * ratio,
            "no_load_speed": motor["no_load_rpm"] * 2 * math.pi / 60 / ratio,
        }
    return out


def holding(posture: dict[str, float], load: float) -> dict[str, float]:
    """Torque each left sagittal joint holds with ``load`` times the weight on the left sole."""
    weight = nk.load_model().total_mass * nk.GRAVITY
    com = nk.center_of_mass(posture)
    sole = nk.forward_kinematics(posture)["l_sole"][:3, 3]
    cop = np.array([com[0], sole[1], sole[2]])
    force = np.array([0.0, 0.0, load * weight])
    return {j: nk.static_joint_torque(f"L{j}", force, cop, posture) for j in SAGITTAL}


def row(posture: dict[str, float], rating: dict[str, dict[str, float]]) -> dict:
    double, single = holding(posture, 0.5), holding(posture, 1.0)
    return {
        "knee": posture["LKneePitch"],
        "com_height_L": nk.com_height_above_soles(posture) / NAO.leg_length,
        "joints": {
            j: {
                "double_support": double[j],
                "single_support": single[j],
                "single_over_nominal": abs(single[j]) / rating[j]["nominal"],
                "single_over_stall": abs(single[j]) / rating[j]["stall"],
            }
            for j in SAGITTAL
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env-yaml", type=Path, help="A trained tracker's params/env.yaml.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "nao" / "results" / "static_torque.json"
    )
    args = parser.parse_args()

    motors = json.loads((ROOT / "nao" / "body" / "motors.json").read_text(encoding="utf-8"))
    rating = ratings(motors)
    model = nk.load_model()
    result = {
        "weight_N": model.total_mass * nk.GRAVITY,
        "ratings": {j: rating[j] for j in SAGITTAL},
        "urdf_effort": {
            j: next(x.effort for x in model.joints if x.name == f"L{j}") for j in SAGITTAL
        },
        "sweep": [row(crouch(k), rating) for k in KNEES],
    }
    if args.env_yaml is not None:
        tracker = row(posture_from(args.env_yaml), rating)
        result["tracker"] = {"env_yaml": args.env_yaml.as_posix(), **tracker}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"weight {result['weight_N']:.2f} N; ratings at the joint (ideal gearbox):")
    for j in SAGITTAL:
        r = rating[j]
        print(
            f"  {j:10s} nominal {r['nominal']:.2f}  stall {r['stall']:.2f} N m  "
            f"URDF {result['urdf_effort'][j]}  no-load {r['no_load_speed']:.2f} rad/s"
        )
    rows = result["sweep"] + ([result["tracker"]] if "tracker" in result else [])
    for entry in rows:
        cells = "  ".join(
            f"{j}: {v['double_support']:+.2f}/{v['single_support']:+.2f} "
            f"({v['single_over_nominal']:.2f}x nominal)"
            for j, v in entry["joints"].items()
        )
        print(f"knee {entry['knee']:.3f}  CoM {entry['com_height_L']:.3f} L  {cells}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
