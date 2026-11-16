"""How hard a captured NAO gait drives its leg motors, against their datasheet.

Reads captures written by ``transfer/scripts/capture_contract.py --joint-load``
and, for each sagittal and lateral leg joint, reports the torque the drive
applied (RMS, 95th percentile, peak), how often it sat at the simulator's effort
limit, the joint speed (95th percentile) and how often the pair (torque, speed)
lay beyond the motor's linear torque-speed line, ``|tau| > tau_stall (1 - |w| /
w_0)``, with an ideal gearbox (``nao/body/motors.json``). Steps after a fall are
left out. The RMS against the motor's nominal (continuous) torque is the
thermal reading: a real NAO's boards lower a joint's stiffness when it runs hot.

Usage::

    python nao/scripts/joint_load.py nao/rollouts/<capture>.npz [...] \
        --output nao/results/joint_load_<name>.json
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

JOINTS = ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch", "AnklePitch", "AnkleRoll")


def ratings() -> dict[str, dict[str, float]]:
    motors = json.loads((ROOT / "nao" / "body" / "motors.json").read_text(encoding="utf-8"))
    out = {}
    for joint in JOINTS:
        kind, reduction = motors["joints"][joint]
        motor = motors["motors"][kind]
        ratio = motor["reduction"][reduction]
        out[joint] = {
            "nominal": motor["nominal_torque_mNm"] * 1e-3 * ratio,
            "stall": motor["stall_torque_mNm"] * 1e-3 * ratio,
            "no_load_speed": motor["no_load_rpm"] * 2 * math.pi / 60 / ratio,
        }
    return out


def alive(terminated: np.ndarray) -> np.ndarray:
    """``(T, N)`` steps of each env before its first fall."""
    fallen = np.cumsum(terminated, axis=0) > 0
    return ~fallen


def effort_limits() -> dict[str, float]:
    """The simulator's torque ceiling per joint: the URDF's effort, used verbatim."""
    return {j.name: j.effort for j in nk.load_model().joints if j.effort is not None}


def load(capture: Path, rating: dict[str, dict[str, float]], limits: dict[str, float]) -> dict:
    data = np.load(capture, allow_pickle=False)
    if "joint_torque" not in data:
        raise ValueError(f"{capture} was captured without --joint-load")
    names = [str(n) for n in data["joint_names"]]
    keep = alive(data["terminated"])
    out = {"capture": capture.as_posix(), "steps": int(keep.sum()), "joints": {}}
    for joint in JOINTS:
        torque, speed = [], []
        for side in ("L", "R"):
            name = f"{side}{joint}"
            if joint == "HipYawPitch" and side == "R":
                continue  # one motor drives both; the right one is a mimic constraint
            i = names.index(name)
            torque.append(data["joint_torque"][..., i][keep])
            speed.append(data["joint_vel"][..., i][keep])
        tau, w = np.abs(np.concatenate(torque)), np.abs(np.concatenate(speed))
        r = rating[joint]
        curve = r["stall"] * np.clip(1.0 - w / r["no_load_speed"], 0.0, None)
        out["joints"][joint] = {
            "rms": float(np.sqrt(np.mean(tau**2))),
            "p95": float(np.percentile(tau, 95)),
            "peak": float(tau.max()),
            "rms_over_nominal": float(np.sqrt(np.mean(tau**2)) / r["nominal"]),
            "share_above_nominal": float(np.mean(tau > r["nominal"])),
            "share_at_effort_limit": float(np.mean(tau >= 0.99 * limits[f"L{joint}"])),
            "speed_p95": float(np.percentile(w, 95)),
            "share_beyond_torque_speed_line": float(np.mean(tau > curve)),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rating = ratings()
    limits = effort_limits()
    result = {"ratings": rating, "runs": [load(c, rating, limits) for c in args.captures]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    for run in result["runs"]:
        print(f"{run['capture']} ({run['steps']} env-steps alive)")
        for joint, v in run["joints"].items():
            print(
                f"  {joint:11s} rms {v['rms']:.2f} ({v['rms_over_nominal']:.2f}x nominal)  "
                f"p95 {v['p95']:.2f}  peak {v['peak']:.2f} N m  "
                f"above nominal {v['share_above_nominal']:.2f}  "
                f"at limit {v['share_at_effort_limit']:.3f}  speed p95 {v['speed_p95']:.2f} "
                f"rad/s  beyond line {v['share_beyond_torque_speed_line']:.3f}"
            )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
