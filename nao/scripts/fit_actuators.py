"""Fit each leg joint's drive to a harness record, and write the ranges to retrain with.

Reads ``deploy.tools.sysid_record``'s output; for each joint fits
:class:`~humanoid_transfer.nao.sim2real.actuator_fit.Drive` to its steps and
chirp (``actuator_fit.fit_joint``), starting from the simulator's nominal drive.
Writes the per-joint fit and a robot card -- the settings of
``humanoid_transfer.nao.sim2real.robot_model`` that the measurement fixes, as
ranges around what was measured -- for ``HUMANOID_ROBOT_CARD`` to retrain with.

Usage::

    python nao/scripts/fit_actuators.py logs/sysid_<stamp>.npz --output nao/results/actuators.json \
        --card nao/results/robot_card.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.deploy.lola import JOINTS  # noqa: E402
from humanoid_transfer.nao.sim2real.actuator_fit import (  # noqa: E402
    Drive,
    fit_joint,
    peak_demand,
    plant_for,
)
from humanoid_transfer.nao.sim2real.motors import reflected_inertia  # noqa: E402

RATIOS = {
    "stiffness_scale": "stiffness_ratio",
    "damping_scale": "damping_ratio",
    "effort_scale": "ceiling_ratio",
    "armature_scale": "armature_ratio",
}
"""A student's drive parameters (``humanoid_transfer.nao.sim2real.params``), each the robot's
factor: the mean of its joints' fitted ratios."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bundle", type=Path, required=True, help="A deployment bundle: the simulated drives."
    )
    parser.add_argument("--card", type=Path, default=None)
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Write the robot's parameters as a student reads them (deploy.tools.run_policy "
        "--params): latency and each drive factor, the mean over the fitted joints.",
    )
    parser.add_argument("--margin", type=float, default=0.15, help="Range around the fit, ±.")
    args = parser.parse_args()
    data = np.load(args.record, allow_pickle=False)
    joints = [str(j) for j in data["joints"]]
    base = dict(zip(JOINTS, (float(v) for v in data["base"]), strict=True))
    base["RHipYawPitch"] = base["LHipYawPitch"]
    limits = {j.name: j.effort for j in nk.load_model().joints}
    drives = json.loads((args.bundle / "runtime.json").read_text(encoding="utf-8"))["drives"]
    nominal_stiffness = {n: d["stiffness"] for n, d in drives.items()}
    nominal_damping = {n: d["damping"] for n, d in drives.items()}
    fits = {}
    for joint in joints:
        index = JOINTS.index(joint)
        records = []
        for kind in ("steps", "chirp"):
            key = f"{joint}/{kind}"
            if f"{key}/time" not in data:
                continue
            records.append({
                "time": data[f"{key}/time"],
                "target": data[f"{key}/target"],
                "position": data[f"{key}/position"][:, index],
            })  # fmt: skip
        plant = plant_for(joint, base)
        start = Drive(
            stiffness=nominal_stiffness[joint],
            damping=nominal_damping[joint],
            delay=0.03,
            armature=float(reflected_inertia([joint])[0]),
            ceiling=float(limits[joint]),
            friction=0.02,
        )
        drive, rms = fit_joint(plant, records, start)
        demand = peak_demand(plant, drive, records)
        fits[joint] = {
            **asdict(drive),
            "rms_rad": rms,
            "inertia": plant.inertia,
            "stiffness_ratio": drive.stiffness / nominal_stiffness[joint],
            "damping_ratio": drive.damping / nominal_damping[joint],
            "ceiling_ratio": drive.ceiling / float(limits[joint]),
            "armature_ratio": drive.armature / max(float(reflected_inertia([joint])[0]), 1e-9),
            # A ceiling the record never reached does not shape the fit: the fitted value is
            # arbitrary above the largest torque asked for.
            "peak_demand": demand,
            "ceiling_identified": bool(demand >= drive.ceiling),
        }
        print(f"{joint}: " + ", ".join(f"{k} {v:.4g}" for k, v in fits[joint].items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"record": str(args.record), "joints": fits}, indent=1))
    if args.card is not None:
        m = args.margin

        def span(key: str) -> list[float]:
            values = np.array([f[key] for f in fits.values()])
            return [float(values.min() * (1 - m)), float(values.max() * (1 + m))]

        delays = 1000 * np.array([f["delay"] for f in fits.values()])
        card = {
            "stiffness_scale": span("stiffness_ratio"),
            "damping_scale": span("damping_ratio"),
            "armature_scale": span("armature_ratio"),
            # The policy's 20 ms step held over LoLA's 12 ms cycle adds up to one cycle.
            "delay_ms": [float(delays.min() * (1 - m)), float(delays.max() * (1 + m) + 12.0)],
        }
        saturated = [f["ceiling_ratio"] for f in fits.values() if f["ceiling_identified"]]
        if saturated:
            low, high = min(saturated), max(saturated)
            card["effort_scale"] = [float(low * (1 - m)), float(high * (1 + m))]
        else:
            print("no joint reached its ceiling: the card keeps the simulator's effort range")
        args.card.write_text(json.dumps(card, indent=1))
        print(f"robot card: {card}")
    if args.params is not None:
        params = {}
        for name, ratio in RATIOS.items():
            rows = [f for f in fits.values() if ratio != "ceiling_ratio" or f["ceiling_identified"]]
            if rows:
                params[name] = float(np.mean([f[ratio] for f in rows]))
            else:
                # Left out: a student keeps its own estimate of it.
                print(f"{name}: not identified by this record, left to the student's estimate")
        # The policy's 20 ms step waits for LoLA's next 12 ms cycle: 6 ms on average.
        params["delay_ms"] = 1000.0 * float(np.mean([f["delay"] for f in fits.values()])) + 6.0
        args.params.write_text(json.dumps(params, indent=1))
        print(f"robot parameters: {params}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
