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
from humanoid_transfer.nao.assets.nao import NAO_DAMPING, NAO_STIFFNESS  # noqa: E402
from humanoid_transfer.nao.deploy.lola import JOINTS  # noqa: E402
from humanoid_transfer.nao.sim2real.actuator_fit import Drive, fit_joint, plant_for  # noqa: E402
from humanoid_transfer.nao.sim2real.events import reflected_inertia  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.15, help="Range around the fit, ±.")
    args = parser.parse_args()
    data = np.load(args.record, allow_pickle=False)
    joints = [str(j) for j in data["joints"]]
    base = dict(zip(JOINTS, (float(v) for v in data["base"]), strict=True))
    base["RHipYawPitch"] = base["LHipYawPitch"]
    limits = {j.name: j.effort for j in nk.load_model().joints}
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
            stiffness=NAO_STIFFNESS[joint],
            damping=NAO_DAMPING[joint],
            delay=0.03,
            armature=float(reflected_inertia([joint])[0]),
            ceiling=float(limits[joint]),
            friction=0.02,
        )
        drive, rms = fit_joint(plant, records, start)
        fits[joint] = {
            **asdict(drive),
            "rms_rad": rms,
            "inertia": plant.inertia,
            "stiffness_ratio": drive.stiffness / NAO_STIFFNESS[joint],
            "damping_ratio": drive.damping / NAO_DAMPING[joint],
            "ceiling_ratio": drive.ceiling / float(limits[joint]),
            "armature_ratio": drive.armature / max(float(reflected_inertia([joint])[0]), 1e-9),
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
            "effort_scale": span("ceiling_ratio"),
            "armature_scale": span("armature_ratio"),
            # The policy's 20 ms step held over LoLA's 12 ms cycle adds up to one cycle.
            "delay_ms": [float(delays.min() * (1 - m)), float(delays.max() * (1 + m) + 12.0)],
        }
        args.card.write_text(json.dumps(card, indent=1))
        print(f"robot card: {card}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
