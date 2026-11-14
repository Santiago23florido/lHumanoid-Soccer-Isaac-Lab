"""Read a sensitivity sweep's captures: how much each of a real NAO's imperfections costs.

Each capture (``transfer/scripts/capture_contract.py --robot-model ...``) holds
one setting of one imperfection. For each this reports the falls, the walking
speed, the gait's signature (``humanoid_transfer.common.gait``: stride period,
duty factor) and the contract error, against the capture named ``nominal``.

Usage::

    python nao/scripts/sensitivity.py nao/rollouts/sens_<tag>_*.npz \
        --output nao/results/sensitivity_<tag>.json
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

from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402


def summary(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    z = data["skill_state"].astype(np.float64)
    done = data["done"]
    terminated = data["terminated"]
    steps, envs = terminated.shape
    length = float(data["leg_length"])
    time_unit = math.sqrt(length / 9.81)
    velocity_unit = math.sqrt(9.81 * length)
    dt = float(data["control_dt"])
    signature = gait_signature(z, done, dt / time_unit)
    fields = SKILL_LAYOUT.unpack(z)
    forward = float(fields["com_velocity"][..., 0].mean()) * velocity_unit
    first = np.argmax(terminated, axis=0)
    fell = terminated.any(axis=0)
    return {
        "settings": json.loads(str(data["robot_model"])) if "robot_model" in data else {},
        "env_steps": int(steps * envs),
        "falls": int(terminated.sum()),
        "falls_per_1000_steps": 1000.0 * float(terminated.sum()) / (steps * envs),
        "envs_fallen": float(fell.mean()),
        "first_fall_s": float((first[fell] * dt).mean()) if fell.any() else None,
        "forward_speed_m_s": forward,
        "stride_period_s": signature.stride_period * time_unit,
        "duty_factor": float(np.mean(signature.duty_factor)),
        "contract_error_median": float(np.median(data["contract_error"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--nominal", default="nominal", help="Name (file suffix) of the reference.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for path in args.captures:
        name = path.stem.split("__")[-1]
        rows[name] = summary(path)
    reference = rows.get(args.nominal)
    for name, row in rows.items():
        if reference is not None:
            row["speed_ratio"] = row["forward_speed_m_s"] / reference["forward_speed_m_s"]
            row["stride_period_ratio"] = row["stride_period_s"] / reference["stride_period_s"]
            row["duty_change"] = row["duty_factor"] - reference["duty_factor"]
        print(
            f"{name:26s} falls/1000 {row['falls_per_1000_steps']:6.2f}  "
            f"fallen {row['envs_fallen']:.2f}  "
            f"v {row['forward_speed_m_s']:.3f} m/s ({row.get('speed_ratio', float('nan')):.2f})  "
            f"stride {row['stride_period_s']:.3f} s  duty {row['duty_factor']:.3f}  "
            f"err {row['contract_error_median']:.2f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"captures": [str(p) for p in args.captures], "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
