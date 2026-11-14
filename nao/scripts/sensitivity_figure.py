"""Figure: what each of a real NAO's imperfections costs a tracker, from sensitivity JSONs.

Two panels sharing the settings as rows -- falls per 1000 steps and the contract error
(median per environment, mean over environments), each with its standard error over
environments -- one bar per tracker. Settings missing from a tracker's sweep are left
blank for it.

Usage::

    python nao/scripts/sensitivity_figure.py nao/results/sensitivity_r1_3498_sym_v3.json \
        nao/results/sensitivity_<student>.json --labels "adopted tracker" "robot-trained" \
        --output nao/img/sensitivity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ("#2a78d6", "#eb6834")
"""Categorical slots 1 and 2 of the reference palette (validated for adjacent pairs)."""
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = (
    "nominal", "delay_10", "delay_20", "delay_30", "delay_40", "delay_60",
    "stiffness_0.5", "stiffness_0.75", "stiffness_1.5", "effort_0.6", "effort_0.8",
    "armature_0.5", "armature_1", "armature_2", "friction_0.4", "friction_0.6", "friction_1.2",
    "mass_0.85", "mass_1.15", "added_mass_0.2", "base_com_1.5cm", "sensors", "encoder_1deg",
    "encoder_3deg", "imu_2deg", "leg_odometry_velocity", "robot_estimator_contact",
    "robot_estimator_fsr", "robot_estimator_encoder_no_vel", "odometry_path", "robot_pipeline",
    "robot_expected",
)  # fmt: skip
NAMES = {
    "nominal": "simulator as trained",
    "delay_10": "latency 10 ms", "delay_20": "latency 20 ms", "delay_30": "latency 30 ms",
    "delay_40": "latency 40 ms", "delay_60": "latency 60 ms",
    "stiffness_0.5": "drive stiffness ×0.5", "stiffness_0.75": "drive stiffness ×0.75",
    "stiffness_1.5": "drive stiffness ×1.5",
    "effort_0.6": "torque ceiling ×0.6", "effort_0.8": "torque ceiling ×0.8",
    "armature_0.5": "rotor inertia ×0.5", "armature_1": "rotor inertia ×1",
    "armature_2": "rotor inertia ×2",
    "friction_0.4": "sole friction 0.4", "friction_0.6": "sole friction 0.6",
    "friction_1.2": "sole friction 1.2",
    "mass_0.85": "link masses ×0.85", "mass_1.15": "link masses ×1.15",
    "added_mass_0.2": "0.2 kg on the torso", "base_com_1.5cm": "torso CoM ±1.5 cm",
    "sensors": "encoders, IMU, finite differences", "encoder_1deg": "encoder offsets ±1°",
    "encoder_3deg": "encoder offsets ±3°", "imu_2deg": "IMU tilt ±2°",
    "leg_odometry_velocity": "velocity by leg odometry",
    "robot_estimator_contact": "robot's estimator, exact loads",
    "robot_estimator_fsr": "robot's estimator, force sensors",
    "robot_estimator_encoder_no_vel": "robot's estimator, learned loads",
    "odometry_path": "path from odometry", "robot_pipeline": "robot's whole pipeline",
    "robot_expected": "all of it, as expected on the robot",
}  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(p.read_text(encoding="utf-8"))["rows"] for p in args.sweeps]
    names = [n for n in ORDER if any(n in r for r in rows)]
    y = np.arange(len(names))
    height = 0.8 / len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 0.28 * len(names) + 1.2), sharey=True)
    panels = (
        ("falls_per_1000_steps", "falls_per_1000_steps_se", "falls per 1000 steps"),
        ("contract_error_env_median_mean", "contract_error_se", "contract error (tolerances)"),
    )
    for ax, (key, se_key, title) in zip(axes, panels, strict=True):
        for i, (sweep, label) in enumerate(zip(rows, args.labels, strict=True)):
            values = np.array([sweep.get(n, {}).get(key, np.nan) for n in names])
            errors = np.array([sweep.get(n, {}).get(se_key, 0.0) for n in names])
            offset = (i - (len(rows) - 1) / 2) * height
            ax.barh(
                y + offset,
                values,
                height=height * 0.9,
                color=SERIES[i],
                label=label,
                xerr=errors,
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0},
            )
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [NAMES.get(n, n) for n in names], color=INK)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
