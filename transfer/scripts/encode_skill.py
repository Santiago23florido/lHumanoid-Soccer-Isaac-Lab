"""Encode the teacher's gait as a skill state, and check the encoding against physics.

The skill state is built to be shared between bodies; whether it is a faithful
description of *this* body's gait is a separate claim, and the rollout can
test it. Four checks, each against something the encoder did not use:

1. **Weight.** Averaged over the gait the feet must carry exactly the body's
   weight, because the CoM does not drift vertically: ``<load_L + load_R> = 1``.
   The capture this replaces would have reported 0.
2. **Newton, vertical, per sample.** ``load_L + load_R - 1 = c_z'' / g``
   instantaneously. The contact sensor and the body accelerations are
   independent outputs of the simulator.
3. **Spin term.** How much of ``k_G`` the term the old capture dropped carries.
4. **Re-encoding.** The raw per-body state kept for a few environments,
   encoded offline, must reproduce the state computed online.

Then what the state is for: the gait's dimensionless signature, the two bodies'
dimensionless groups and the defect between them, and Froude matching redone
with one consistent definition of leg length.

Every number lands in ``transfer/results/skill_space_teacher.json``, which is
where the documentation takes them from.

Usage::

    python transfer/scripts/encode_skill.py
    python transfer/scripts/encode_skill.py --rollouts g1/rollouts/teacher.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.frames import quat_to_matrix  # noqa: E402
from humanoid_transfer.common.gait import episode_start, gait_signature  # noqa: E402
from humanoid_transfer.common.scaling import froude_number  # noqa: E402
from humanoid_transfer.common.similarity import VELOCITY, UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    SKILL_LAYOUT,
    MultibodySnapshot,
    encode,
)
from humanoid_transfer.transfer.embodiment import (  # noqa: E402
    G1,
    G1_TEACHER_TASK,
    NAO,
    pi_groups,
    similarity_defect,
    task_groups,
)

DEFAULT_ROLLOUTS = REPO_ROOT / "g1" / "rollouts" / "teacher.npz"
DEFAULT_TRAINING = REPO_ROOT / "g1" / "rollouts" / "teacher_dart.npz"
DEFAULT_OUTPUT = REPO_ROOT / "transfer" / "results" / "skill_space_teacher.json"

# The values the teacher's command was sized with, pinned by tests/test_teacher_policy.py.
PINNED_TEACHER_LENGTH = 0.74
PINNED_STUDENT_LENGTH = 0.2689
PINNED_STUDENT_SPEED = 0.15
THRESHOLDS = (0.01, 0.02, 0.05, 0.10)


def _within_episodes(done: np.ndarray, before: int, after: int) -> np.ndarray:
    """``(T, N)`` True where steps ``t - before .. t + after`` exist and share one episode."""
    episode = np.cumsum(episode_start(done), axis=0)
    steps = done.shape[0]
    valid = np.zeros_like(done, dtype=bool)
    inner = slice(before, steps - after)
    valid[inner] = episode[: steps - after - before] == episode[before + after :]
    return valid


def certify(data: np.lib.npyio.NpzFile) -> dict:
    mass, length = float(data["total_mass"]), float(data["leg_length"])
    units = UnitSystem(length, mass)
    dt = float(data["control_dt"])
    done = data["done"]

    load = data["foot_force"][..., 2].sum(axis=-1) / (mass * units.gravity)
    acc = data["com_acc"][..., 2] / units.gravity
    vel = data["com_vel"][..., 2]
    finite = np.full_like(vel, np.nan)
    finite[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt) / units.gravity
    valid = _within_episodes(done, before=1, after=1)

    residual_sim = (load - 1.0 - acc)[valid]
    residual_fd = (load - 1.0 - finite)[valid]
    spin = np.linalg.norm(data["angular_momentum_spin"], axis=-1)
    total = np.linalg.norm(data["angular_momentum"], axis=-1)

    raw = data["raw_com_pos"].shape[1]
    snapshot = MultibodySnapshot(
        masses=data["masses"],
        inertia_local=data["inertia_link"],
        link_rot=quat_to_matrix(data["raw_link_quat"].astype(float)),
        com_pos=data["raw_com_pos"].astype(float),
        com_lin_vel=data["raw_com_lin_vel"].astype(float),
        ang_vel=data["raw_ang_vel"].astype(float),
        foot_force=data["foot_force"][:, :raw].astype(float),
        base_index=int(data["base_index"]),
        foot_index=tuple(int(i) for i in data["foot_index"]),
        sole_from_com=data["sole_from_com"],
    )
    reencoded = encode(snapshot, units)

    return {
        "mean_load_over_weight": float(load[valid].mean()),
        "mean_load_within_2_percent": bool(abs(load[valid].mean() - 1.0) <= 0.02),
        "newton_vertical_residual_rms_over_g": {
            "simulator_acceleration": float(np.sqrt(np.mean(residual_sim**2))),
            "finite_difference_acceleration": float(np.sqrt(np.mean(residual_fd**2))),
            "load_fluctuation_rms_for_scale": float((load[valid] - 1.0).std()),
        },
        "com_acceleration_simulator_vs_finite_difference": {
            "correlation_vertical": float(np.corrcoef(acc[valid], finite[valid])[0, 1]),
            "rms_difference_over_g": float(np.sqrt(np.mean((acc - finite)[valid] ** 2))),
        },
        "spin_share_of_angular_momentum": {
            "median": float(np.median(spin / np.maximum(total, 1e-9))),
            "p90": float(np.percentile(spin / np.maximum(total, 1e-9), 90)),
            "ratio_of_mean_norms": float(spin.mean() / total.mean()),
        },
        "reencoding_max_abs_error": float(np.abs(reencoded - data["skill_state"][:, :raw]).max()),
    }


def gait(data: np.lib.npyio.NpzFile) -> dict:
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    dt_star = float(data["control_dt"]) / units.time
    z, done = data["skill_state"].astype(float), data["done"]
    signature = gait_signature(z, done, dt_star).as_dict()
    # Alexander (1976): animals across a wide size range take strides of
    # about 2.3 Fr^0.3 hip heights. A reference point, not a target.
    signature["alexander_stride_length"] = 2.3 * signature["froude"] ** 0.3
    sensitivity = {
        f"{t:.2f}": gait_signature(z, done, dt_star, threshold=t).duty_factor for t in THRESHOLDS
    }
    fields = SKILL_LAYOUT.unpack(z)
    return {
        "signature": signature,
        "duty_factor_by_threshold": sensitivity,
        "mean_load_share": fields["foot_load"].mean(axis=(0, 1)).tolist(),
        "mean_height": float(fields["height"].mean()),
    }


def froude(data: np.lib.npyio.NpzFile) -> dict:
    commanded = float(data["commanded_speed"])
    achieved = float(np.linalg.norm(data["com_vel"][..., :2], axis=-1).mean())
    teacher, student = G1.units, NAO.units
    return {
        "pinned": {
            "teacher_length": PINNED_TEACHER_LENGTH,
            "teacher_length_is": "G1 pelvis spawn height",
            "student_length": PINNED_STUDENT_LENGTH,
            "student_length_is": "NAO CoM height above the soles, crouched",
            "student_speed": PINNED_STUDENT_SPEED,
            "teacher_speed": commanded,
        },
        "consistent": {
            "length_is": "hip-pitch joint to sole, leg straight, on both bodies",
            "teacher_length": teacher.length,
            "student_length": student.length,
            "length_ratio_pinned": PINNED_STUDENT_LENGTH / PINNED_TEACHER_LENGTH,
            "length_ratio_consistent": student.length / teacher.length,
            "student_equivalent_of_commanded": float(
                teacher.convert_to(student, commanded, VELOCITY)
            ),
            "student_equivalent_of_achieved": float(
                teacher.convert_to(student, achieved, VELOCITY)
            ),
            "froude_commanded": froude_number(commanded, teacher.length),
            "froude_achieved": froude_number(achieved, teacher.length),
        },
    }


def training_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = np.load(path)
    done = data["done"]
    ends = np.argwhere(done)
    command = data["command"]
    return {
        "file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "task": str(data["task"]),
        "steps_by_envs": list(done.shape),
        "action_noise": float(data["action_noise"]),
        "teacher_is_pinned": bool(data["teacher_is_pinned"]),
        "episode_ends": int(done.sum()),
        "episodes_terminated": int(data["terminated"].sum())
        if "terminated" in data.files
        else None,
        "episode_end_steps": sorted({int(t) for t, _ in ends}),
        "command_forward_range": [float(command[..., 0].min()), float(command[..., 0].max())],
        "command_yaw_rate_abs_max": float(np.abs(command[..., 2]).max()),
        "mean_speed": float(np.linalg.norm(data["com_vel"][..., :2], axis=-1).mean()),
        "mean_load_over_weight": float(
            (data["foot_force"][..., 2].sum(axis=-1) / (float(data["total_mass"]) * 9.81)).mean()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.rollouts.is_file():
        print(f"no rollouts at {args.rollouts}; run g1/scripts/capture_rollouts.py first")
        return 2
    data = np.load(args.rollouts)
    if "skill_state" not in data.files:
        print(f"{args.rollouts} predates the skill state; re-capture it")
        return 2

    units = G1.units
    report = {
        "source": {
            "rollouts": str(args.rollouts.relative_to(REPO_ROOT)).replace("\\", "/"),
            "task": str(data["task"]),
            "steps_by_envs": list(data["done"].shape),
            "control_dt": float(data["control_dt"]),
            "teacher_sha256": str(data["teacher_sha256"]),
            "teacher_is_pinned": bool(data["teacher_is_pinned"]),
        },
        "units": {
            "teacher": {"L": units.length, "m": units.mass, "T": units.time, "V": units.velocity},
            "student": {
                "L": NAO.units.length,
                "m": NAO.units.mass,
                "T": NAO.units.time,
                "V": NAO.units.velocity,
            },
        },
        "certificate": certify(data),
        "gait": gait(data),
        "embodiment": {
            "pi_teacher": pi_groups(G1),
            "pi_student": pi_groups(NAO),
            "defect_ln_student_over_teacher": similarity_defect(G1, NAO),
            "task_groups_teacher": task_groups(units, **G1_TEACHER_TASK),
            "task_groups_student": "set by the student task, which does not exist yet",
        },
        "froude": froude(data),
        "training_data": training_summary(args.training),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    cert, sig = report["certificate"], report["gait"]["signature"]
    fr = report["froude"]["consistent"]
    print("SKILL STATE OF THE TEACHER GAIT")
    print(f"  rollouts             : {report['source']['rollouts']}")
    print(f"  mean load / weight   : {cert['mean_load_over_weight']:.4f}")
    newton = cert["newton_vertical_residual_rms_over_g"]
    print(
        f"  Newton residual rms  : {newton['simulator_acceleration']:.4f} g (sim acc), "
        f"{newton['finite_difference_acceleration']:.4f} g (fin. diff.), "
        f"load fluctuation {newton['load_fluctuation_rms_for_scale']:.4f} g"
    )
    print(f"  spin share of |k|    : median {cert['spin_share_of_angular_momentum']['median']:.3f}")
    print(f"  re-encoding error    : {cert['reencoding_max_abs_error']:.2e}")
    print(
        f"  gait                 : Fr {sig['froude']:.4f}, stride {sig['stride_length']:.3f} L "
        f"(Alexander {sig['alexander_stride_length']:.3f}), period {sig['stride_period']:.2f} T, "
        f"duty {sig['duty_factor'][0]:.2f}/{sig['duty_factor'][1]:.2f}, "
        f"asymmetry {sig['asymmetry']:.3f}"
    )
    print(
        f"  froude, consistent L : student equivalent of the command "
        f"{fr['student_equivalent_of_commanded']:.4f} m/s (pinned assumption 0.15)"
    )
    print(f"  written              : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
