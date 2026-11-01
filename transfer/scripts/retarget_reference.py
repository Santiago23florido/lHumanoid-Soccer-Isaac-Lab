"""Turn a G1 capture into the NAO's joint-target reference (B3), one value map at a time.

Reads a ``capture_rollouts.py`` archive of the G1's teacher (``joint_pos``,
``skill_state``, ``base_ang_vel``, ``done``) and writes, for each uninterrupted
episode, the NAO's reference at the NAO's control step: joint targets
(:func:`~humanoid_transfer.transfer.retarget.retarget_angles`), their rates, and the
base's velocity (forward, lateral, yaw), in the NAO's units. The clips are
concatenated, with where each starts and how long it is.

Usage::

    python transfer/scripts/retarget_reference.py --source g1/rollouts/flat_dart.npz \
        --mode offset --output nao/rollouts/retarget_flat_dart_offset.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.transfer.embodiment import NAO  # noqa: E402
from humanoid_transfer.transfer.retarget import (  # noqa: E402
    MODES,
    JointRecord,
    episodes,
    resample,
    retarget_angles,
)

CONTROL_DT = 0.02
"""The NAO's control step, s (``nao/tasks``: decimation 4 at 0.005 s)."""
SHORTEST = 50
"""Clips shorter than this many of the NAO's steps are left out: an episode cut short."""


def teacher_record(names: list[str]) -> JointRecord:
    card = json.loads((ROOT / "g1" / "body" / "body_card.json").read_text(encoding="utf-8"))
    joints = card["joints"]
    order = [joints["names"].index(n) for n in names]
    limits = np.array(joints["pos_limits"], dtype=float)[order]
    default = np.array(joints["default_pos"], dtype=float)[order]
    return JointRecord(tuple(names), default, limits[:, 0], limits[:, 1])


def student_record() -> JointRecord:
    names = tuple(nk.ACTUATED_JOINTS)
    joints = {j.name: j for j in nk.load_model().joints}
    default = np.array([nk.NOMINAL_STAND_JOINT_POS.get(n, 0.0) for n in names])
    lower = np.array([joints[n].lower for n in names], dtype=float)
    upper = np.array([joints[n].upper for n in names], dtype=float)
    return JointRecord(names, default, lower, upper)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.source)
    teacher_units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    units = NAO.units
    teacher = teacher_record([str(n) for n in data["joint_names"]])
    student = student_record()
    # One NAO step spans this many of the teacher's: Δt_s T_t / (T_s Δt_t).
    factor = CONTROL_DT * teacher_units.time / (units.time * float(data["control_dt"]))

    angles = retarget_angles(data["joint_pos"].astype(float), teacher, student, args.mode)
    velocity = SKILL_LAYOUT.unpack(data["skill_state"].astype(float))["com_velocity"]
    yaw_rate = data["base_ang_vel"][..., 2].astype(float) * teacher_units.time
    base = np.concatenate(
        [velocity[..., :2] * units.velocity, yaw_rate[..., None] / units.time], -1
    )

    clips_q, clips_v, starts, lengths = [], [], [], []
    for env, first, last in episodes(data["done"]):
        q = resample(angles[first : last + 1, env], factor)
        if len(q) < SHORTEST:
            continue
        clips_q.append(q)
        clips_v.append(resample(base[first : last + 1, env], factor))
        starts.append(sum(lengths))
        lengths.append(len(q))
    q = np.concatenate(clips_q).astype(np.float32)
    rate = np.zeros_like(q)
    for start, length in zip(starts, lengths, strict=True):
        clip = q[start : start + length]
        rate[start : start + length] = np.gradient(clip, CONTROL_DT, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joint_pos=q,
        joint_vel=rate.astype(np.float32),
        base_velocity=np.concatenate(clips_v).astype(np.float32),
        clip_start=np.array(starts, dtype=np.int64),
        clip_length=np.array(lengths, dtype=np.int64),
        joint_names=np.array(student.names),
        mode=np.array(args.mode),
        source=np.array(str(args.source)),
        time_factor=np.float64(factor),
        control_dt=np.float32(CONTROL_DT),
    )
    clipped = np.mean(
        np.isclose(q, student.lower, atol=1e-6) | np.isclose(q, student.upper, atol=1e-6)
    )
    print(
        f"{args.mode}: {len(lengths)} clips, {q.shape[0]} steps of the NAO "
        f"({factor:.3f} of the teacher's each); {clipped:.1%} of angles at a limit; "
        f"written {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
