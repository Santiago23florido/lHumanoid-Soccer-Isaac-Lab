"""How much of each swing the leg drives deliver: the targets' excursion against the joints'.

``swing_bandwidth.py`` bounds what a leg can do; this reads what the drives do with the
targets a tracker sends. Per planned swing of each capture given (the swings of
``clearance_mechanism.swings``, through ``swing_bandwidth.capture_swings``), over the
window from two samples before the swing starts to its end:

* **excursions** -- each swing joint's (hip, knee, ankle pitch) target and position, from
  the swing's start, in the direction that lifts the foot: the target's largest excursion
  against the joint's;
* **the isolated-joint check** -- from each recorded state, ``I q̈ = clip(K (q* − q) − D q̇,
  ±effort)`` with ``q̇`` capped at the velocity limit, integrated over one control step
  against the leg's own inertia, predicts the next position. The slope and R² of the
  actual steps on the predicted ones say whether a swing joint moves as an isolated PD
  joint does;
* **an open-loop replay** -- the same targets, from the same state two samples before the
  swing, through the isolated joints with the leg gains scaled as ``K g`` and ``D √g``.
  That keeps the damping ratio against every inertia, and divides the drive's time
  constant ``D / K`` by ``√g``. The planar sole rise (``swing_bandwidth.ankle_rise``, foot
  kept flat) of the replayed hip and knee is reported as a share of the plan's asked
  lift, beside the rise the same planar leg gives for the recorded joints. It is a
  prediction that ignores the tracker's response to a faster leg; ``g = 1`` checks the
  replay against the record.

    python transfer/scripts/swing_drive.py \
        --nao-contract logs/contract/walk_dagger2e/generator.pt \
        --glob "nao/rollouts/zs_r1_3498_sym_v3_linrep_*_auto_kin_*_encoder.npz" \
        --glob "nao/rollouts/zs_warm_nao_r1_s3[23]_sym_*_auto_kin_*_encoder.npz" \
        --glob "g1/rollouts/zs_warm_r3_logload_sym_*_auto_kin_*_encoder.npz" \
        --glob "g1/rollouts/zs_warm_r3_s4[23]_log_sym_*_auto_kin_*_encoder.npz" \
        --output transfer/results/swing_drive.json

The NAO's glob leaves out the ``_gen`` captures (W1's other interface) by its prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import swing_bandwidth as sb  # noqa: E402
from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402

GAINS = (1.0, 2.0, 4.0, 8.0)
ACTION_SCALE = {"nao": 0.25, "g1": 0.5}
"""``JointPositionActionCfg.scale`` of each tracker (NAO task; Isaac Lab's G1 velocity task)."""
SIDES = {
    "nao": (
        {"hip": "LHipPitch", "knee": "LKneePitch", "ankle": "LAnklePitch"},
        {"hip": "RHipPitch", "knee": "RKneePitch", "ankle": "RAnklePitch"},
    ),
    "g1": (
        {
            "hip": "left_hip_pitch_joint",
            "knee": "left_knee_joint",
            "ankle": "left_ankle_pitch_joint",
        },
        {
            "hip": "right_hip_pitch_joint",
            "knee": "right_knee_joint",
            "ankle": "right_ankle_pitch_joint",
        },
    ),
}
LIFT = {"hip": -1.0, "knee": 1.0, "ankle": -1.0}
"""The sign of a joint's motion that lifts the foot: hip flexion, knee flexion, the ankle
flexing back to keep the foot level (both bodies' conventions)."""
SUBSTEP = 2.5e-4


def isolated_step(q, qd, target, joint: dict, dt: float, gain: float = 1.0):
    """``(q, q̇)`` after one control step of an isolated PD joint (arrays broadcast)."""
    k, d = joint["stiffness"] * gain, joint["damping"] * math.sqrt(gain)
    inertia = joint["swing_inertia"] + joint.get("armature", 0.0)
    for _ in range(int(round(dt / SUBSTEP))):
        tau = np.clip(k * (target - q) - d * qd, -joint["effort"], joint["effort"])
        qd = np.clip(qd + tau / inertia * SUBSTEP, -joint["velocity"], joint["velocity"])
        q = q + qd * SUBSTEP
    return q, qd


def capture(path: Path, body_name: str, body: dict) -> dict:
    """Per-swing excursions, isolated-joint step pairs and replayed rises of one capture."""
    data = np.load(path)
    names = [str(n) for n in data["joint_names"]]
    n = len(names)
    rows, dt, _, _ = sb.capture_swings(path, sb.KNEES[body_name])
    # capture_swings aligns its swings with z[1:]; the joint state at that sample is
    # encoder_input[1:], and the target applied after it is action_executed[1:].
    state = data["encoder_input"][1:].astype(float)
    action = data["action_executed"][1:].astype(float)
    acting = list(nk.ACTUATED_JOINTS) if body_name == "nao" else names
    scale = ACTION_SCALE[body_name]
    keep = [r for r in rows if r["start"] >= 2 and r["start"] + r["length"] + 1 <= len(state)]
    width = 3 + max(r["length"] for r in keep)
    # (swings, width) windows from two samples before the swing; past its end, padded
    # with the last sample and masked out.
    offsets = np.arange(width)
    stop = np.array([r["length"] + 3 for r in keep])
    mask = offsets[None, :] < stop[:, None]
    times = np.array([r["start"] - 2 for r in keep])[:, None] + np.minimum(
        offsets[None, :], stop[:, None] - 1
    )
    envs = np.array([r["env"] for r in keep])[:, None]
    sides = np.array([r["side"] for r in keep])
    series = {}
    for role in sb.JOINTS:
        columns = np.array([names.index(SIDES[body_name][s][role]) for s in sides])[:, None]
        acts = np.array([acting.index(SIDES[body_name][s][role]) for s in sides])[:, None]
        series[role] = (
            state[times, envs, 9 + columns],
            state[times, envs, 9 + n + columns],
            scale * action[times, envs, acts],
        )
    swing = mask & (offsets[None, :] >= 2)
    act = mask & (offsets[None, :] >= 1)
    out = {"excursion": {}, "step": {}, "rise": {}}
    for role in sb.JOINTS:
        q, qd, target = series[role]
        s = LIFT[role]
        # From the swing's start (window index 2); the target acting on the first swing
        # step was sent one sample earlier.
        out["excursion"][role] = {
            "target": np.where(act, s * (target - target[:, 1:2]), -np.inf).max(axis=1),
            "joint": np.where(swing, s * (q - q[:, 2:3]), -np.inf).max(axis=1),
        }
        predicted, _ = isolated_step(
            q[:, :-1], qd[:, :-1], target[:, :-1], body["joints"][role], dt
        )
        valid = mask[:, 1:]
        out["step"][role] = {
            "predicted": (predicted - q[:, :-1])[valid],
            "actual": (q[:, 1:] - q[:, :-1])[valid],
        }
    asked = np.array([r["asked"] for r in keep]) * body["leg_length"]

    def rise(hip: np.ndarray, knee: np.ndarray) -> np.ndarray:
        lift = sb.ankle_rise(body["thigh"], body["shank"], hip - hip[:, 2:3], knee - knee[:, 2:3])
        return np.where(swing, lift, -np.inf).max(axis=1) / np.maximum(asked, 1e-6)

    out["rise"]["recorded"] = rise(series["hip"][0], series["knee"][0])
    for g in GAINS:
        replay = {}
        for role in ("hip", "knee"):
            q, qd, target = series[role]
            path, q_now, qd_now = [q[:, 0]], q[:, 0], qd[:, 0]
            for t in range(width - 1):
                q_now, qd_now = isolated_step(
                    q_now, qd_now, target[:, t], body["joints"][role], dt, gain=g
                )
                path.append(q_now)
            replay[role] = np.stack(path, axis=1)
        out["rise"][f"g{g:g}"] = rise(replay["hip"], replay["knee"])
    out["asked"] = asked
    return out


def merge(parts: list[dict]) -> dict:
    """Concatenate the per-capture lists and summarise them."""
    summary = {"swings": int(sum(len(p["asked"]) for p in parts))}
    for role in sb.JOINTS:
        target = np.concatenate([p["excursion"][role]["target"] for p in parts])
        joint = np.concatenate([p["excursion"][role]["joint"] for p in parts])
        predicted = np.concatenate([p["step"][role]["predicted"] for p in parts])
        actual = np.concatenate([p["step"][role]["actual"] for p in parts])
        slope = float(np.sum(predicted * actual) / np.sum(predicted * predicted))
        r2 = float(1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
        summary[role] = {
            "target_excursion_median_rad": float(np.median(target)),
            "joint_excursion_median_rad": float(np.median(joint)),
            "share_of_target_delivered_median": float(
                np.median(joint[target > 0.01] / target[target > 0.01])
            )
            if (target > 0.01).any()
            else float("nan"),
            "swings_target_moves": float((target > 0.01).mean()),
            "isolated_step_slope": slope,
            "isolated_step_r2": r2,
        }
    summary["rise_over_asked_median"] = {
        key: float(np.median(np.concatenate([p["rise"][key] for p in parts])))
        for key in parts[0]["rise"]
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nao-contract", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--nao-gain",
        type=float,
        default=1.0,
        help="The NAO's leg gains as captured (HUMANOID_NAO_LEG_GAIN): K g, D sqrt(g).",
    )
    args = parser.parse_args()
    paths = list(args.capture) + [p for g in args.glob for p in sorted(REPO_ROOT.glob(g))]
    bodies = {"nao": sb.nao_body(args.nao_contract), "g1": sb.g1_body()}
    for joint in bodies["nao"]["joints"].values():
        joint["stiffness"] *= args.nao_gain
        joint["damping"] *= math.sqrt(args.nao_gain)
    parts: dict[str, list[dict]] = {"nao": [], "g1": []}
    for path in paths:
        name = "nao" if Path(path).resolve().parent.parent.name == "nao" else "g1"
        if name == "nao" and "_gen_" in Path(path).name:
            continue
        parts[name].append(capture(Path(path), name, bodies[name]))
        print(f"{Path(path).name}: {len(parts[name][-1]['asked'])} swings")
    report = {}
    for name, body in bodies.items():
        if not parts[name]:
            continue
        unit = math.sqrt(body["leg_length"] / nk.GRAVITY)
        drives = {
            role: {
                "time_constant_T": body["joints"][role]["damping"]
                / body["joints"][role]["stiffness"]
                / unit
            }
            for role in sb.JOINTS
        }
        report[name] = {"drives": drives, **merge(parts[name])}
        s = report[name]
        print(name)
        for role in sb.JOINTS:
            v = s[role]
            print(
                f"  {role:5} D/K {drives[role]['time_constant_T']:.3f} T  target "
                f"{v['target_excursion_median_rad']:.3f} rad, joint "
                f"{v['joint_excursion_median_rad']:.3f}, delivered "
                f"{v['share_of_target_delivered_median']:.2f} (target moves in "
                f"{v['swings_target_moves']:.2f})  isolated step slope "
                f"{v['isolated_step_slope']:.2f}, R2 {v['isolated_step_r2']:.2f}"
            )
        print(
            "  rise / asked, median: "
            + ", ".join(f"{k} {v:.2f}" for k, v in s["rise_over_asked_median"].items())
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
