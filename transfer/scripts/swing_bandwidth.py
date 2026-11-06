"""Can each body's swing leg lift its foot as high, as fast, as its plans ask?

Read with ``clearance_mechanism.py`` (research_report.md §2.2, item 4). The trackers'
actions are joint position targets, held for a control step, that an implicit PD drive
(stiffness K, damping D, effort limit) follows at the physics rate. In the air a leg joint
moves only the leg below it -- its distal inertia I, not the whole body the NAO's leg gains
were sized against (``nao/scripts/derive_gains.py``). Per body, for the sagittal chain of
one leg (hip pitch, knee, ankle pitch) at the tracker's crouch -- the NAO's from its URDF at
``contract_posture``'s knee, the G1's from its measured body card at the teacher's default:

* **the drive in swing**, per joint: natural frequency sqrt(K/I), damping ratio
  D / (2 sqrt(K I)) and its response time -- the slow pole's time constant when
  overdamped, 1 / sqrt(K/I) when not -- in seconds and in the body's time unit
  T = sqrt(L/g): how far ahead of the motion a policy must put its targets;
* **the excursion ceiling**, per joint: whatever the targets, the largest excursion of a
  joint at rest that must be back within a time t, accelerating at most effort / I and, where
  stated, moving at most at its velocity limit (full effort up, full braking, the return
  free to end at any speed). Each joint is bounded on its own: gravity and the joints'
  coupling are left out, which would lower it;
* **the lift ceiling**: the highest the sole rises, foot kept flat (the ankle takes minus the
  hip's and knee's sum), over hip and knee excursions within their ceilings and the three
  joints' ranges, hard or soft (the soft range is where ``dof_pos_limits`` starts to charge);
* **against the plans**: per planned swing of the captures (``clearance_mechanism.swings``),
  the lift ceiling over the asked lift, for the airborne time and for one more control step
  (flexion that starts while the foot still bears load), beside the clearance ratio the
  body reached, by the swing's length;
* **whether the ceiling holds**: the ceiling assumes a leg at rest at lift-off, so it bounds
  what a body does only if its swings start that way. Per variant, the share of swings
  whose reached lift exceeds the ceiling, and the swing knee's speed at the body's own
  lift-off (its first airborne sample in ``clearance_mechanism``'s widened window; positive
  flexes, both bodies' convention);
* **what the swing knee did**: its peak speed over each planned swing, read from the
  captures' ``encoder_input`` (joint velocities, articulation order), in rad/s and rad/T,
  and the share of swings in which it reached its velocity limit.

The NAO's velocity limit is its URDF's (6.40 rad/s on the leg joints); the captured knee
speeds show whether the simulator enforces it, and the ceiling is also given without it.
The G1's limits are the ones its body card measured in the simulator.

    python transfer/scripts/swing_bandwidth.py \
        --nao-contract logs/contract/walk_dagger2e/generator.pt \
        --capture nao/rollouts/zs_r1_3498_sym_f5_map.npz \
        --capture nao/rollouts/zs_r1_3498_sym_f5_corners.npz \
        --glob "nao/rollouts/zs_warm_nao_r1_s3?_sym_*_auto_kin_*_encoder.npz" \
        --capture g1/rollouts/zs_warm_r3_logload_sym_map.npz \
        --capture g1/rollouts/zs_warm_r3_logload_sym_corners.npz \
        --glob "g1/rollouts/zs_warm_r3_logload_sym_*_auto_kin_*_encoder.npz" \
        --output transfer/results/swing_bandwidth.json
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import clearance_mechanism as cm  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)
from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.transfer.embodiment import NAO  # noqa: E402

JOINTS = ("hip", "knee", "ankle")
NAO_JOINTS = {"hip": "LHipPitch", "knee": "LKneePitch", "ankle": "LAnklePitch"}
NAO_LINKS = {"hip": "LThigh", "knee": "LTibia", "ankle": "LAnklePitch"}  # each joint's child
G1_JOINTS = {"hip": "left_hip_pitch", "knee": "left_knee", "ankle": "left_ankle_pitch"}
KNEES = {"nao": ("LKneePitch", "RKneePitch"), "g1": ("left_knee_joint", "right_knee_joint")}
AT_CAP = 0.98  # a knee within 2 % of its velocity limit is at it
G1_DISTAL = {
    "hip": ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"),
    "knee": ("knee", "ankle_pitch", "ankle_roll"),
    "ankle": ("ankle_pitch", "ankle_roll"),
}
VARIANTS = {
    "hard": {"range": "hard", "velocity": True},
    "soft": {"range": "soft", "velocity": True},
    "hard_no_velocity_cap": {"range": "hard", "velocity": False},
}
EXTRA_STEPS = (0, 1)
GRID = 121
SOFT_FACTOR = 0.9  # both bodies' soft_joint_pos_limit_factor
NAO_SOURCE = REPO_ROOT / "source/humanoid_transfer/humanoid_transfer/nao/assets/nao.py"
G1_CARD = REPO_ROOT / "g1" / "body" / "body_card.json"


def excursion_ceiling(alpha: float, velocity: float, duration: float) -> float:
    """Largest excursion of a joint at rest that must be back within ``duration``.

    Rising for ``t_r`` from rest to rest covers at most ``alpha t_r^2 / 4`` (or, once the
    velocity limit binds, ``v (t_r - v / alpha)``); returning from rest within the rest of
    the time at any final speed covers at most ``alpha t_f^2 / 2`` (or
    ``v (t_f - v / (2 alpha))``). The peak is the best split. Without a velocity limit it
    is ``(1 - 1/sqrt 2)^2 alpha duration^2``.
    """
    t_r = np.linspace(0.0, duration, 4001)
    t_f = duration - t_r
    up = np.where(
        t_r <= 2 * velocity / alpha, alpha * t_r**2 / 4, velocity * (t_r - velocity / alpha)
    )
    down = np.where(
        t_f <= velocity / alpha, alpha * t_f**2 / 2, velocity * (t_f - velocity / (2 * alpha))
    )
    return float(np.max(np.minimum(up, down)))


def _assigned_literal(path: Path, name: str):
    """The expression assigned to ``name`` at the top level of ``path``, without importing it."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value:
            return node.value
    raise KeyError(f"{name} not assigned in {path}")


def g1_actuators() -> dict:
    """``G1_CFG``'s leg and foot drives, parsed from Isaac Lab's source (it needs Kit to import)."""
    spec = importlib.util.find_spec("isaaclab_assets")
    path = Path(spec.submodule_search_locations[0]) / "robots" / "unitree.py"
    call = _assigned_literal(path, "G1_CFG")
    actuators = next(k.value for k in call.keywords if k.arg == "actuators")
    groups = {
        key.value: {
            k.arg: ast.literal_eval(k.value) for k in value.keywords if k.arg != "joint_names_expr"
        }
        for key, value in zip(actuators.keys, actuators.values, strict=True)
    }

    def pick(group: dict, field: str, joint: str) -> float:
        value = group[field]
        if not isinstance(value, dict):
            return float(value)
        return float(next(v for pattern, v in value.items() if re.fullmatch(pattern, joint)))

    out = {"source": str(path)}
    for role, joint in G1_JOINTS.items():
        group = groups["feet"] if role == "ankle" else groups["legs"]
        out[role] = {
            f: pick(group, f, f"{joint}_joint") for f in ("stiffness", "damping", "armature")
        }
    return out


def _sagittal(vector: np.ndarray) -> np.ndarray:
    return np.array([vector[0], vector[2]])


def ankle_rise(thigh: np.ndarray, shank: np.ndarray, d_hip, d_knee):
    """The ankle's rise, hip fixed, when the hip and knee pitch turn by ``d_hip``, ``d_knee``.

    Pitch is a rotation about +y: (x, z) -> (x cos + z sin, -x sin + z cos).
    """

    def z(vector, angle):
        return -vector[0] * np.sin(angle) + vector[1] * np.cos(angle)

    return z(thigh, d_hip) + z(shank, d_hip + d_knee) - (thigh[1] + shank[1])


def nao_crouch(contract: Path) -> dict[str, float]:
    """``nao_track_env_cfg.contract_posture``: the crouch at the contract's mean height."""
    import torch

    saved = torch.load(contract, map_location="cpu", weights_only=False)
    target = float(saved["generator"]["skill_norm.mean"][0])

    def bent(knee: float) -> dict[str, float]:
        posture = dict(nk.NOMINAL_STAND_JOINT_POS)
        for side in ("L", "R"):
            posture[f"{side}HipPitch"] = -knee / 2
            posture[f"{side}KneePitch"] = knee
            posture[f"{side}AnklePitch"] = -knee / 2
        return posture

    low, high = 0.0, 2.0
    for _ in range(40):
        middle = (low + high) / 2
        height = nk.com_height_above_soles(bent(middle)) / NAO.leg_length
        low, high = (middle, high) if height > target else (low, middle)
    return bent((low + high) / 2)


def nao_body(contract: Path) -> dict:
    posture = nao_crouch(contract)
    stiffness = ast.literal_eval(_assigned_literal(NAO_SOURCE, "NAO_STIFFNESS"))
    damping = ast.literal_eval(_assigned_literal(NAO_SOURCE, "NAO_DAMPING"))
    urdf = {j.name: j for j in nk.load_model().joints}
    poses = nk.forward_kinematics(posture)
    point = {role: poses[link][:3, 3] for role, link in NAO_LINKS.items()}
    thigh = _sagittal(point["knee"] - point["hip"])
    shank = _sagittal(point["ankle"] - point["knee"])
    # The planar chain must be the URDF's: check bent cases against forward kinematics.
    for d_hip, d_knee in ((-0.1, 0.2), (0.05, 0.1)):
        bent = dict(posture)
        bent["LHipPitch"] += d_hip
        bent["LKneePitch"] += d_knee
        bent["LAnklePitch"] -= d_hip + d_knee
        sole = nk.forward_kinematics(bent)["l_sole"][2, 3] - poses["l_sole"][2, 3]
        planar = ankle_rise(thigh, shank, d_hip, d_knee)
        if abs(sole - planar) > 1e-6:
            raise ValueError(f"the planar NAO leg disagrees with the URDF: {sole} vs {planar}")
    joints = {}
    for role, name in NAO_JOINTS.items():
        j = urdf[name]
        joints[role] = {
            "joint": name,
            "q0": posture[name],
            "lower": j.lower,
            "upper": j.upper,
            "effort": j.effort,
            "velocity": j.velocity,
            "stiffness": stiffness[name],
            "damping": damping[name],
            "armature": 0.0,
            "swing_inertia": nk.composite_inertia_about_joint(name, posture),
        }
    return {
        "leg_length": float(NAO.leg_length),
        "thigh": thigh,
        "shank": shank,
        "joints": joints,
        "knees": KNEES["nao"],
    }


def _matrix(quat_wxyz) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def g1_body() -> dict:
    """The G1's left sagittal leg from its measured body card, at the teacher's default."""
    card = json.loads(G1_CARD.read_text(encoding="utf-8"))
    posture = card["postures"]["teacher_default"]
    index = {name: i for i, name in enumerate(card["bodies"]["names"])}
    drives = g1_actuators()
    joints, point = {}, {}
    for role, joint in G1_JOINTS.items():
        link = index[f"{joint}_link"]
        origin = np.array(posture["link_pos"][link])
        axis = _matrix(posture["link_quat_wxyz"][link]) @ np.array([0.0, 1.0, 0.0])
        if axis[1] < 0.95:
            raise ValueError(f"the G1 {joint} axis {axis} is not +y in the default posture")
        inertia = 0.0
        for part in G1_DISTAL[role]:
            i = index[f"left_{part}_link"]
            rotation = _matrix(posture["link_quat_wxyz"][i])
            own = rotation @ np.array(card["bodies"]["inertia_link"][i]) @ rotation.T
            lever = np.array(posture["com_pos"][i]) - origin
            lever -= axis * (axis @ lever)
            inertia += float(axis @ own @ axis) + card["bodies"]["mass"][i] * float(lever @ lever)
        k = card["joints"]["names"].index(f"{joint}_joint")
        joints[role] = {
            "joint": f"{joint}_joint",
            "q0": card["joints"]["default_pos"][k],
            "lower": card["joints"]["pos_limits"][k][0],
            "upper": card["joints"]["pos_limits"][k][1],
            "effort": card["joints"]["effort_limits"][k],
            "velocity": card["joints"]["vel_limits"][k],
            **drives[role],
            "swing_inertia": inertia + drives[role]["armature"],
        }
        point[role] = origin
    return {
        "leg_length": float(card["leg_length"]),
        "thigh": _sagittal(point["knee"] - point["hip"]),
        "shank": _sagittal(point["ankle"] - point["knee"]),
        "joints": joints,
        "knees": KNEES["g1"],
        "actuator_source": drives["source"],
    }


def drive_in_swing(joint: dict, time_unit: float) -> dict:
    k, d, inertia = joint["stiffness"], joint["damping"], joint["swing_inertia"]
    omega = math.sqrt(k / inertia)
    zeta = d / (2 * math.sqrt(k * inertia))
    response = 1 / (omega * (zeta - math.sqrt(zeta**2 - 1))) if zeta > 1 else 1 / omega
    return {
        "natural_frequency_rad_s": omega,
        "damping_ratio": zeta,
        "response_time_s": response,
        "response_time_T": response / time_unit,
        "effort_over_inertia_rad_s2": joint["effort"] / inertia,
    }


def _room(joint: dict, kind: str) -> tuple[float, float]:
    lower, upper = joint["lower"], joint["upper"]
    if kind == "soft":
        middle, half = (lower + upper) / 2, SOFT_FACTOR * (upper - lower) / 2
        lower, upper = middle - half, middle + half
    return lower - joint["q0"], upper - joint["q0"]


def lift_ceiling(body: dict, duration: float | None, variant: dict) -> float:
    """Highest flat-foot rise of the sole, in metres, within ``duration`` (None: no time)."""
    bounds = {}
    for role in JOINTS:
        joint = body["joints"][role]
        low, high = _room(joint, variant["range"])
        if duration is not None:
            alpha = joint["effort"] / joint["swing_inertia"]
            velocity = joint["velocity"] if variant["velocity"] else math.inf
            reach = excursion_ceiling(alpha, velocity, duration)
            low, high = max(low, -reach), min(high, reach)
        bounds[role] = (low, high)
    if any(low > 0 or high < 0 for low, high in bounds.values()):
        return 0.0
    # The ankle takes -(hip + knee), so for each hip excursion the knee's interval is its
    # own cut by the ankle's. Sampling that interval, ends included, keeps the grid on the
    # boundary where the highest rise lies; a square grid lands on it at points that move
    # with the bounds, and could lower the ceiling when a limit is removed.
    (hip_low, hip_high), (knee_low, knee_high), (ankle_low, ankle_high) = (
        bounds[role] for role in JOINTS
    )
    d_hip = np.linspace(hip_low, hip_high, GRID)[:, None]
    low = np.maximum(knee_low, -ankle_high - d_hip)
    high = np.minimum(knee_high, -ankle_low - d_hip)
    d_knee = low + (high - low) * np.linspace(0.0, 1.0, GRID)[None, :]
    feasible = np.broadcast_to(high >= low, d_knee.shape)
    rise = ankle_rise(body["thigh"], body["shank"], d_hip, d_knee)
    return float(max(0.0, rise[feasible].max())) if feasible.any() else 0.0


def capture_swings(
    path: Path, knees: tuple[str, str]
) -> tuple[list[dict], float, np.ndarray, np.ndarray]:
    """The planned swings, the control step, both knees' signed velocity and the body's
    contacts, aligned with the swings."""
    data = np.load(path)
    if "probes" in data.files:
        probes = json.loads(str(data["probes"]))
        keep = np.isin(
            data["probe_index"],
            [i for i, p in enumerate(probes) if p.get("group") in ("grid", "corner")],
        )
        probe_of_env = data["probe_index"][keep]
    else:
        keep = np.ones(data["skill_state"].shape[1], dtype=bool)
        probe_of_env = np.zeros(int(keep.sum()), dtype=int)
    z = data["skill_state"][1:, keep].astype(float)
    due = data["contract_due"][:-1, keep].astype(float)
    fields = data["contract_fields"][:-1, keep].astype(float)
    done = data["done"][:, keep].astype(bool)
    valid = ~(done[:-1] | done[1:])
    valid[:20] = False
    names = [str(n) for n in data["joint_names"]]
    velocity = data["encoder_input"][1:, keep][..., 9 + len(names) : 9 + 2 * len(names)]
    knee_velocity = velocity[..., [names.index(k) for k in knees]].astype(float)
    contact = SKILL_LAYOUT.unpack(z)["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    rows = cm.swings(z, due, fields, valid, probe_of_env)
    return rows, float(data["control_dt"]), knee_velocity, contact


def _liftoff_speed(r: dict, knee_velocity: np.ndarray, contact: np.ndarray) -> float:
    """The swing knee's velocity at the body's first airborne sample of the swing's window
    (``clearance_mechanism.swings`` widens it by two samples each way); NaN if it never
    leaves the ground."""
    window = np.arange(r["start"] - 2, r["start"] + r["length"] + 2)
    air = ~contact[window, r["env"], r["side"]]
    if not air.any():
        return math.nan
    return float(knee_velocity[window[np.argmax(air)], r["env"], r["side"]])


def against_plans(body: dict, paths: list[Path], unit: float) -> dict:
    rows, steps = [], set()
    cap = body["joints"]["knee"]["velocity"]
    for path in paths:
        swings, dt, knee_velocity, contact = capture_swings(path, body["knees"])
        for r in swings:
            window = knee_velocity[r["start"] : r["start"] + r["length"], r["env"], r["side"]]
            asked = r["asked"] * body["leg_length"]
            reached = r["reached"] * body["leg_length"]
            liftoff = _liftoff_speed(r, knee_velocity, contact)
            rows.append(
                (r["length"], r["ratio"], asked, dt, float(np.abs(window).max()), reached, liftoff)
            )
        steps |= {(r["length"] + e, dt) for r in swings for e in EXTRA_STEPS}
    table = {
        name: {key: lift_ceiling(body, key[0] * key[1], variant) for key in sorted(steps)}
        for name, variant in VARIANTS.items()
    }

    def summary(members) -> dict:
        out = {
            "swings": len(members),
            "reached_ratio_median": float(np.median([r[1] for r in members])),
            "asked_L_median": float(np.median([r[2] for r in members]) / body["leg_length"]),
        }
        for name in VARIANTS:
            for extra in EXTRA_STEPS:
                ceiling = np.array([table[name][(r[0] + extra, r[3])] for r in members])
                values = ceiling / np.maximum([r[2] for r in members], 1e-6)
                out[f"ceiling_{name}_plus{extra}_median"] = float(np.median(values))
                if extra == 0:
                    out[f"share_asked_above_ceiling_{name}"] = float(np.mean(values < 1.0))
                # A body that beats a ceiling is not bound by it: its swings break an
                # assumption of the ceiling (for one, a leg at rest at lift-off).
                out[f"share_reached_above_ceiling_{name}_plus{extra}"] = float(
                    np.mean(np.array([r[5] for r in members]) > ceiling)
                )
        peak = np.array([r[4] for r in members])
        reached = np.array([r[1] for r in members])
        liftoff = np.array([r[6] for r in members])
        at_cap = peak >= AT_CAP * cap
        out.update(
            {
                "knee_speed_at_liftoff_median_rad_s": float(np.nanmedian(liftoff)),
                "knee_speed_at_liftoff_median_over_limit": float(np.nanmedian(liftoff) / cap),
                "knee_peak_speed_median_rad_s": float(np.median(peak)),
                "knee_peak_speed_median_rad_per_T": float(np.median(peak) * unit),
                # Above the limit only if the simulator does not enforce it.
                "knee_peak_speed_max_rad_s": float(peak.max()),
                "share_knee_at_velocity_limit": float(at_cap.mean()),
                "reached_ratio_median_knee_at_limit": (
                    float(np.median(reached[at_cap])) if at_cap.any() else None
                ),
                "reached_ratio_median_knee_below_limit": (
                    float(np.median(reached[~at_cap])) if (~at_cap).any() else None
                ),
            }
        )
        return out

    by_length = {}
    for n in sorted({r[0] for r in rows}):
        members = [r for r in rows if r[0] == n]
        if len(members) >= 50:
            by_length[str(n)] = summary(members)
    return {
        "captures": [p.resolve().relative_to(REPO_ROOT).as_posix() for p in paths],
        **summary(rows),
        "by_swing_length": by_length,
    }


def _print_lengths(plan: dict) -> None:
    for n, b in plan["by_swing_length"].items():
        print(
            f"    {n:>2s} steps ({b['swings']:6d}): asked {b['asked_L_median']:.3f} L, "
            f"reached {b['reached_ratio_median']:.2f}; ceiling hard "
            f"{b['ceiling_hard_plus0_median']:.2f} (+1 step "
            f"{b['ceiling_hard_plus1_median']:.2f}), soft {b['ceiling_soft_plus0_median']:.2f}"
            f", no velocity cap {b['ceiling_hard_no_velocity_cap_plus0_median']:.2f}; knee at "
            f"its limit {b['share_knee_at_velocity_limit']:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nao-contract", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.capture) + [p for g in args.glob for p in sorted(REPO_ROOT.glob(g))]
    legs = {p: float(np.load(p)["leg_length"]) for p in paths}
    report = {}
    for name, body in (("nao", nao_body(args.nao_contract)), ("g1", g1_body())):
        unit = math.sqrt(body["leg_length"] / nk.GRAVITY)
        mine = [p for p, leg in legs.items() if abs(leg - body["leg_length"]) < 1e-3]
        # Probe captures (the family's map and corners) and task captures (the skills' L4)
        # apart: W6's target is read on the second.
        groups = {
            "probes": [p for p in mine if "probes" in np.load(p).files],
            "tasks": [p for p in mine if "probes" not in np.load(p).files],
        }
        plans = {g: against_plans(body, members, unit) for g, members in groups.items() if members}
        ranges = {
            v: lift_ceiling(body, None, VARIANTS[v]) / body["leg_length"] for v in ("hard", "soft")
        }
        report[name] = entry = {
            "time_unit_s": unit,
            "joints": body["joints"],
            "in_swing": {r: drive_in_swing(body["joints"][r], unit) for r in JOINTS},
            "range_only_lift_ceiling_L": ranges,
            "against_plans": plans,
        }
        print(f"{name}: T {unit * 1e3:.0f} ms")
        for role in JOINTS:
            s, j = entry["in_swing"][role], body["joints"][role]
            print(
                f"  {role:5s} q0 {j['q0']:+.3f} in [{j['lower']:+.3f}, {j['upper']:+.3f}]; "
                f"zeta {s['damping_ratio']:.2f}, response {s['response_time_s'] * 1e3:.0f}"
                f" ms = {s['response_time_T']:.2f} T; effort/I "
                f"{s['effort_over_inertia_rad_s2']:.0f} rad/s^2"
            )
        print(
            f"  lift within the ranges alone: hard {ranges['hard']:.3f} L, "
            f"soft {ranges['soft']:.3f} L"
        )
        cap = body["joints"]["knee"]["velocity"]
        for group, plan in plans.items():
            print(
                f"  {group}: {plan['swings']} swings, reached {plan['reached_ratio_median']:.2f}, "
                f"ceiling {plan['ceiling_hard_plus0_median']:.2f} (+1 step "
                f"{plan['ceiling_hard_plus1_median']:.2f}, no velocity cap "
                f"{plan['ceiling_hard_no_velocity_cap_plus0_median']:.2f}), asked above it "
                f"{plan['share_asked_above_ceiling_hard']:.2f}; knee limit {cap:.2f} rad/s = "
                f"{cap * unit:.2f} rad/T, peak median "
                f"{plan['knee_peak_speed_median_rad_per_T']:.2f} rad/T, at the limit in "
                f"{plan['share_knee_at_velocity_limit']:.2f} of swings"
            )
            print(
                "    reached above the ceiling in "
                f"{plan['share_reached_above_ceiling_hard_plus0']:.3f} of swings (+1 step "
                f"{plan['share_reached_above_ceiling_hard_plus1']:.3f}); "
                f"knee at lift-off {plan['knee_speed_at_liftoff_median_rad_s']:+.2f} rad/s = "
                f"{plan['knee_speed_at_liftoff_median_over_limit']:+.2f} of its limit"
            )
            _print_lengths(plan)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
