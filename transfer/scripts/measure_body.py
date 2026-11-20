"""Measure any registered body in the simulator and write its body card.

Generalises ``g1/scripts/measure_body.py`` to every body in
``transfer/bodies/bodies.json``, and adds what a body descriptor needs beyond the
units: the kinematic tree, each joint's axis and position in the zero pose, the
drives as the simulation applies them, and the control period.

What is measured:

``mass``, ``inertia``, ``leg_length``, ``foot``, ``postures``
    As for the G1 (see that script): the leg length is the hip-pitch joint to the
    sole with the legs at the zero pose, the sole is the centre of the bottom of the
    foot's collision bounding box.
``tree``
    Each link's parent, from PhysX's articulation metatype.
``joint_geometry``
    For every degree of freedom, its child link, the parent link, the joint's
    position (the child link's origin) and its axis, both in the root frame of the
    zero pose. The axis is measured by turning the joint by a small angle and
    reading the child link's rotation, so it is what PhysX integrates whatever the
    USD's joint frames say.
``drives``
    Per joint: actuator class, stiffness, damping, armature, friction, effort and
    velocity limits, as the actuator models hold them.
``control``
    Physics step, decimation, control period, and the action term's scale and
    offset convention.

Usage::

    python transfer/scripts/measure_body.py --body h1 --headless
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure a registered body and write its card.")
parser.add_argument("--body", required=True, help="A key of transfer/bodies/bodies.json.")
parser.add_argument(
    "--registry", type=Path, default=_REPO_ROOT / "transfer" / "bodies" / "bodies.json"
)
parser.add_argument("--output", type=Path, default=None, help="Defaults to the registry's card.")
parser.add_argument(
    "--turn", type=float, default=0.05, help="Joint turn (rad) used to measure each axis."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_transfer.tasks  # noqa: F401  (registers this extension's task ids)
import isaaclab_tasks  # noqa: F401  (registers Isaac Lab's own task ids)

ROOT_HEIGHT = 1.0
"""Height the root is placed at. Nothing is simulated; every result is relative."""

MOVED = 1e-7
"""A link whose position (m) or rotation (rad) changes by more than this has moved."""


def _pose(env, robot, joint_pos: torch.Tensor) -> dict[str, np.ndarray]:
    """Place the root upright at a fixed point, set the joints, and read every body."""
    device = robot.device
    root = torch.zeros((1, 7), device=device)
    root[:, 2] = ROOT_HEIGHT
    root[:, 3] = 1.0
    robot.write_root_pose_to_sim(root)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    env.sim.physics_sim_view.update_articulations_kinematic()
    robot.update(env.physics_dt)

    data = robot.data
    offset = np.array([0.0, 0.0, ROOT_HEIGHT])
    return {
        "link_pos": data.body_link_pos_w[0].cpu().numpy().astype(float) - offset,
        "link_quat_wxyz": data.body_link_quat_w[0].cpu().numpy().astype(float),
        "com_pos": data.body_com_pos_w[0].cpu().numpy().astype(float) - offset,
    }


def _foot_geometry(robot_prim_path: str, link_name: str) -> dict:
    """Bounding box of a foot's collision prims, in the foot link's own frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)
    link = next(
        (p for p in Usd.PrimRange(root) if p.GetName() == link_name and p.IsA(UsdGeom.Xformable)),
        None,
    )
    if link is None:
        raise RuntimeError(f"no prim named {link_name} under {robot_prim_path}")

    # Instanceable assets (H1, Cassie, the converted NAO) keep their collision prims
    # behind instance proxies, which a plain PrimRange does not enter.
    collisions = [
        p
        for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
    ]
    if not collisions:
        raise RuntimeError(f"{link_name} has no collision geometry")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    )
    lows, highs = [], []
    for prim in collisions:
        box = cache.ComputeRelativeBound(prim, link).ComputeAlignedRange()
        lows.append(np.array(box.GetMin()))
        highs.append(np.array(box.GetMax()))
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    return {
        "collision_prims": [str(p.GetPath()).rsplit("/", 1)[-1] for p in collisions],
        "collision_types": sorted({p.GetTypeName() for p in collisions}),
        "bbox_min_link": low.round(6).tolist(),
        "bbox_max_link": high.round(6).tolist(),
        "sole_point_link": [
            round(float((low[0] + high[0]) / 2), 6),
            round(float((low[1] + high[1]) / 2), 6),
            round(float(low[2]), 6),
        ],
        "length": round(float(high[0] - low[0]), 6),
        "width": round(float(high[1] - low[1]), 6),
    }


def _foot_lowest_face(robot_prim_path: str, link_name: str, rotation: np.ndarray) -> dict:
    """A foot whose sole is not the link's lowest z (Digit): the collision geometry from its
    own schema attributes (never an authored ``extent``, which an asset may store already
    scaled), and the sole as the face lowest in the world with the link at ``rotation``
    (its default-pose orientation). Registry entries opt in with ``"sole": "lowest_face"``.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_prim_path)
    link = next(
        (p for p in Usd.PrimRange(root) if p.GetName() == link_name and p.IsA(UsdGeom.Xformable)),
        None,
    )
    if link is None:
        raise RuntimeError(f"no prim named {link_name} under {robot_prim_path}")
    collisions = [
        p
        for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
    ]
    if not collisions:
        raise RuntimeError(f"{link_name} has no collision geometry")
    time = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(time)
    corners = []
    for prim in collisions:
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), time)
        lo, hi = np.array(extent[0], dtype=float), np.array(extent[1], dtype=float)
        local = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        matrix, _ = cache.ComputeRelativeTransform(prim, link)
        homogeneous = np.c_[local, np.ones(len(local))] @ np.array(matrix, dtype=float)
        corners.append(homogeneous[:, :3])
    points = np.concatenate(corners)
    world = points @ rotation.T
    lowest = world[:, 2].min()
    sole = points[world[:, 2] < lowest + 1e-4]
    sole_world = sole @ rotation.T
    low, high = points.min(axis=0), points.max(axis=0)
    return {
        "collision_prims": [str(p.GetPath()).rsplit("/", 1)[-1] for p in collisions],
        "collision_types": sorted({p.GetTypeName() for p in collisions}),
        "sole_rule": "lowest_face",
        "bbox_min_link": low.round(6).tolist(),
        "bbox_max_link": high.round(6).tolist(),
        "sole_corners_link": sole.round(6).tolist(),
        "sole_point_link": sole.mean(axis=0).round(6).tolist(),
        "length": round(float(np.ptp(sole_world[:, 0])), 6),
        "width": round(float(np.ptp(sole_world[:, 1])), 6),
    }


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The axis-angle vector of a rotation matrix (angle below pi)."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return skew * angle / (2.0 * np.sin(angle))


def _joint_geometry(env, robot, zero_q, zero, names, parents, turn: float) -> list[dict]:
    """Child link, parent link, position and axis of every degree of freedom.

    A joint's child link is the moved link whose parent did not move when only that
    joint turned; the axis is the child's rotation divided by the turn.
    """
    parent_of = dict(zip(names, parents, strict=True))
    index = {name: i for i, name in enumerate(names)}
    rotations = [_quat_to_matrix(q) for q in zero["link_quat_wxyz"]]
    geometry = []
    for dof, dof_name in enumerate(robot.joint_names):
        q = zero_q.clone()
        q[0, dof] += turn
        turned = _pose(env, robot, q)
        moved = set()
        for i, name in enumerate(names):
            shift = np.linalg.norm(turned["link_pos"][i] - zero["link_pos"][i])
            spin = np.linalg.norm(
                _rotation_vector(_quat_to_matrix(turned["link_quat_wxyz"][i]) @ rotations[i].T)
            )
            if shift > MOVED or spin > MOVED:
                moved.add(name)
        roots = [n for n in moved if parent_of[n] not in moved]
        if len(roots) != 1:
            raise RuntimeError(f"joint {dof_name}: expected one child link, got {sorted(roots)}")
        child = roots[0]
        c = index[child]
        relative = _quat_to_matrix(turned["link_quat_wxyz"][c]) @ rotations[c].T
        rotation_vector = _rotation_vector(relative)
        angle = float(np.linalg.norm(rotation_vector))
        if angle > 0.5 * turn:
            kind, axis = "revolute", rotation_vector / turn
        else:
            translation = turned["link_pos"][c] - zero["link_pos"][c]
            kind, axis = "prismatic", translation / turn
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        geometry.append(
            {
                "joint": dof_name,
                "kind": kind,
                "child_link": child,
                "parent_link": parent_of[child],
                "position_root": zero["link_pos"][c].round(6).tolist(),
                "axis_root": axis.round(6).tolist(),
                "turn_measured": round(angle, 6),
            }
        )
    _pose(env, robot, zero_q)
    return geometry


def _drives(robot) -> list[dict]:
    """Per joint, the actuator model and its parameters as the simulation holds them."""
    joint_names = list(robot.joint_names)
    rows: list[dict | None] = [None] * len(joint_names)

    def value(tensor, k):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            flat = tensor[0] if tensor.ndim == 2 else tensor
            return round(float(flat[k] if flat.numel() > 1 else flat), 8)
        return round(float(tensor), 8)

    for name, actuator in robot.actuators.items():
        indices = actuator.joint_indices
        if isinstance(indices, slice):
            indices = list(range(len(joint_names)))[indices]
        indices = [int(i) for i in (indices.tolist() if hasattr(indices, "tolist") else indices)]
        for k, joint in enumerate(indices):
            rows[joint] = {
                "joint": joint_names[joint],
                "actuator_group": name,
                "actuator_class": type(actuator).__name__,
                "stiffness": value(actuator.stiffness, k),
                "damping": value(actuator.damping, k),
                "armature": value(getattr(actuator, "armature", None), k),
                "friction": value(getattr(actuator, "friction", None), k),
                "effort_limit": value(actuator.effort_limit, k),
                "velocity_limit": value(actuator.velocity_limit, k),
                "saturation_effort": value(getattr(actuator, "saturation_effort", None), k),
            }
    missing = [joint_names[i] for i, row in enumerate(rows) if row is None]
    if missing:
        raise RuntimeError(f"joints without an actuator model: {missing}")
    return rows


def _control(cfg) -> dict:
    """Physics step, decimation and the joint action's scale and offset convention."""
    out = {
        "physics_dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "control_period": float(cfg.sim.dt * cfg.decimation),
    }
    actions = getattr(cfg, "actions", None)
    term = getattr(actions, "joint_pos", None) if actions is not None else None
    if term is not None:
        scale = getattr(term, "scale", None)
        out["action_scale"] = scale if isinstance(scale, (int, float)) else str(scale)
        out["action_use_default_offset"] = bool(getattr(term, "use_default_offset", False))
        out["action_term"] = type(term).__name__
    return out


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    registry = json.loads(args_cli.registry.read_text("utf-8"))
    if args_cli.body not in registry:
        raise SystemExit(f"{args_cli.body!r} is not in {args_cli.registry}")
    entry = registry[args_cli.body]
    output = args_cli.output or (_REPO_ROOT / entry["card"])
    task_id = entry["measure_task"]

    cfg = parse_env_cfg(task_id, num_envs=1)
    env = gym.make(task_id, cfg=cfg)
    inner = env.unwrapped
    robot = inner.scene["robot"]
    env.reset()

    names = list(robot.body_names)
    metatype = robot.root_physx_view.shared_metatype
    parents = list(metatype.link_parents)
    if list(metatype.link_names) != names:
        raise RuntimeError("PhysX link order differs from Isaac Lab's body order")
    index = {name: i for i, name in enumerate(names)}
    base, feet = entry["base_body"], tuple(entry["foot_bodies"])
    for required in (base, *feet):
        if required not in index:
            raise RuntimeError(f"body {required!r} not found; bodies are {names}")
    joint_names = list(robot.joint_names)
    for required in (*entry["hip_pitch_joints"], *entry["hip_roll_joints"]):
        if required not in joint_names:
            raise RuntimeError(f"joint {required!r} not found; joints are {joint_names}")

    view = robot.root_physx_view
    masses = view.get_masses()[0].cpu().numpy().astype(float)
    inertias = view.get_inertias()[0].cpu().numpy().astype(float).reshape(-1, 3, 3)
    com_pos_b = robot.data.body_com_pos_b[0].cpu().numpy().astype(float)

    zero_q = torch.zeros_like(robot.data.default_joint_pos)
    zero = _pose(inner, robot, zero_q)
    geometry = _joint_geometry(inner, robot, zero_q, zero, names, parents, args_cli.turn)
    default = _pose(inner, robot, robot.data.default_joint_pos.clone())
    zero = _pose(inner, robot, zero_q)

    robot_prim = cfg.scene.robot.prim_path.replace("env_.*", "env_0")
    if entry.get("sole") == "lowest_face":
        foot_geometry = {
            name: _foot_lowest_face(
                robot_prim, name, _quat_to_matrix(default["link_quat_wxyz"][index[name]])
            )
            for name in feet
        }
    else:
        foot_geometry = {name: _foot_geometry(robot_prim, name) for name in feet}

    def sole_world(posture: dict, foot: str) -> np.ndarray:
        rotation = _quat_to_matrix(posture["link_quat_wxyz"][index[foot]])
        return posture["link_pos"][index[foot]] + rotation @ np.array(
            foot_geometry[foot]["sole_point_link"]
        )

    def com_of(posture: dict) -> np.ndarray:
        return (masses[:, None] * posture["com_pos"]).sum(axis=0) / masses.sum()

    child_of = {row["joint"]: row["child_link"] for row in geometry}
    legs = []
    for pitch, roll, foot in zip(
        entry["hip_pitch_joints"], entry["hip_roll_joints"], feet, strict=True
    ):
        sole = sole_world(zero, foot)
        legs.append(
            {
                "foot": foot,
                "hip_pitch_joint": pitch,
                "hip_pitch_to_sole": float(
                    np.linalg.norm(zero["link_pos"][index[child_of[pitch]]] - sole)
                ),
                "hip_roll_to_sole": float(
                    np.linalg.norm(zero["link_pos"][index[child_of[roll]]] - sole)
                ),
            }
        )
    leg_length = float(np.mean([leg["hip_pitch_to_sole"] for leg in legs]))

    def posture_summary(posture: dict) -> dict:
        soles = np.stack([sole_world(posture, foot) for foot in feet])
        com = com_of(posture)
        return {
            "com_height_above_soles": float(com[2] - soles[:, 2].min()),
            "sole_lateral_separation": float(abs(soles[0, 1] - soles[1, 1])),
            "soles": soles.round(6).tolist(),
            "com": com.round(6).tolist(),
            "link_pos": posture["link_pos"].round(6).tolist(),
            "link_quat_wxyz": posture["link_quat_wxyz"].round(6).tolist(),
            "com_pos": posture["com_pos"].round(6).tolist(),
        }

    try:
        from importlib.metadata import version

        isaaclab_version = version("isaaclab")
    except Exception:  # pragma: no cover - depends on how Isaac Lab was installed
        isaaclab_version = "unknown"

    spawn = cfg.scene.robot.spawn
    card = {
        "$comment": [
            f"The body '{args_cli.body}' as its simulation integrates it.",
            "Written by transfer/scripts/measure_body.py from transfer/bodies/bodies.json.",
            "Lengths in metres, masses in kg, SI. Postures and joint geometry are relative",
            "to a root placed upright at the origin, in the zero pose unless named.",
        ],
        "provenance": {
            "script": "transfer/scripts/measure_body.py",
            "task_id": task_id,
            "asset": getattr(spawn, "usd_path", None) or getattr(spawn, "asset_path", None),
            "isaaclab": isaaclab_version,
            "measured_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "body": args_cli.body,
        "base_body": base,
        "foot_bodies": list(feet),
        "total_mass": float(masses.sum()),
        "leg_length": leg_length,
        "legs": legs,
        "foot": foot_geometry,
        "bodies": {
            "names": names,
            "parents": parents,
            "mass": masses.round(6).tolist(),
            "inertia_link": inertias.round(8).tolist(),
            "com_offset_link": com_pos_b.round(6).tolist(),
        },
        "joints": {
            "names": joint_names,
            "pos_limits": robot.data.joint_pos_limits[0].cpu().numpy().round(6).tolist(),
            "vel_limits": robot.data.joint_vel_limits[0].cpu().numpy().round(6).tolist(),
            "effort_limits": robot.data.joint_effort_limits[0].cpu().numpy().round(6).tolist(),
            "default_pos": robot.data.default_joint_pos[0].cpu().numpy().round(6).tolist(),
        },
        "joint_geometry": geometry,
        "drives": _drives(robot),
        "control": _control(cfg),
        "postures": {
            "zero": posture_summary(zero),
            "default": posture_summary(default),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"BODY MEASURED: {args_cli.body}")
    print("=" * 78)
    print(f"  file                    : {output}")
    print(f"  bodies / joints         : {len(names)} / {len(joint_names)}")
    print(f"  total mass              : {masses.sum():.4f} kg")
    for leg in legs:
        print(
            f"  {leg['foot']:<24}: hip-pitch->sole {leg['hip_pitch_to_sole']:.4f} m, "
            f"hip-roll->sole {leg['hip_roll_to_sole']:.4f} m"
        )
    print(f"  leg length L            : {leg_length:.4f} m")
    print(f"  control period          : {card['control']['control_period']:.4f} s")
    for name, foot in foot_geometry.items():
        print(
            f"  {name:<24}: {foot['collision_types']} length {foot['length']:.4f} "
            f"width {foot['width']:.4f} sole z {foot['sole_point_link'][2]:+.4f}"
        )
    for label in ("zero", "default"):
        summary = card["postures"][label]
        print(
            f"  {label:<24}: CoM above soles {summary['com_height_above_soles']:.4f} m, "
            f"soles {summary['sole_lateral_separation']:.4f} m apart"
        )
    print("=" * 78 + "\n")

    env.close()
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close(skip_cleanup=True)
    raise SystemExit(exit_code)
