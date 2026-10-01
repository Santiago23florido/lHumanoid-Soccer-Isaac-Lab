"""Measure the body the teacher was trained in, and write it down.

The NAO's numbers come from a URDF this repository vendors, so a test can
re-derive every one of them. The G1 arrives as a USD downloaded from Nucleus at
run time, so its numbers can only be read out of a running simulator. This
script does that once and records the result in ``g1/body/body_card.json``,
which the transfer package then restates and a test checks against.

What is measured, and why each is needed by the embodiment-agnostic skill state
and the dimensionless groups that describe a body:

``mass`` and per-body inertia
    The mass unit, and the composite inertia ``I_c / (m L^2)``. Read from the
    PhysX view, which is what the simulation actually integrates, not from the
    configuration.
``leg_length``
    The length unit ``L``: hip-pitch joint to sole with the leg straight. A
    property of the body alone, so the posture a task chooses (how much it
    crouches) stays in the state as ``c_z / L`` instead of hiding in the unit.
``foot``
    The sole point in the foot frame and the foot's extent, from the bounding
    box of its collision prims -- what the ground actually touches. The prim
    types are recorded, because a box around spheres is not a support polygon.
``postures``
    Every body's pose in two postures: the zero pose (legs straight, arms down)
    that makes posture-dependent groups comparable across robots, and the
    teacher's default pose, which is where it actually stands.

Usage::

    python g1/scripts/measure_body.py --headless
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

parser = argparse.ArgumentParser(description="Measure the G1 body the teacher was trained in.")
parser.add_argument(
    "--output",
    type=Path,
    default=_REPO_ROOT / "g1" / "body" / "body_card.json",
    help="Where to write the body card.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_transfer.tasks  # noqa: F401  (registers the task ids)
from humanoid_transfer.g1.tasks.g1_walk import PLAY_TASK_ID

BASE_BODY = "pelvis"
"""The body carrying both hip joints: the base of the skill state's heading frame."""

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
HIP_PITCH_BODIES = ("left_hip_pitch_link", "right_hip_pitch_link")
"""Child links of the hip-pitch joints. A child link's origin is its joint's origin."""

HIP_ROLL_BODIES = ("left_hip_roll_link", "right_hip_roll_link")
"""Recorded so the sensitivity of ``L`` to which hip axis is used can be reported:
the G1's three hip axes do not meet at one point, unlike the NAO's."""

ROOT_HEIGHT = 1.0
"""Height the root is placed at for kinematic measurement. Arbitrary: nothing is
simulated, the articulation is only posed, and every result is relative."""


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
    return {
        "link_pos": data.body_link_pos_w[0].cpu().numpy() - np.array([0.0, 0.0, ROOT_HEIGHT]),
        "link_quat_wxyz": data.body_link_quat_w[0].cpu().numpy(),
        "com_pos": data.body_com_pos_w[0].cpu().numpy() - np.array([0.0, 0.0, ROOT_HEIGHT]),
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

    collisions = [
        p for p in Usd.PrimRange(link) if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim)
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


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(PLAY_TASK_ID, num_envs=1)
    env = gym.make(PLAY_TASK_ID, cfg=cfg)
    inner = env.unwrapped
    robot = inner.scene["robot"]
    env.reset()

    names = list(robot.body_names)
    index = {name: i for i, name in enumerate(names)}
    for required in (BASE_BODY, *FOOT_BODIES, *HIP_PITCH_BODIES, *HIP_ROLL_BODIES):
        if required not in index:
            raise RuntimeError(f"body {required!r} not found; bodies are {names}")

    view = robot.root_physx_view
    masses = view.get_masses()[0].cpu().numpy().astype(float)
    inertias = view.get_inertias()[0].cpu().numpy().astype(float).reshape(-1, 3, 3)
    com_pos_b = robot.data.body_com_pos_b[0].cpu().numpy().astype(float)

    zero = _pose(inner, robot, torch.zeros_like(robot.data.default_joint_pos))
    default = _pose(inner, robot, robot.data.default_joint_pos.clone())

    robot_prim = cfg.scene.robot.prim_path.replace("env_.*", "env_0")
    feet = {name: _foot_geometry(robot_prim, name) for name in FOOT_BODIES}

    def sole_world(posture: dict, foot: str) -> np.ndarray:
        rotation = _quat_to_matrix(posture["link_quat_wxyz"][index[foot]])
        return posture["link_pos"][index[foot]] + rotation @ np.array(feet[foot]["sole_point_link"])

    def com_of(posture: dict) -> np.ndarray:
        return (masses[:, None] * posture["com_pos"]).sum(axis=0) / masses.sum()

    legs = []
    for hip_pitch, hip_roll, foot in zip(
        HIP_PITCH_BODIES, HIP_ROLL_BODIES, FOOT_BODIES, strict=True
    ):
        sole = sole_world(zero, foot)
        legs.append(
            {
                "foot": foot,
                "hip_pitch_to_sole": float(
                    np.linalg.norm(zero["link_pos"][index[hip_pitch]] - sole)
                ),
                "hip_roll_to_sole": float(np.linalg.norm(zero["link_pos"][index[hip_roll]] - sole)),
            }
        )
    leg_length = float(np.mean([leg["hip_pitch_to_sole"] for leg in legs]))

    def posture_summary(posture: dict) -> dict:
        soles = np.stack([sole_world(posture, foot) for foot in FOOT_BODIES])
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

    joint_names = list(robot.joint_names)
    limits = robot.data.joint_pos_limits[0].cpu().numpy()
    vel_limits = robot.data.joint_vel_limits[0].cpu().numpy()
    effort_limits = robot.data.joint_effort_limits[0].cpu().numpy()

    try:
        from importlib.metadata import version

        isaaclab_version = version("isaaclab")
    except Exception:  # pragma: no cover - depends on how Isaac Lab was installed
        isaaclab_version = "unknown"

    card = {
        "$comment": [
            "The source robot's body as the teacher's simulation integrates it.",
            "Written by g1/scripts/measure_body.py; restated, with this file named as",
            "the source, in humanoid_transfer.transfer.embodiment and checked against",
            "it by tests/test_embodiment.py. Lengths in metres, masses in kg, SI.",
            "Postures are relative to a root placed upright at the origin.",
        ],
        "provenance": {
            "script": "g1/scripts/measure_body.py",
            "task_id": PLAY_TASK_ID,
            "usd": cfg.scene.robot.spawn.usd_path,
            "isaaclab": isaaclab_version,
            "measured_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "base_body": BASE_BODY,
        "foot_bodies": list(FOOT_BODIES),
        "total_mass": float(masses.sum()),
        "leg_length": leg_length,
        "legs": legs,
        "foot": feet,
        "bodies": {
            "names": names,
            "mass": masses.round(6).tolist(),
            "inertia_link": inertias.round(8).tolist(),
            "com_offset_link": com_pos_b.round(6).tolist(),
        },
        "joints": {
            "names": joint_names,
            "pos_limits": limits.round(6).tolist(),
            "vel_limits": vel_limits.round(6).tolist(),
            "effort_limits": effort_limits.round(6).tolist(),
            "default_pos": robot.data.default_joint_pos[0].cpu().numpy().round(6).tolist(),
        },
        "postures": {
            "zero": posture_summary(zero),
            "teacher_default": posture_summary(default),
        },
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(card, indent=1) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("G1 BODY MEASURED")
    print("=" * 78)
    print(f"  file                    : {args_cli.output}")
    print(f"  bodies / joints         : {len(names)} / {len(joint_names)}")
    print(f"  total mass              : {masses.sum():.4f} kg")
    for leg in legs:
        print(
            f"  {leg['foot']:<24}: hip-pitch->sole {leg['hip_pitch_to_sole']:.4f} m, "
            f"hip-roll->sole {leg['hip_roll_to_sole']:.4f} m"
        )
    print(f"  leg length L            : {leg_length:.4f} m")
    for name, foot in feet.items():
        print(
            f"  {name:<24}: {foot['collision_types']} length {foot['length']:.4f} "
            f"width {foot['width']:.4f} sole z {foot['sole_point_link'][2]:+.4f}"
        )
    for label in ("zero", "teacher_default"):
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
