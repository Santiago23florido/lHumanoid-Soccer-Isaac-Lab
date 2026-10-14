"""A body's sagittal mirror, measured by its own kinematics.

A tracker that walks a mirror-symmetric family of plans on a mirror-symmetric
robot has an optimal policy that is equivariant under the mirror: what it does
for a plan reflected left to right is its action reflected. Training it so
(Mittal et al., 2024) needs the mirror in joint space -- which joint is each
joint's partner, and whether a positive angle on one is a positive angle on
the other. Names give the partner. The sign depends on how each joint's axis
was drawn in the robot's model, and is not guessed here.

Every joint is moved by ``±delta`` from the default pose, one environment per
move, and the links' positions are read in the root frame. Reflecting a
configuration (``y → −y``, every link swapped with its partner) must reproduce
the partner joint's move with one of the two signs; the sign with the smaller
residual is the mirror's, and the default pose must reflect onto itself. Both
residuals are written, so a joint the kinematics cannot tell apart (none moved
a link) shows as such.

Usage::

    python transfer/scripts/measure_mirror.py --headless --task-id Isaac-Velocity-Flat-G1-v0 \
        --prefix left_ right_ --output g1/body/mirror.json
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "source" / "humanoid_transfer"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="A body's sagittal mirror, from its kinematics.")
parser.add_argument("--task-id", required=True)
parser.add_argument(
    "--prefix",
    nargs=2,
    action="append",
    required=True,
    metavar=("LEFT", "RIGHT"),
    help="A pair of name prefixes naming a body's two sides; repeat for several.",
)
parser.add_argument("--delta", type=float, default=0.1, help="Joint move, radians.")
parser.add_argument("--max-joints", type=int, default=40)
parser.add_argument(
    "--mimic",
    nargs=3,
    action="append",
    default=[],
    metavar=("FOLLOWER", "LEADER", "MULTIPLIER"),
    help="A joint driven by another (the model's mimic), recorded for joint_mirror.",
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.utils.math as math_utils
import torch

import humanoid_transfer.tasks  # noqa: F401


def partner(name: str, prefixes: list[tuple[str, str]]) -> str:
    """The name of the other side's part: a side's prefix swapped, or itself on the midline."""
    for left, right in prefixes:
        if name.startswith(left):
            return right + name[len(left) :]
        if name.startswith(right):
            return left + name[len(right) :]
    return name


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    prefixes = [tuple(p) for p in args_cli.prefix]
    # One environment per move; a simulation app holds one environment, so it is made
    # with room for any humanoid's joints and the first 1 + 2 * joints are used.
    cfg = parse_env_cfg(args_cli.task_id, num_envs=1 + 2 * args_cli.max_joints)
    env = gym.make(args_cli.task_id, cfg=cfg)
    inner = env.unwrapped
    env.reset()
    robot = inner.scene["robot"]
    device = inner.device
    joints = list(robot.data.joint_names)
    if len(joints) > args_cli.max_joints:
        raise ValueError(f"{len(joints)} joints: raise --max-joints")
    n = inner.num_envs
    links = list(robot.data.body_names)
    joint_partner = [joints.index(partner(j, prefixes)) for j in joints]
    link_partner = [links.index(partner(b, prefixes)) for b in links]

    default = robot.data.default_joint_pos[:1].clone()
    position = default.expand(n, -1).clone()
    for j in range(len(joints)):
        position[1 + 2 * j, j] += args_cli.delta
        position[2 + 2 * j, j] -= args_cli.delta
    ids = torch.arange(n, device=device)
    pose = torch.zeros(n, 7, device=device)
    pose[:, :3] = inner.scene.env_origins + torch.tensor([0.0, 0.0, 1.5], device=device)
    pose[:, 3] = 1.0
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=ids)
    robot.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=ids)
    inner.sim.forward()
    view = robot.root_physx_view
    reached = view.get_dof_positions().to(device)
    transforms = view.get_link_transforms().to(device)
    # A link's origin lies on the axis of the joint that turns it, so the last link of a
    # chain (a foot about its ankle's roll) moves no origin; its centre of mass does.
    rotation = math_utils.convert_quat(transforms[..., 3:7], to="wxyz")
    centre = view.get_coms().to(device)[..., :3]
    masses = transforms[..., :3] + math_utils.quat_apply(rotation, centre)
    points = torch.cat([transforms[..., :3], masses], dim=1) - pose[:, None, :3]
    both = link_partner + [k + len(links) for k in link_partner]

    def reflected(p: torch.Tensor) -> torch.Tensor:
        return p[:, both] * torch.tensor([1.0, -1.0, 1.0], device=device)

    scale = float((points[0] - points[0].mean(0)).norm(dim=-1).mean())
    symmetric = float((reflected(points[:1]) - points[:1]).norm(dim=-1).max()) / scale
    report = {"task": args_cli.task_id, "delta": args_cli.delta, "joints": [], "links": links}
    report["default_pose_residual"] = symmetric
    report["mimic"] = {f: [leader, float(m)] for f, leader, m in args_cli.mimic}
    # Displacements from the default pose, not positions: the pose's own small asymmetry
    # (a few millimetres) cancels, and a joint that moves its links by one still shows.
    moves = points - points[:1]
    for j, name in enumerate(joints):
        k = joint_partner[j]
        moved = float(moves[1 + 2 * j].norm(dim=-1).max()) / scale
        target = reflected(moves[1 + 2 * j : 2 + 2 * j])[0]
        plus = float((moves[1 + 2 * k] - target).norm(dim=-1).max()) / scale
        minus = float((moves[2 + 2 * k] - target).norm(dim=-1).max()) / scale
        report["joints"].append(
            {
                "name": name,
                "partner": joints[k],
                "sign": 1 if plus <= minus else -1,
                "residual": min(plus, minus) / max(moved, 1e-9),
                "other_residual": max(plus, minus) / max(moved, 1e-9),
                "moved": moved,
                "reached": float((reached[1 + 2 * j, j] - position[1 + 2 * j, j]).abs()),
            }
        )
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"default pose reflects onto itself to {symmetric:.2e} (link spread {scale:.3f} m)")
    for row in report["joints"]:
        print(
            f"{row['name']:28s} -> {row['partner']:28s} sign {row['sign']:+d}  residual "
            f"{row['residual']:.1e} (other {row['other_residual']:.1e}), moved {row['moved']:.3f}"
        )
    print(f"written {args_cli.output}")
    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close(skip_cleanup=True)
    raise SystemExit(code)
