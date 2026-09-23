"""Record what the teacher does, in the form the student can be shown.

A trained teacher is only useful to the transfer study once its behaviour is
written down. What gets written down is the decision, because it fixes which of
the three candidate signals can be compared at all:

``joint``
    Positions and velocities of every actuated joint, plus the names, so the
    correspondence map can be applied afterwards rather than baked in here.
``centroidal``
    Centre-of-mass position and velocity, and centroidal angular momentum. This
    is the signal that is invariant to the joint-count mismatch.
``contact``
    Per-foot normal force, from which the contact schedule -- when each foot
    lands, and where -- is recovered.

All three are recorded together. Deciding later which to use costs nothing;
re-running the capture costs a GPU hour.

Everything is stored in the **teacher's own units**. Scaling to the student is
the transfer package's job, and doing it here would bake in one answer to the
question the study is asking.

Usage::

    python g1/scripts/capture_rollouts.py --headless \
        --checkpoint logs/rsl_rl/g1_flat/<run>/model_1499.pt \
        --output g1/rollouts/teacher.npz
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
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

parser = argparse.ArgumentParser(description="Record teacher rollouts for transfer.")
parser.add_argument(
    "--checkpoint", type=Path, required=True, help="TorchScript policy or .pt checkpoint."
)
parser.add_argument("--num-envs", type=int, default=32, help="Parallel rollouts.")
parser.add_argument(
    "--steps", type=int, default=600, help="Control steps to record. Default 600 (12 s)."
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=100,
    help="Steps discarded before recording, so the gait is periodic. Default 100.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=_REPO_ROOT / "g1" / "rollouts" / "teacher.npz",
    help="Where to write the archive.",
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
from humanoid_transfer.g1.tasks.g1_walk.g1_walk_env_cfg import FROUDE_MATCHED_SPEED


def centroidal_state(robot, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Centre of mass, its velocity, and centroidal angular momentum.

    Isaac Lab exposes per-body states, not centroidal ones, so the aggregation
    is done here:

    .. math::

        c = \\frac{1}{M}\\sum_i m_i c_i, \\qquad
        L = \\sum_i \\bigl[ m_i (c_i - c) \\times (v_i - \\dot c) + I_i \\omega_i \\bigr]

    The inertial term uses each body's diagonal inertia in its own frame, which
    is an approximation: it ignores the rotation of the inertia tensor into the
    world frame. For a gait signal the translational term dominates, and the
    approximation is recorded here rather than hidden.
    """
    masses = robot.data.default_mass.to(device)
    total = masses.sum(dim=1, keepdim=True)
    weights = (masses / total).unsqueeze(-1)

    positions = robot.data.body_com_pos_w
    velocities = robot.data.body_com_lin_vel_w

    com = (positions * weights).sum(dim=1)
    com_vel = (velocities * weights).sum(dim=1)

    relative_pos = positions - com.unsqueeze(1)
    relative_vel = velocities - com_vel.unsqueeze(1)
    orbital = torch.cross(relative_pos, relative_vel * masses.unsqueeze(-1), dim=-1)
    momentum = orbital.sum(dim=1)

    return com, com_vel, momentum


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(PLAY_TASK_ID, num_envs=args_cli.num_envs)
    env = gym.make(PLAY_TASK_ID, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    robot = inner.scene["robot"]

    policy = torch.jit.load(str(args_cli.checkpoint), map_location=device)
    policy.eval()

    foot_ids, foot_names = robot.find_bodies(".*ankle_roll.*")
    if not foot_ids:
        foot_ids, foot_names = robot.find_bodies(".*ankle.*")

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            action = policy(obs["policy"] if isinstance(obs, dict) else obs)
        obs, _, _, _, _ = env.step(action)

    record: dict[str, list] = {k: [] for k in (
        "joint_pos", "joint_vel", "com", "com_vel", "angular_momentum",
        "base_quat", "base_lin_vel", "foot_pos", "foot_force",
    )}

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            action = policy(obs["policy"] if isinstance(obs, dict) else obs)
        obs, _, _, _, _ = env.step(action)

        com, com_vel, momentum = centroidal_state(robot, device)
        origins = inner.scene.env_origins

        record["joint_pos"].append(robot.data.joint_pos.clone().cpu())
        record["joint_vel"].append(robot.data.joint_vel.clone().cpu())
        record["com"].append((com - origins).cpu())
        record["com_vel"].append(com_vel.cpu())
        record["angular_momentum"].append(momentum.cpu())
        record["base_quat"].append(robot.data.root_quat_w.clone().cpu())
        record["base_lin_vel"].append(robot.data.root_lin_vel_b.clone().cpu())
        record["foot_pos"].append(
            (robot.data.body_pos_w[:, foot_ids, :] - origins.unsqueeze(1)).cpu()
        )
        record["foot_force"].append(
            inner.scene["contact_forces"].data.net_forces_w[:, foot_ids, 2].clone().cpu()
        )

    # (steps, envs, ...) -> arrays.
    arrays = {k: torch.stack(v).numpy().astype(np.float32) for k, v in record.items()}

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        joint_names=np.array(robot.data.joint_names),
        foot_names=np.array(foot_names),
        control_dt=np.float32(inner.step_dt),
        commanded_speed=np.float32(FROUDE_MATCHED_SPEED),
        total_mass=np.float32(float(robot.data.default_mass.sum(dim=1)[0])),
        **arrays,
    )

    size_mb = args_cli.output.stat().st_size / 1e6
    speed = float(np.linalg.norm(arrays["com_vel"][:, :, :2], axis=-1).mean())

    print("\n" + "=" * 78)
    print("TEACHER ROLLOUTS CAPTURED")
    print("=" * 78)
    print(f"  file                  : {args_cli.output}  ({size_mb:.1f} MB)")
    print(f"  shape                 : {args_cli.steps} steps x {inner.num_envs} envs")
    print(f"  control rate          : {1.0 / inner.step_dt:.0f} Hz")
    print(f"  actuated joints       : {len(robot.data.joint_names)}")
    print(f"  feet tracked          : {', '.join(foot_names)}")
    print(f"  commanded speed       : {FROUDE_MATCHED_SPEED:.3f} m/s")
    print(f"  achieved mean speed   : {speed:.3f} m/s")
    print(f"  tracking error        : {abs(speed - FROUDE_MATCHED_SPEED):.3f} m/s")
    print("\n  Stored in the teacher's own units. Scaling to the student is the")
    print("  transfer package's job; doing it here would bake in one answer to")
    print("  the question the study asks.")
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
