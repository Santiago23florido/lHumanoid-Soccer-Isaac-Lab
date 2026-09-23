"""Measure whether the teacher tracks its command, and whether it walks.

Reward and episode length both saturate for a policy that stands still. They
saturate for one that slides. They saturate for one that walks. Separating those
takes two numbers that the training log does not contain:

**Tracking.** Commanded forward velocity against achieved forward velocity, in
the robot's own heading frame. A policy that stands still scores well on every
term except this one.

**Foot clearance.** How far each foot rises above the ground over a gait cycle.
Walking lifts feet; sliding does not. A few millimetres is contact jitter,
several centimetres is a swing phase.

Usage::

    python g1/scripts/measure_gait.py --checkpoint <...>/exported/policy.pt
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

parser = argparse.ArgumentParser(description="Measure the teacher gait.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--settle-steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_transfer.tasks  # noqa: F401  (registers the task ids)
from humanoid_transfer.g1.tasks.g1_walk import PLAY_TASK_ID
from humanoid_transfer.g1.tasks.g1_walk.g1_walk_env_cfg import FROUDE_MATCHED_SPEED


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

    def act(observation):
        with torch.inference_mode():
            tensor = observation["policy"] if isinstance(observation, dict) else observation
            return policy(tensor)

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        obs, _, _, _, _ = env.step(act(obs))

    start = robot.data.root_pos_w[:, :2].clone()
    forward: list[torch.Tensor] = []
    lateral: list[torch.Tensor] = []
    foot_height: list[torch.Tensor] = []

    for _ in range(args_cli.steps):
        obs, _, _, _, _ = env.step(act(obs))
        # Body-frame linear velocity: x is forward regardless of heading.
        forward.append(robot.data.root_lin_vel_b[:, 0].clone())
        lateral.append(robot.data.root_lin_vel_b[:, 1].clone())
        ground = inner.scene.env_origins[:, 2].unsqueeze(1)
        foot_height.append((robot.data.body_pos_w[:, foot_ids, 2] - ground).clone())

    forward_t = torch.stack(forward)
    lateral_t = torch.stack(lateral)
    heights = torch.stack(foot_height)

    displacement = robot.data.root_pos_w[:, :2] - start
    elapsed = args_cli.steps * inner.step_dt
    net_speed = float(displacement.norm(dim=-1).mean()) / elapsed

    mean_forward = float(forward_t.mean())
    clearance = heights.max(dim=0).values - heights.min(dim=0).values

    print("\n" + "=" * 78)
    print("TEACHER GAIT MEASUREMENT")
    print("=" * 78)
    print(f"  environments          : {inner.num_envs}")
    print(f"  duration              : {elapsed:.1f} s at {1.0 / inner.step_dt:.0f} Hz")

    print("\n  1. DOES IT TRACK THE COMMAND?")
    print(f"     commanded forward     : {FROUDE_MATCHED_SPEED:+.3f} m/s")
    print(f"     achieved forward      : {mean_forward:+.3f} m/s (body frame, mean)")
    print(f"     tracking error        : {mean_forward - FROUDE_MATCHED_SPEED:+.3f} m/s")
    print(f"     lateral drift         : {float(lateral_t.mean()):+.3f} m/s")
    print(f"     net displacement speed: {net_speed:.3f} m/s")
    ratio = mean_forward / FROUDE_MATCHED_SPEED if FROUDE_MATCHED_SPEED else float("nan")
    print(f"     fraction of command   : {ratio:.2f}")

    print("\n  2. DOES IT LIFT ITS FEET?")
    for index, name in enumerate(foot_names):
        values = clearance[:, index]
        print(
            f"     {name:<28s} mean {float(values.mean()) * 1e3:6.1f} mm   "
            f"max {float(values.max()) * 1e3:6.1f} mm"
        )
    peak = float(clearance.max()) * 1e3
    print("     Under ~10 mm is contact jitter, not a swing phase.")

    print("\n  VERDICT")
    tracks = ratio > 0.5
    steps = peak > 10.0
    print(f"     tracks the command    : {'yes' if tracks else 'NO'}")
    print(f"     lifts its feet        : {'yes' if steps else 'NO'}")
    if tracks and steps:
        print("     walking")
    elif steps and not tracks:
        print("     stepping in place: a gait that goes nowhere")
    elif tracks and not steps:
        print("     sliding: tracks the command without a swing phase")
    else:
        print("     standing still")
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
