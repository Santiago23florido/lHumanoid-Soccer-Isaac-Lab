"""Validate the teacher environment before committing GPU hours to it.

Answers the questions that decide whether a training run is worth starting, and
that a reward curve will not answer later:

1. Does the articulation load, and does it have the joints the correspondence
   map expects? The map is written against joint *names*, and a renamed or
   absent joint fails silently as a dropped channel rather than as an error.
2. Which teacher joints have no student counterpart? Those are the degrees of
   freedom whose signal has nowhere to go, and their count is the crudest
   measure of the embodiment gap.
3. Is the commanded speed range the Froude-matched one, rather than Isaac Lab's
   default? Getting this wrong wastes the run on a gait the student can never
   reach.
4. Does the environment step without NaN?

Usage::

    python g1/scripts/check_env.py --headless
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

parser = argparse.ArgumentParser(description="Validate the G1 teacher environment.")
parser.add_argument("--num-envs", type=int, default=16, help="Environments. Default 16.")
parser.add_argument("--steps", type=int, default=100, help="Control steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_transfer.tasks  # noqa: F401  (registers the task ids)
from humanoid_transfer.g1.assets.g1 import describe_source_robot
from humanoid_transfer.g1.tasks.g1_walk import TASK_ID
from humanoid_transfer.g1.tasks.g1_walk.g1_walk_env_cfg import (
    FROUDE_MATCHED_SPEED,
    STUDENT_TARGET_SPEED,
)
from humanoid_transfer.transfer.correspondence import (
    TEACHER_JOINTS,
    build_correspondence,
    unmapped_teacher_joints,
)


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(TASK_ID, num_envs=args_cli.num_envs)
    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    robot = inner.scene["robot"]

    described = describe_source_robot()
    joint_names = list(robot.data.joint_names)

    print("\n" + "=" * 78)
    print("G1 TEACHER ENVIRONMENT CHECK")
    print("=" * 78)

    print("\n  1. ARTICULATION")
    print(f"     source robot          : {described['name']} ({described['config']})")
    print(f"     bodies                : {robot.num_bodies}")
    print(f"     joints reported       : {robot.num_joints}")
    print(f"     joints expected       : {described['actuated_dof']}")
    print(f"     environments          : {inner.num_envs}")

    print("\n  2. CORRESPONDENCE WITH THE STUDENT")
    correspondence = build_correspondence()
    expected = {n for names in TEACHER_JOINTS.values() for n in names}
    missing = sorted(expected - set(joint_names))
    unmapped = unmapped_teacher_joints(joint_names)

    print(f"     shared kinematic roles: {len(correspondence.roles)}")
    print(f"     asymmetric roles      : {', '.join(correspondence.asymmetric) or 'none'}")
    if missing:
        print(f"     MAPPED BUT ABSENT     : {', '.join(missing)}")
        print("     The correspondence names joints this robot does not have.")
        print("     Those channels would be silently dropped, not reported.")
    else:
        print("     every mapped joint exists on the robot")
    print(f"     unmapped teacher joints ({len(unmapped)}):")
    for name in unmapped:
        print(f"       {name}")
    print("     Those are degrees of freedom whose teacher signal has nowhere")
    print("     to go on the student. Whatever they achieve has to be achieved")
    print("     some other way, or not at all.")

    print("\n  3. COMMAND RANGE")
    ranges = cfg.commands.base_velocity.ranges
    print(f"     student target speed  : {STUDENT_TARGET_SPEED:.3f} m/s")
    print(f"     Froude-matched speed  : {FROUDE_MATCHED_SPEED:.3f} m/s")
    print(f"     lin_vel_x             : {tuple(round(v, 3) for v in ranges.lin_vel_x)}")
    print(f"     lin_vel_y             : {tuple(round(v, 3) for v in ranges.lin_vel_y)}")
    print(f"     ang_vel_z             : {tuple(round(v, 3) for v in ranges.ang_vel_z)}")
    capped = abs(ranges.lin_vel_x[1] - FROUDE_MATCHED_SPEED) < 1e-6
    print(f"     capped at the match   : {'yes' if capped else 'NO -- still Isaac Lab default'}")

    print("\n  4. STEPPING")
    obs, _ = env.reset()
    actions = torch.zeros(
        inner.num_envs, inner.action_space.shape[1], device=inner.device
    )
    finite = True
    for _ in range(args_cli.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        tensor = obs["policy"] if isinstance(obs, dict) else obs
        if not torch.isfinite(tensor).all() or not torch.isfinite(reward).all():
            finite = False
            break
    height = float(robot.data.root_pos_w[:, 2].mean())
    print(f"     steps run             : {args_cli.steps} with zero action")
    print(f"     observations finite   : {'yes' if finite else 'NO'}")
    print(f"     mean base height      : {height:.4f} m")
    print("     Zero action is not a standing controller here, so the robot")
    print("     falling is expected; only the finiteness matters.")

    ok = finite and not missing and capped
    print("\n" + "=" * 78)
    print(f"  {'PASS' if ok else 'ATTENTION NEEDED'}")
    print("=" * 78 + "\n")

    env.close()
    return 0 if ok else 1


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
