#!/usr/bin/env python
"""Roll out a trained PPO policy on the Franka reach task (RSL-RL).

Examples
--------
    # Visualize the latest checkpoint (GUI):
    python scripts/play_rl.py --checkpoint logs/rsl_rl/lmbrl_franka_reach/<run>/model_1000.pt

    # Headless eval with a few envs:
    python scripts/play_rl.py --headless --num_envs 16 \
        --checkpoint logs/rsl_rl/lmbrl_franka_reach/<run>/model_1000.pt

Uses the small ``-Play-v0`` scene (few envs, no observation noise).
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained PPO policy on the Franka reach task.")
parser.add_argument(
    "--task", type=str, default="LMBRL-Franka-Reach-Play-v0", help="Registered task id."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to a saved RSL-RL model_*.pt."
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of parallel envs (overrides cfg)."
)
parser.add_argument(
    "--steps", type=int, default=1000, help="Number of environment steps to roll out."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402, I001
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import lagrangian_mbrl.rl.tasks  # noqa: E402, F401  (registers the task ids)
from lagrangian_mbrl.rl.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    FrankaReachPPORunnerCfg,
)
from lagrangian_mbrl.rl.franka_reach_env_cfg import FrankaReachEnvCfg_PLAY  # noqa: E402


def main() -> None:
    env_cfg = FrankaReachEnvCfg_PLAY()
    agent_cfg = FrankaReachPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs, _ = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if sys.platform == "win32":
            os._exit(1)
        simulation_app.close()
        raise
    else:
        simulation_app.close(skip_cleanup=sys.platform == "win32")
