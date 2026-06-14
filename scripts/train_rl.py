#!/usr/bin/env python
"""Train a model-free PPO policy on the Franka reach task with RSL-RL.

Self-contained Isaac Lab launcher (no Hydra): it launches Isaac Sim, builds the
project's ``LMBRL-Franka-Reach-v0`` environment, and runs PPO via RSL-RL.

Examples
--------
    # Train headless on an 8 GB GPU (conservative default):
    python scripts/train_rl.py --headless --num_envs 256 --max_iterations 1000

    # Quick smoke run:
    python scripts/train_rl.py --headless --num_envs 64 --max_iterations 10

Checkpoints and TensorBoard logs land under
``logs/rsl_rl/<experiment_name>/<timestamp>/``. Visualize a trained policy with
``scripts/play_rl.py``.

Notes
-----
Isaac Sim must be launched *before* importing any ``isaaclab`` task modules, so
all such imports live below ``AppLauncher(...)`` on purpose.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import traceback

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + launch Isaac Sim (must happen before importing isaaclab task modules).
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train PPO on the Franka reach task (RSL-RL).")
parser.add_argument("--task", type=str, default="LMBRL-Franka-Reach-v0", help="Registered task id.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of parallel envs (overrides cfg)."
)
parser.add_argument("--max_iterations", type=int, default=None, help="PPO training iterations.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything below runs with the simulator live.
# ---------------------------------------------------------------------------
from datetime import datetime  # noqa: E402, I001

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

# Registers LMBRL-Franka-Reach-v0 / -Play-v0 and pulls in the cfg classes.
import lagrangian_mbrl.rl.tasks  # noqa: E402, F401
from lagrangian_mbrl.rl.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    FrankaReachPPORunnerCfg,
)
from lagrangian_mbrl.rl.franka_reach_env_cfg import FrankaReachEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = FrankaReachEnvCfg()
    agent_cfg = FrankaReachPPORunnerCfg()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # --- apply CLI overrides ---
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    # keep env + agent seeds in sync (some randomization happens at env init)
    env_cfg.seed = agent_cfg.seed

    # --- logging directory: logs/rsl_rl/<experiment>/<timestamp>[_<run>] ---
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_dir)
    print(f"[INFO] Logging experiment to: {log_dir}")

    # --- build env, wrap for RSL-RL, run PPO ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # persist the resolved configs alongside the run
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print_dict(agent_cfg.to_dict(), nesting=0)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

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
        # Graceful Kit shutdown can deadlock on Windows with Isaac Sim 5.1.
        simulation_app.close(skip_cleanup=sys.platform == "win32")
