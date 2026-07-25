"""Gym registration scaffold for the humanoid soccer Direct RL task."""

from __future__ import annotations

TASK_ID = "HumanoidSoccer-Direct-v0"


def register_task() -> bool:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    if TASK_ID in gym.envs.registry:
        return True

    gym.register(
        id=TASK_ID,
        entry_point=(
            "humanoid_soccer_lab.tasks.direct.humanoid_soccer.humanoid_soccer_env:"
            "HumanoidSoccerEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "humanoid_soccer_lab.tasks.direct.humanoid_soccer.humanoid_soccer_env_cfg:"
                "HumanoidSoccerEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                "humanoid_soccer_lab.tasks.direct.humanoid_soccer.agents.rsl_rl_ppo_cfg:"
                "HumanoidSoccerPPORunnerCfg"
            ),
        },
    )
    return True


REGISTERED = register_task()

__all__ = ["REGISTERED", "TASK_ID", "register_task"]
