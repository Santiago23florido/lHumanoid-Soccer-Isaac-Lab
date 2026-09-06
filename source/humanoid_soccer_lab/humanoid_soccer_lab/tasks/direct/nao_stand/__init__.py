"""NAO standing-balance task.

Registers ``NaoStand-Direct-v0``, the project's first reinforcement learning
environment. The robot must stay upright under random pushes whose magnitude
grows with training, up to the zero-step capturable limit computed from its own
geometry.

Registration is guarded so that importing the package on a machine without
Isaac Lab -- which the tests do -- degrades to ``REGISTERED = False`` rather
than raising.
"""

from __future__ import annotations

TASK_ID = "NaoStand-Direct-v0"
PLAY_TASK_ID = "NaoStand-Direct-Play-v0"

_MODULE = "humanoid_soccer_lab.tasks.direct.nao_stand"
_AGENTS = f"{_MODULE}.agents"


def register_tasks() -> bool:
    """Register the training and play variants. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    entry_point = f"{_MODULE}.nao_stand_env:NaoStandEnv"

    if TASK_ID not in gym.envs.registry:
        gym.register(
            id=TASK_ID,
            entry_point=entry_point,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_MODULE}.nao_stand_env_cfg:NaoStandEnvCfg",
                "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:NaoStandPPORunnerCfg",
            },
        )

    if PLAY_TASK_ID not in gym.envs.registry:
        gym.register(
            id=PLAY_TASK_ID,
            entry_point=entry_point,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_MODULE}.nao_stand_env_cfg:NaoStandEnvPlayCfg",
                "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:NaoStandPPOPlayRunnerCfg",
            },
        )

    return True


REGISTERED = register_tasks()

__all__ = ["REGISTERED", "TASK_ID", "PLAY_TASK_ID", "register_tasks"]
