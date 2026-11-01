"""The NAO imitating the G1's retargeted joint trajectories: the joint-target baseline.

``NaoImitate-v0`` trains on the reference ``HUMANOID_REFERENCE`` names
(``nao_imitate_env_cfg.py``); ``NaoImitate-Play-v0`` replays one clip per environment.
Registration is guarded, so importing without Isaac Lab gives ``REGISTERED = False``.
"""

from __future__ import annotations

TASK_ID = "NaoImitate-v0"
PLAY_TASK_ID = "NaoImitate-Play-v0"

_MODULE = "humanoid_transfer.nao.tasks.nao_imitate"


def register_tasks() -> bool:
    """Register the baseline's tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    agents = f"{_MODULE}.agents.rsl_rl_ppo_cfg:NaoImitatePPORunnerCfg"
    for task_id, cfg in ((TASK_ID, "NaoImitateEnvCfg"), (PLAY_TASK_ID, "NaoImitateEnvCfg_PLAY")):
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.nao_imitate_env_cfg:{cfg}",
                    "rsl_rl_cfg_entry_point": agents,
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = ["PLAY_TASK_ID", "REGISTERED", "TASK_ID", "register_tasks"]
