"""The NAO trained on the G1's velocity tasks: the oracle baseline, ``NaoVelocity*-v0``.

Isaac Lab's recipe for ``Isaac-Velocity-{Flat,Rough}-G1-v0``, on the NAO and in its
units (``nao_velocity_env_cfg.py``). Registration is guarded, so importing without
Isaac Lab gives ``REGISTERED = False``.
"""

from __future__ import annotations

FLAT_TASK_ID = "NaoVelocityFlat-v0"
FLAT_PLAY_TASK_ID = "NaoVelocityFlat-Play-v0"
ROUGH_TASK_ID = "NaoVelocityRough-v0"
ROUGH_PLAY_TASK_ID = "NaoVelocityRoughOnFlat-Play-v0"

_MODULE = "humanoid_transfer.nao.tasks.nao_velocity"


def register_tasks() -> bool:
    """Register the oracle's tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    flat = f"{_MODULE}.agents.rsl_rl_ppo_cfg:NaoVelocityFlatPPORunnerCfg"
    rough = f"{_MODULE}.agents.rsl_rl_ppo_cfg:NaoVelocityRoughPPORunnerCfg"
    for task_id, cfg, agents in (
        (FLAT_TASK_ID, "NaoVelocityFlatEnvCfg", flat),
        (FLAT_PLAY_TASK_ID, "NaoVelocityFlatEnvCfg_PLAY", flat),
        (ROUGH_TASK_ID, "NaoVelocityRoughEnvCfg", rough),
        (ROUGH_PLAY_TASK_ID, "NaoVelocityRoughOnFlatEnvCfg_PLAY", rough),
    ):
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.nao_velocity_env_cfg:{cfg}",
                    "rsl_rl_cfg_entry_point": agents,
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = [
    "FLAT_PLAY_TASK_ID",
    "FLAT_TASK_ID",
    "REGISTERED",
    "ROUGH_PLAY_TASK_ID",
    "ROUGH_TASK_ID",
    "register_tasks",
]
