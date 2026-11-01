"""The NAO trained on the G1's velocity tasks: the oracle baseline, ``NaoVelocity*-v0``.

Isaac Lab's recipe for ``Isaac-Velocity-{Flat,Rough}-G1-v0``, on the NAO and in its
units (``nao_velocity_env_cfg.py``), and ``NaoVelocity*Bounded*``, the same recipe with its
air-time term bounded. Registration is guarded, so importing without
Isaac Lab gives ``REGISTERED = False``.
"""

from __future__ import annotations

FLAT_TASK_ID = "NaoVelocityFlat-v0"
FLAT_PLAY_TASK_ID = "NaoVelocityFlat-Play-v0"
ROUGH_TASK_ID = "NaoVelocityRough-v0"
ROUGH_PLAY_TASK_ID = "NaoVelocityRoughOnFlat-Play-v0"
BOUNDED_FLAT_TASK_ID = "NaoVelocityFlatBounded-v0"
BOUNDED_FLAT_PLAY_TASK_ID = "NaoVelocityFlatBounded-Play-v0"
BOUNDED_ROUGH_TASK_ID = "NaoVelocityRoughBounded-v0"
BOUNDED_ROUGH_PLAY_TASK_ID = "NaoVelocityRoughBoundedOnFlat-Play-v0"

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
        (BOUNDED_FLAT_TASK_ID, "NaoVelocityFlatBoundedEnvCfg", flat),
        (BOUNDED_FLAT_PLAY_TASK_ID, "NaoVelocityFlatBoundedEnvCfg_PLAY", flat),
        (BOUNDED_ROUGH_TASK_ID, "NaoVelocityRoughBoundedEnvCfg", rough),
        (BOUNDED_ROUGH_PLAY_TASK_ID, "NaoVelocityRoughBoundedOnFlatEnvCfg_PLAY", rough),
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
    "BOUNDED_FLAT_PLAY_TASK_ID",
    "BOUNDED_FLAT_TASK_ID",
    "BOUNDED_ROUGH_PLAY_TASK_ID",
    "BOUNDED_ROUGH_TASK_ID",
    "FLAT_PLAY_TASK_ID",
    "FLAT_TASK_ID",
    "REGISTERED",
    "ROUGH_PLAY_TASK_ID",
    "ROUGH_TASK_ID",
    "register_tasks",
]
