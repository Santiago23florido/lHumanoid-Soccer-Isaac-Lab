"""Isaac Lab's rough-terrain G1 policy on flat ground: ``G1RoughOnFlat-v0`` / ``-Play-v0``.

The blind task of the zero-shot study (``transfer/docs/skill_space.md`` §8):
the published ``Isaac-Velocity-Rough-G1-v0`` checkpoint drives it through
``g1/scripts/capture_rollouts.py --checkpoint``. Registration is guarded like
the teacher's, so importing the package without Isaac Lab degrades to
``REGISTERED = False``.
"""

from __future__ import annotations

TASK_ID = "G1RoughOnFlat-v0"
PLAY_TASK_ID = "G1RoughOnFlat-Play-v0"

_MODULE = "humanoid_transfer.g1.tasks.g1_rough_flat"
_AGENT = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg"
    ":G1RoughPPORunnerCfg"
)


def register_tasks() -> bool:
    """Register the tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    for task_id, cfg in (
        (TASK_ID, "G1RoughOnFlatEnvCfg"),
        (PLAY_TASK_ID, "G1RoughOnFlatEnvCfg_PLAY"),
    ):
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.g1_rough_flat_env_cfg:{cfg}",
                    "rsl_rl_cfg_entry_point": _AGENT,
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = ["PLAY_TASK_ID", "REGISTERED", "TASK_ID", "register_tasks"]
