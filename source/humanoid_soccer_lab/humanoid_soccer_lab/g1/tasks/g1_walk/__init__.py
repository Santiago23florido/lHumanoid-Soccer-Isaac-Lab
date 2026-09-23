"""Teacher walking task on the source robot.

Registers ``G1Walk-Teacher-v0``. The environment is Isaac Lab's flat-ground
G1 locomotion task with its command range narrowed to the speeds the student
could plausibly reach; see :mod:`g1_walk_env_cfg` for why that narrowing is the
only change.

The PPO configuration is Isaac Lab's own ``G1FlatPPORunnerCfg``. There is no
reason to retune it: the contribution of this project is not a better way to
make a G1 walk.

Registration is guarded so that importing the package on a machine without
Isaac Lab -- which the tests do -- degrades to ``REGISTERED = False`` rather
than raising.
"""

from __future__ import annotations

TASK_ID = "G1Walk-Teacher-v0"
PLAY_TASK_ID = "G1Walk-Teacher-Play-v0"

_MODULE = "humanoid_soccer_lab.g1.tasks.g1_walk"
_ISAACLAB_AGENTS = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg"
)


def register_tasks() -> bool:
    """Register the teacher task. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    entry_point = "isaaclab.envs:ManagerBasedRLEnv"

    if TASK_ID not in gym.envs.registry:
        gym.register(
            id=TASK_ID,
            entry_point=entry_point,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_MODULE}.g1_walk_env_cfg:G1WalkTeacherEnvCfg",
                "rsl_rl_cfg_entry_point": f"{_ISAACLAB_AGENTS}:G1FlatPPORunnerCfg",
            },
        )

    if PLAY_TASK_ID not in gym.envs.registry:
        gym.register(
            id=PLAY_TASK_ID,
            entry_point=entry_point,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_MODULE}.g1_walk_env_cfg:G1WalkTeacherEnvCfg_PLAY",
                "rsl_rl_cfg_entry_point": f"{_ISAACLAB_AGENTS}:G1FlatPPORunnerCfg",
            },
        )

    return True


REGISTERED = register_tasks()

__all__ = ["REGISTERED", "TASK_ID", "PLAY_TASK_ID", "register_tasks"]
