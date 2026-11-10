"""Base tasks on the source robot (``g1_base_env_cfg``).

Registers ``G1Base-{Backward,Sidestep,Jog}-v0`` and their ``-Play-v0`` variants. Each is
Isaac Lab's flat-ground G1 task with only its command range changed, trained with Isaac
Lab's ``G1FlatPPORunnerCfg``, as ``G1Walk-Teacher-v0`` is.

Registration is guarded so that importing the package on a machine without Isaac Lab -- which
the tests do -- degrades to ``REGISTERED = False`` rather than raising.
"""

from __future__ import annotations

TASKS = ("Backward", "Sidestep", "Jog")
_MODULE = "humanoid_transfer.g1.tasks.g1_base"
_ISAACLAB_AGENTS = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg"
)


def task_id(name: str, play: bool = False) -> str:
    return f"G1Base-{name}-Play-v0" if play else f"G1Base-{name}-v0"


def register_tasks() -> bool:
    """Register the base tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False
    for name in TASKS:
        for play in (False, True):
            tid = task_id(name, play)
            if tid in gym.envs.registry:
                continue
            suffix = "_PLAY" if play else ""
            gym.register(
                id=tid,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.g1_base_env_cfg:G1Base{name}EnvCfg{suffix}",
                    "rsl_rl_cfg_entry_point": f"{_ISAACLAB_AGENTS}:G1FlatPPORunnerCfg",
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = ["REGISTERED", "TASKS", "register_tasks", "task_id"]
