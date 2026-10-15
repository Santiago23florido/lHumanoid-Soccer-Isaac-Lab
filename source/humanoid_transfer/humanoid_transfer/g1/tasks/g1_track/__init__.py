"""The G1 following a contract: ``G1Track-v0``.

The walk's contract generator (``logs/contract/walk_warp/generator.pt``, or the
path in ``HUMANOID_CONTRACT``) draws plans of ``z*``; the G1 is rewarded for
reaching them and never sees the command. ``G1Track-Play-v0`` replays on the
deployed path, with the body's encoder feeding the contract, and exposes the
teacher's own observations so captures can be labelled.

Registration is guarded like the teacher's, so importing the package without
Isaac Lab degrades to ``REGISTERED = False``.
"""

from __future__ import annotations

TASK_ID = "G1Track-v0"
PLAY_TASK_ID = "G1Track-Play-v0"

_MODULE = "humanoid_transfer.g1.tasks.g1_track"


def register_tasks() -> bool:
    """Register the tracking tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    agents = f"{_MODULE}.agents.rsl_rl_ppo_cfg:G1TrackPPORunnerCfg"
    for task_id, cfg in (
        (TASK_ID, "G1TrackEnvCfg"),
        (PLAY_TASK_ID, "G1TrackEnvCfg_PLAY"),
    ):
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.g1_track_env_cfg:{cfg}",
                    "rsl_rl_cfg_entry_point": agents,
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = ["PLAY_TASK_ID", "REGISTERED", "TASK_ID", "register_tasks"]
