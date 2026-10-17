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
PHYSICS_TASK_ID = "G1TrackPhysics-v0"
PHYSICS_PLAY_TASK_ID = "G1TrackPhysics-Play-v0"
CORE_TASK_ID = "G1TrackCore-v0"
CORE_PLAY_TASK_ID = "G1TrackCore-Play-v0"
PHYSICS_CORE_TASK_ID = "G1TrackPhysicsCore-v0"
PHYSICS_CORE_PLAY_TASK_ID = "G1TrackPhysicsCore-Play-v0"

_MODULE = "humanoid_transfer.g1.tasks.g1_track"


def register_tasks() -> bool:
    """Register the tracking tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    plain = f"{_MODULE}.agents.rsl_rl_ppo_cfg:G1TrackPPORunnerCfg"
    physics = f"{_MODULE}.agents.rsl_rl_ppo_cfg:G1TrackPhysicsPPORunnerCfg"
    core = f"{_MODULE}.agents.rsl_rl_ppo_cfg:G1TrackCorePPORunnerCfg"
    physics_core = f"{_MODULE}.agents.rsl_rl_ppo_cfg:G1TrackPhysicsCorePPORunnerCfg"
    for task_id, cfg, agents in (
        (TASK_ID, "g1_track_env_cfg:G1TrackEnvCfg", plain),
        (PLAY_TASK_ID, "g1_track_env_cfg:G1TrackEnvCfg_PLAY", plain),
        (PHYSICS_TASK_ID, "g1_track_physics_env_cfg:G1TrackPhysicsEnvCfg", physics),
        (PHYSICS_PLAY_TASK_ID, "g1_track_physics_env_cfg:G1TrackPhysicsEnvCfg_PLAY", physics),
        (CORE_TASK_ID, "g1_track_core_env_cfg:G1TrackCoreEnvCfg", core),
        (CORE_PLAY_TASK_ID, "g1_track_core_env_cfg:G1TrackCoreEnvCfg_PLAY", core),
        (PHYSICS_CORE_TASK_ID, "g1_track_core_env_cfg:G1TrackPhysicsCoreEnvCfg", physics_core),
        (
            PHYSICS_CORE_PLAY_TASK_ID,
            "g1_track_core_env_cfg:G1TrackPhysicsCoreEnvCfg_PLAY",
            physics_core,
        ),
    ):
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.{cfg}",
                    "rsl_rl_cfg_entry_point": agents,
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = [
    "CORE_PLAY_TASK_ID",
    "CORE_TASK_ID",
    "PHYSICS_CORE_PLAY_TASK_ID",
    "PHYSICS_CORE_TASK_ID",
    "PHYSICS_PLAY_TASK_ID",
    "PHYSICS_TASK_ID",
    "PLAY_TASK_ID",
    "REGISTERED",
    "TASK_ID",
    "register_tasks",
]
