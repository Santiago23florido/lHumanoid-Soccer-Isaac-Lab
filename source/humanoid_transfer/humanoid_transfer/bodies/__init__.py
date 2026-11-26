"""Bodies beyond the G1 and the NAO, described by ``transfer/bodies/bodies.json``.

Registers a contract tracker per body Isaac Lab ships (``H1Track-v0``, ``DigitTrack-v0``,
``CassieTrack-v0`` and their play variants). Registration is guarded, so importing
without gymnasium gives ``REGISTERED = False``.
"""

from __future__ import annotations

_MODULE = "humanoid_transfer.bodies"
TASKS = {
    "H1Track-v0": ("body_track_env_cfg:H1TrackEnvCfg", "agents:H1TrackPPORunnerCfg"),
    "H1Track-Play-v0": ("body_track_env_cfg:H1TrackEnvCfg_PLAY", "agents:H1TrackPPORunnerCfg"),
    "CassieTrack-v0": ("body_track_env_cfg:CassieTrackEnvCfg", "agents:CassieTrackPPORunnerCfg"),
    "CassieTrack-Play-v0": (
        "body_track_env_cfg:CassieTrackEnvCfg_PLAY",
        "agents:CassieTrackPPORunnerCfg",
    ),
    "DigitTrack-v0": ("body_track_env_cfg:DigitTrackEnvCfg", "agents:DigitTrackPPORunnerCfg"),
    "DigitTrack-Play-v0": (
        "body_track_env_cfg:DigitTrackEnvCfg_PLAY",
        "agents:DigitTrackPPORunnerCfg",
    ),
}


def register_tasks() -> bool:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False
    for task_id, (env_cfg, agent_cfg) in TASKS.items():
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.{env_cfg}",
                    "rsl_rl_cfg_entry_point": f"{_MODULE}.{agent_cfg}",
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = ["REGISTERED", "TASKS", "register_tasks"]
