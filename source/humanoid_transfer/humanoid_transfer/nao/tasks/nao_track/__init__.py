"""The NAO following a contract: ``NaoTrack-v0``.

The walk's contract generator, trained on the G1 teacher alone, draws plans of
``z*`` in body units; the NAO is rewarded for reaching them and never sees the
command. Nothing here imports the frozen standing task, only the NAO's body.
Registration is guarded, so importing without Isaac Lab gives ``REGISTERED = False``.
"""

from __future__ import annotations

TASK_ID = "NaoTrack-v0"
PLAY_TASK_ID = "NaoTrack-Play-v0"

_MODULE = "humanoid_transfer.nao.tasks.nao_track"


def register_tasks() -> bool:
    """Register the NAO tracking tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    agents = f"{_MODULE}.agents.rsl_rl_ppo_cfg:NaoTrackPPORunnerCfg"
    for task_id, cfg in ((TASK_ID, "NaoTrackEnvCfg"), (PLAY_TASK_ID, "NaoTrackEnvCfg_PLAY")):
        if task_id not in gym.envs.registry:
            gym.register(
                id=task_id,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={
                    "env_cfg_entry_point": f"{_MODULE}.nao_track_env_cfg:{cfg}",
                    "rsl_rl_cfg_entry_point": agents,
                },
            )
    return True


REGISTERED = register_tasks()

__all__ = ["PLAY_TASK_ID", "REGISTERED", "TASK_ID", "register_tasks"]
