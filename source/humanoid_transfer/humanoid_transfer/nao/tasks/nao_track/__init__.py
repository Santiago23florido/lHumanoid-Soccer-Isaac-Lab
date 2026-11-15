"""The NAO following a contract: ``NaoTrack-v0``; ``NaoTrack-Sim2Real*-v0`` for the real robot.

The walk's contract generator, trained on the G1 teacher alone, draws plans of
``z*`` in body units; the NAO is rewarded for reaching them and never sees the
command. Nothing here imports the frozen standing task, only the NAO's body.
Registration is guarded, so importing without Isaac Lab gives ``REGISTERED = False``.
"""

from __future__ import annotations

TASK_ID = "NaoTrack-v0"
PLAY_TASK_ID = "NaoTrack-Play-v0"
PHYSICS_TASK_ID = "NaoTrackPhysics-v0"
PHYSICS_PLAY_TASK_ID = "NaoTrackPhysics-Play-v0"
SIM2REAL_TASK_ID = "NaoTrack-Sim2Real-v0"
SIM2REAL_PLAY_TASK_ID = "NaoTrack-Sim2Real-Play-v0"
TEACHER_TASK_ID = "NaoTrack-Sim2RealTeacher-v0"
TEACHER_PLAY_TASK_ID = "NaoTrack-Sim2RealTeacher-Play-v0"
STUDENT_TASK_ID = "NaoTrack-Sim2RealStudent-v0"
STUDENT_PLAY_TASK_ID = "NaoTrack-Sim2RealStudent-Play-v0"

_MODULE = "humanoid_transfer.nao.tasks.nao_track"


def register_tasks() -> bool:
    """Register the NAO tracking tasks. Returns False without gymnasium."""
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        return False

    plain = f"{_MODULE}.agents.rsl_rl_ppo_cfg:NaoTrackPPORunnerCfg"
    physics = f"{_MODULE}.agents.rsl_rl_ppo_cfg:NaoTrackPhysicsPPORunnerCfg"
    for task_id, cfg, agents in (
        (TASK_ID, "nao_track_env_cfg:NaoTrackEnvCfg", plain),
        (PLAY_TASK_ID, "nao_track_env_cfg:NaoTrackEnvCfg_PLAY", plain),
        (PHYSICS_TASK_ID, "nao_track_physics_env_cfg:NaoTrackPhysicsEnvCfg", physics),
        (PHYSICS_PLAY_TASK_ID, "nao_track_physics_env_cfg:NaoTrackPhysicsEnvCfg_PLAY", physics),
        (SIM2REAL_TASK_ID, "nao_track_sim2real_env_cfg:NaoTrackSim2RealEnvCfg", plain),
        (SIM2REAL_PLAY_TASK_ID, "nao_track_sim2real_env_cfg:NaoTrackSim2RealEnvCfg_PLAY", plain),
        (TEACHER_TASK_ID, "nao_track_sim2real_env_cfg:NaoTrackSim2RealTeacherEnvCfg", plain),
        (
            TEACHER_PLAY_TASK_ID,
            "nao_track_sim2real_env_cfg:NaoTrackSim2RealTeacherEnvCfg_PLAY",
            plain,
        ),
        (STUDENT_TASK_ID, "nao_track_sim2real_env_cfg:NaoTrackSim2RealStudentEnvCfg", plain),
        (
            STUDENT_PLAY_TASK_ID,
            "nao_track_sim2real_env_cfg:NaoTrackSim2RealStudentEnvCfg_PLAY",
            plain,
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
    "PHYSICS_PLAY_TASK_ID",
    "PHYSICS_TASK_ID",
    "PLAY_TASK_ID",
    "REGISTERED",
    "SIM2REAL_PLAY_TASK_ID",
    "SIM2REAL_TASK_ID",
    "STUDENT_PLAY_TASK_ID",
    "STUDENT_TASK_ID",
    "TASK_ID",
    "TEACHER_PLAY_TASK_ID",
    "TEACHER_TASK_ID",
    "register_tasks",
]
