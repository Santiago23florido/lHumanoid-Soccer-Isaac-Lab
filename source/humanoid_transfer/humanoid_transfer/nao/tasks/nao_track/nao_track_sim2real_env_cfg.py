"""The NAO tracker trained for the real robot: ``NaoTrack-Sim2Real-v0``.

:class:`~.nao_track_env_cfg.NaoTrackEnvCfg` -- the same plans (the procedural
family and the walk's contract, in the same shares), the same contract reward,
the same terminations -- on a simulated robot with a real NAO's imperfections
(:mod:`humanoid_transfer.nao.sim2real`, ranges in :data:`ROBOT_RANGES`):

* the tracker reads only what the robot computes: its gyroscope and IMU tilt,
  its encoders (offsets, resolution, velocities by finite differences), and the
  contract from the robot's own estimator and leg odometry (the same NumPy code
  that runs on the robot) with the feet's loads from their force sensors;
* the drives are late, their gains, torque ceilings and the motors' reflected
  inertia are drawn per robot, as are friction and the masses; the base is
  pushed now and then;
* a short history of what it senses and did (:data:`HISTORY` steps), from which
  it can infer what it cannot sense: its latency and its drives' strength;
* one body cost more, the motors' heat (:func:`~...sim2real.rewards.motor_heat`).

The critic keeps the simulator's truth (asymmetric actor-critic).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ...sim2real.params import PARAMS, robot_params
from ...sim2real.rewards import motor_heat
from ...sim2real.robot_model import LEGS, apply_robot_model
from .nao_track_env_cfg import NaoTrackEnvCfg, NaoTrackEnvCfg_PLAY

ROBOT_RANGES = {
    # Latency: B-Human measures ~36 ms from a LoLA command to the joint's response;
    # the policy's 20 ms step held over LoLA's 12 ms cycle adds 0-12 ms. The adopted tracker
    # falls at 60 ms (nao/results/sensitivity_r1_3498_sym_v3.json): train through it.
    "delay_ms": [10.0, 60.0],
    "stiffness_scale": [0.7, 1.3],
    "damping_scale": [0.7, 1.3],
    # The URDF's ceiling (3.02 N m at the pitch joints) is above the motors' nominal 2.11;
    # what the boards' current limit and the gearbox leave is measured on the robot.
    "effort_scale": [0.7, 1.1],
    "armature_scale": [0.5, 1.5],
    # The robot's soles against a floor of 0.8, combined by product: 0.4 to 1.04 (carpet).
    "friction": [0.5, 1.3],
    "mass_scale": [0.95, 1.05],
    "added_mass_kg": [0.0, 0.15],
    "base_com_m": 0.01,
    "encoder_offset_deg": 1.5,
    "imu_tilt_deg": 1.5,
    "gyro_bias": 0.02,
    "fsr_scale": 0.15,
    "fsr_offset": 0.03,
    "fsr_noise": 0.02,
    "gyro_noise": 0.01,
    "sensors": True,
    "estimator": True,
    "odometry": True,
    "loads": "fsr",
}
"""Ranges drawn per robot (drives, masses, friction) or per episode (latency, sensor
errors); centred on the nominal model, widened where the robot is unmeasured."""

PARAM_RANGES = tuple(tuple(float(x) for x in ROBOT_RANGES[key]) for key in PARAMS)
"""The ranges ``robot_params`` is normalised over: fixed, so a measured robot card changes
what is drawn, never what the numbers mean."""

ROBOT_CARD = os.environ.get("HUMANOID_ROBOT_CARD", "")
"""A JSON of settings measured on the robot (``nao/scripts/fit_actuators.py``): each key
replaces its range in :data:`ROBOT_RANGES`."""
if ROBOT_CARD:
    ROBOT_RANGES = {**ROBOT_RANGES, **json.loads(Path(ROBOT_CARD).read_text(encoding="utf-8"))}

HISTORY = int(os.environ.get("HUMANOID_SIM2REAL_HISTORY", "3"))
"""Steps of history appended to the tracker's input (0: none)."""
HISTORY_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
HEAT_WEIGHT = float(os.environ.get("HUMANOID_HEAT_WEIGHT", "-0.01"))
PUSH_SPEED = 0.1
"""m/s, about 0.06 of the NAO's velocity unit sqrt(g L), every 5 to 10 s."""
JOINT_SPREAD = 0.1
"""The teacher's robots: each drive factor drawn once per robot, each joint ``1 ±`` this."""
ADAPT_HISTORY = int(os.environ.get("HUMANOID_ADAPT_HISTORY", "25"))
"""Steps (0.5 s) of proprioception and actions the student's adaptation module reads."""


def add_history(policy, length: int = HISTORY) -> None:
    """Append ``<term>_history`` for each of :data:`HISTORY_TERMS`: the term's last ``length``
    readings, oldest first, after every other input (so a tracker without them widens)."""
    if length <= 0:
        return
    for name in HISTORY_TERMS:
        term = getattr(policy, name)
        setattr(
            policy,
            f"{name}_history",
            ObsTerm(
                func=term.func,
                params=dict(term.params),
                noise=term.noise,
                history_length=length,
                flatten_history_dim=True,
            ),
        )


def add_robot_params(group) -> None:
    """Append ``robot_params`` (:mod:`~...sim2real.params`) after every other input."""
    group.robot_params = ObsTerm(func=robot_params, params={"ranges": PARAM_RANGES})


def add_adaptation_history(policy, length: int = ADAPT_HISTORY) -> None:
    """Append ``<term>_adapt_history`` for each of :data:`HISTORY_TERMS`: the longer history
    the student's adaptation module reads, after every other input."""
    for name in HISTORY_TERMS:
        term = getattr(policy, name)
        setattr(
            policy,
            f"{name}_adapt_history",
            ObsTerm(
                func=term.func,
                params=dict(term.params),
                noise=term.noise,
                history_length=length,
                flatten_history_dim=True,
            ),
        )


def make_robot_task(cfg, ranges: dict) -> None:
    """The tracking task on the robot model ``ranges``, with the history, the motors' heat
    and pushes."""
    add_history(cfg.observations.policy)
    apply_robot_model(cfg, ranges)
    cfg.rewards.motor_heat = RewTerm(
        func=motor_heat, weight=HEAT_WEIGHT, params={"asset_cfg": LEGS}
    )
    cfg.events.push_robot = EventTermCfg(
        func=isaac_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-PUSH_SPEED, PUSH_SPEED), "y": (-PUSH_SPEED, PUSH_SPEED)}},
    )


@configclass
class NaoTrackSim2RealEnvCfg(NaoTrackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        make_robot_task(self, ROBOT_RANGES)


@configclass
class NaoTrackSim2RealEnvCfg_PLAY(NaoTrackEnvCfg_PLAY):
    """The plain play scene with the trained tracker's inputs; captures add the robot model
    they test (``transfer/scripts/capture_contract.py --robot-model``)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        add_history(self.observations.policy)


@configclass
class NaoTrackSim2RealTeacherEnvCfg(NaoTrackEnvCfg):
    """``NaoTrack-Sim2RealTeacher-v0``: the robot task, each robot's drives drawn as one factor
    (:data:`JOINT_SPREAD`), and the actor and critic told the robot's parameters
    (``robot_params``)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        make_robot_task(self, {**ROBOT_RANGES, "joint_spread": JOINT_SPREAD})
        add_robot_params(self.observations.policy)
        add_robot_params(self.observations.critic)


@configclass
class NaoTrackSim2RealStudentEnvCfg(NaoTrackSim2RealTeacherEnvCfg):
    """``NaoTrack-Sim2RealStudent-v0``: the teacher's task with the adaptation module's history
    appended to the policy group; the true ``robot_params`` stay in it as the student's
    regression target, and a student replaces them with its estimate."""

    def __post_init__(self) -> None:
        super().__post_init__()
        add_adaptation_history(self.observations.policy)


@configclass
class NaoTrackSim2RealTeacherEnvCfg_PLAY(NaoTrackSim2RealEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        add_robot_params(self.observations.policy)


@configclass
class NaoTrackSim2RealStudentEnvCfg_PLAY(NaoTrackSim2RealTeacherEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        add_adaptation_history(self.observations.policy)


__all__ = [
    "ADAPT_HISTORY",
    "HISTORY",
    "JOINT_SPREAD",
    "NaoTrackSim2RealEnvCfg",
    "NaoTrackSim2RealEnvCfg_PLAY",
    "NaoTrackSim2RealStudentEnvCfg",
    "NaoTrackSim2RealStudentEnvCfg_PLAY",
    "NaoTrackSim2RealTeacherEnvCfg",
    "NaoTrackSim2RealTeacherEnvCfg_PLAY",
    "PARAM_RANGES",
    "ROBOT_RANGES",
    "add_adaptation_history",
    "add_history",
    "add_robot_params",
    "make_robot_task",
]
