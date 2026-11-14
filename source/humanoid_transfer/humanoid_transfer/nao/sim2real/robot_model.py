"""Put a real NAO's imperfections into a tracking environment's configuration.

:func:`apply_robot_model` takes a dictionary of settings -- every key optional,
an absent key leaving that part of the simulation as it was -- and edits an
environment configuration in place. Ranges are ``[low, high]``, drawn per
environment at startup (drives, masses, friction) or per episode (latency,
sensor errors); single numbers are symmetric bounds ``±``.

========================  ===============================================================
``delay_ms``              latency from the policy's output to the drives
``stiffness_scale``       leg drives' stiffness, times the derived gains
``damping_scale``         leg drives' damping, times the derived gains
``effort_scale``          leg joints' torque ceilings, times the URDF's
``armature_scale``        leg joints' reflected rotor inertia, times the nominal
``friction``              the feet's static friction coefficient (dynamic: 3/4 of it)
``mass_scale``            every link's mass
``added_mass_kg``         a load on the torso (a marker, a camera)
``base_com_m``            the torso's centre of mass shifted, metres
``encoder_offset_deg``    each joint encoder's calibration offset
``imu_tilt_deg``          the IMU's mounting tilt, roll and pitch
``gyro_bias``             the gyroscope's bias, rad/s
``fsr_scale``             each foot's force-sensor gain, ``1 ±``
``fsr_offset``            each foot's force-sensor offset, units of ``m g``
``sensors``               the tracker reads its IMU and joints through those errors, and
                          joint velocities by finite differences
``estimator``             ``ẑ*`` from the robot's estimator (:mod:`.command`)
``odometry``              the reference path placed by the robot's leg odometry
``loads``                 the estimator's load source (:data:`.command.LOAD_SOURCES`)
``base_velocity``         false: the plain kinematic estimator measures velocities from
                          the stance sole (its leg-odometry variant)
========================  ===============================================================
"""

from __future__ import annotations

import math

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg

from ..assets import nao_kinematics as nk
from . import sensors
from .actions import RobotJointPositionActionCfg
from .command import robot_command
from .events import scale_effort_limits, set_reflected_inertia
from .state import sample_robot_errors

LEGS = SceneEntityCfg("robot", joint_names=list(nk.LEG_ACTUATED_JOINTS), preserve_order=True)
KEYS = (
    "delay_ms", "stiffness_scale", "damping_scale", "effort_scale", "armature_scale",
    "friction", "mass_scale", "added_mass_kg", "base_com_m", "encoder_offset_deg",
    "imu_tilt_deg", "gyro_bias", "fsr_scale", "fsr_offset", "sensors", "estimator",
    "odometry", "loads", "base_velocity", "fsr_noise", "gyro_noise",
)  # fmt: skip


def _pair(value) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    low, high = value
    return float(low), float(high)


def apply_robot_model(cfg, settings: dict) -> None:
    """Edit the environment configuration ``cfg`` in place (see the module's table)."""
    unknown = set(settings) - set(KEYS)
    if unknown:
        raise KeyError(f"unknown robot-model settings {sorted(unknown)}")
    s = settings
    events = cfg.events

    action = cfg.actions.joint_pos
    robot_action = RobotJointPositionActionCfg()
    for key, value in vars(action).items():
        setattr(robot_action, key, value)
    robot_action.class_type = RobotJointPositionActionCfg.class_type
    if "delay_ms" in s:
        low, high = _pair(s["delay_ms"])
        robot_action.delay_s = (low / 1000.0, high / 1000.0)
    cfg.actions.joint_pos = robot_action

    errors = {
        "encoder_offset": math.radians(float(s.get("encoder_offset_deg", 0.0))),
        "imu_tilt": math.radians(float(s.get("imu_tilt_deg", 0.0))),
        "gyro_bias": float(s.get("gyro_bias", 0.0)),
        "fsr_scale": float(s.get("fsr_scale", 0.0)),
        "fsr_offset": float(s.get("fsr_offset", 0.0)),
    }
    events.robot_errors = EventTermCfg(func=sample_robot_errors, mode="reset", params=errors)

    if "stiffness_scale" in s or "damping_scale" in s:
        params = {"asset_cfg": LEGS, "operation": "scale", "distribution": "uniform"}
        if "stiffness_scale" in s:
            params["stiffness_distribution_params"] = _pair(s["stiffness_scale"])
        if "damping_scale" in s:
            params["damping_distribution_params"] = _pair(s["damping_scale"])
        events.drive_gains = EventTermCfg(
            func=mdp.randomize_actuator_gains, mode="startup", params=params
        )
    if "effort_scale" in s:
        events.effort_limits = EventTermCfg(
            func=scale_effort_limits,
            mode="startup",
            params={"asset_cfg": LEGS, "scale_range": _pair(s["effort_scale"])},
        )
    if "armature_scale" in s:
        events.reflected_inertia = EventTermCfg(
            func=set_reflected_inertia,
            mode="startup",
            params={"asset_cfg": LEGS, "scale_range": _pair(s["armature_scale"])},
        )
    if "friction" in s:
        low, high = _pair(s["friction"])
        events.physics_material.params["static_friction_range"] = (low, high)
        events.physics_material.params["dynamic_friction_range"] = (0.75 * low, 0.75 * high)
    if "mass_scale" in s:
        events.link_masses = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": _pair(s["mass_scale"]),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
    if "added_mass_kg" in s:
        events.add_base_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": _pair(s["added_mass_kg"]),
                "operation": "add",
            },
        )
    if "base_com_m" in s:
        c = float(s["base_com_m"])
        events.base_com = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-c, c), "y": (-c, c), "z": (-c / 2, c / 2)},
            },
        )

    if s.get("sensors", False):
        policy = cfg.observations.policy
        for name, func in (
            ("base_ang_vel", sensors.gyro),
            ("projected_gravity", sensors.gravity),
            ("joint_pos", sensors.encoder_joint_pos_rel),
            ("joint_vel", sensors.encoder_joint_vel),
        ):
            # A term's history reads the same sensor.
            for term_name in (name, f"{name}_history"):
                term = getattr(policy, term_name, None)
                if term is not None:
                    term.func = func

    command = cfg.commands.base_velocity
    if "base_velocity" in s:
        command.base_velocity = bool(s["base_velocity"])
    options = {}
    if s.get("estimator", False):
        options["robot_estimator"] = True
    if s.get("odometry", False):
        options["robot_odometry"] = True
    for key, option in (
        ("loads", "load_source"),
        ("fsr_noise", "fsr_noise"),
        ("gyro_noise", "gyro_noise"),
    ):
        if key in s:
            options[option] = s[key]
    cfg.commands.base_velocity = robot_command(command, **options)


__all__ = ["KEYS", "LEGS", "apply_robot_model"]
