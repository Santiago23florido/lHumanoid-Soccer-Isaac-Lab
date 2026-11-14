"""Each simulated robot's sensor errors: drawn per episode, read by the sensors and the drives.

A real NAO's encoders read the joint plus a calibration offset, and its motor
boards servo the encoder's reading to the target, so the joint settles at the
target minus the offset. Its IMU is mounted with a small tilt, its gyroscope has
a bias, and each foot's force sensors a gain and an offset. One
:class:`RobotErrors` per environment holds them; :func:`sample_robot_errors` (a
reset event) draws them, and the observation terms, the action term and the
estimator read them. Before any draw every error is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.managers import SceneEntityCfg

ENCODER_RESOLUTION = 2.0 * torch.pi / 4096
"""A 12-bit magnetic encoder's step, rad."""
ROBOT = SceneEntityCfg("robot")


@dataclass
class RobotErrors:
    encoder_offset: torch.Tensor
    """``(N, J)`` added to each joint's reading, in the articulation's joint order."""
    imu_tilt: torch.Tensor
    """``(N, 2)`` roll and pitch of the IMU's mounting relative to the base, rad."""
    gyro_bias: torch.Tensor
    """``(N, 3)`` rad/s, in the IMU frame."""
    fsr_scale: torch.Tensor
    """``(N, 2)`` each foot's force-sensor gain."""
    fsr_offset: torch.Tensor
    """``(N, 2)`` each foot's force-sensor offset, in units of ``m g``."""


def robot_errors(env) -> RobotErrors:
    """The environment's :class:`RobotErrors`, created (all zero) on first use."""
    errors = getattr(env, "_robot_errors", None)
    if errors is None:
        robot = env.scene["robot"]
        n, j, device = env.num_envs, robot.num_joints, env.device
        errors = RobotErrors(
            encoder_offset=torch.zeros(n, j, device=device),
            imu_tilt=torch.zeros(n, 2, device=device),
            gyro_bias=torch.zeros(n, 3, device=device),
            fsr_scale=torch.ones(n, 2, device=device),
            fsr_offset=torch.zeros(n, 2, device=device),
        )
        env._robot_errors = errors
    return errors


def _uniform(count: int, width: int, half: float, device) -> torch.Tensor:
    return (2.0 * torch.rand(count, width, device=device) - 1.0) * half


def sample_robot_errors(
    env,
    env_ids: torch.Tensor | None,
    encoder_offset: float = 0.0,
    imu_tilt: float = 0.0,
    gyro_bias: float = 0.0,
    fsr_scale: float = 0.0,
    fsr_offset: float = 0.0,
    asset_cfg: SceneEntityCfg = ROBOT,
) -> None:
    """Draw each error uniformly within ``±`` its bound (rad, rad, rad/s, gain, ``m g``).

    A mimic joint reads its driver's encoder: one motor drives both hip yaw-pitches.
    """
    errors = robot_errors(env)
    robot = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    n, device = len(ids), env.device
    offset = _uniform(n, robot.num_joints, encoder_offset, device)
    for joint, driver in mimic_pairs(robot).items():
        offset[:, joint] = offset[:, driver]
    errors.encoder_offset[ids] = offset
    errors.imu_tilt[ids] = _uniform(n, 2, imu_tilt, device)
    errors.gyro_bias[ids] = _uniform(n, 3, gyro_bias, device)
    errors.fsr_scale[ids] = 1.0 + _uniform(n, 2, fsr_scale, device)
    errors.fsr_offset[ids] = _uniform(n, 2, fsr_offset, device)


def mimic_pairs(robot) -> dict[int, int]:
    """``{mimic joint index: driver index}`` from the URDF, for the joints the robot has."""
    from ..assets import nao_kinematics as nk

    names = list(robot.data.joint_names)
    pairs = {}
    for joint in nk.load_model().joints:
        if joint.mimic is not None and joint.name in names and joint.mimic in names:
            pairs[names.index(joint.name)] = names.index(joint.mimic)
    return pairs


def measured_joint_pos(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    """``(N, J)`` what the encoders read: the joint, its offset, at the encoder's resolution.

    A mimic joint reads its driver's encoder (:func:`sample_robot_errors` gives it the
    driver's offset; the joint itself follows the driver through the constraint).
    """
    robot = env.scene[asset_cfg.name]
    reading = robot.data.joint_pos + robot_errors(env).encoder_offset
    return torch.round(reading / ENCODER_RESOLUTION) * ENCODER_RESOLUTION


def imu_rotation(env) -> torch.Tensor:
    """``(N, 3, 3)`` base-from-IMU rotation: the mounting tilt, roll then pitch."""
    tilt = robot_errors(env).imu_tilt
    roll, pitch = tilt[:, 0], tilt[:, 1]
    cr, sr, cp, sp = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch)
    zero, one = torch.zeros_like(cr), torch.ones_like(cr)
    rx = torch.stack(
        [
            torch.stack([one, zero, zero], -1),
            torch.stack([zero, cr, -sr], -1),
            torch.stack([zero, sr, cr], -1),
        ],
        -2,
    )
    ry = torch.stack(
        [
            torch.stack([cp, zero, sp], -1),
            torch.stack([zero, one, zero], -1),
            torch.stack([-sp, zero, cp], -1),
        ],
        -2,
    )
    return ry @ rx


def measured_gravity(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    """``(N, 3)`` gravity's direction as the tilted IMU reads it."""
    robot = env.scene[asset_cfg.name]
    return torch.einsum("nji,nj->ni", imu_rotation(env), robot.data.projected_gravity_b)


def measured_gyro(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    """``(N, 3)`` the base's angular velocity in the tilted IMU frame, plus the gyro's bias."""
    robot = env.scene[asset_cfg.name]
    turned = torch.einsum("nji,nj->ni", imu_rotation(env), robot.data.root_ang_vel_b)
    return turned + robot_errors(env).gyro_bias


__all__ = [
    "ENCODER_RESOLUTION",
    "RobotErrors",
    "imu_rotation",
    "measured_gravity",
    "measured_gyro",
    "measured_joint_pos",
    "mimic_pairs",
    "robot_errors",
    "sample_robot_errors",
]
