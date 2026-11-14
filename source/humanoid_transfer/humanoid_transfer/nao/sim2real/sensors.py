"""The tracker's observations as the robot's sensors give them.

Drop-in replacements for Isaac Lab's ``base_ang_vel``, ``projected_gravity``,
``joint_pos_rel`` and ``joint_vel_rel``, read through
:mod:`~humanoid_transfer.nao.sim2real.state`'s errors: the gyroscope with its
bias, the IMU's tilted gravity, the encoders with their offsets and resolution,
and joint velocities by finite differences of the encoders over one control
step -- a NAO has no velocity sensor. The terms keep their names, so the mirror
and the tracker's input layout are unchanged.
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from .state import ROBOT, measured_gravity, measured_gyro, measured_joint_pos


def gyro(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return measured_gyro(env, asset_cfg)


def gravity(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    return measured_gravity(env, asset_cfg)


def encoder_joint_pos_rel(env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return measured_joint_pos(env, asset_cfg)[:, ids] - robot.data.default_joint_pos[:, ids]


class encoder_joint_vel(ManagerTermBase):
    """Each joint's velocity as the difference of two encoder readings one control step apart.

    Computed once per step (a second read in the same step returns the same value); zero on
    an episode's first step, when there is no earlier reading.
    """

    def __init__(self, cfg: ObservationTermCfg, env) -> None:
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", ROBOT)
        robot = env.scene[self.asset_cfg.name]
        self.previous = measured_joint_pos(env, self.asset_cfg).clone()
        self.value = torch.zeros_like(robot.data.joint_pos)
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.stamp = -1

    def reset(self, env_ids=None) -> None:
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
        stamp = int(env.common_step_counter)
        if stamp != self.stamp:
            reading = measured_joint_pos(env, asset_cfg)
            velocity = (reading - self.previous) / env.step_dt
            self.value = torch.where(self.fresh[:, None], 0.0, velocity)
            self.previous = reading
            self.fresh[:] = False
            self.stamp = stamp
        return self.value[:, asset_cfg.joint_ids]


__all__ = ["encoder_joint_pos_rel", "encoder_joint_vel", "gravity", "gyro"]
