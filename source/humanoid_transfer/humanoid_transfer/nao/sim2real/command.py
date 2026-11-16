"""The contract command with the robot's own estimator in the loop.

:class:`~humanoid_transfer.transfer.tasks.tracking.command.ReferenceCommand`
estimates ``ẑ*`` from the simulator's link states and places the reference path
with the simulator's world position. :class:`RobotReferenceCommand` can replace
both with what the robot computes
(:class:`~humanoid_transfer.nao.deploy.estimator.Estimator`, the NumPy code the
robot runs, transcribed to torch: :mod:`.estimator_torch`) from the simulated sensors of
:mod:`~humanoid_transfer.nao.sim2real.state`:

* ``robot_estimator``: ``ẑ*`` from forward kinematics of the measured joints, the
  IMU's tilt, the gyroscope, finite differences and the feet's loads;
* ``robot_odometry``: the body's position and heading for the reference path from
  the estimator's leg odometry and integrated yaw;
* ``load_source``: where the loads come from -- ``fsr`` (each foot's contact
  force through its force sensors' gain, offset and noise), ``contact`` (the
  contact force exactly), ``encoder`` (the contract's learned encoder, which
  reads the base's true linear velocity) or ``encoder_no_vel`` (the same encoder
  with that velocity zeroed).

The reward and every privileged quantity keep the truth. Both options off, it
is the plain command.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from ...common.skill_state import SKILL_LAYOUT
from ...transfer.tasks.tracking.command import ReferenceCommand, ReferenceCommandCfg
from ..deploy.kinematics import BodyModel
from .body_model import body_model
from .estimator_torch import TorchEstimator
from .state import measured_gravity, measured_gyro, measured_joint_pos, robot_errors

_LOAD = SKILL_LAYOUT.slice_of("foot_load")
LOAD_SOURCES = ("fsr", "contact", "encoder", "encoder_no_vel")


class RobotReferenceCommand(ReferenceCommand):
    cfg: RobotReferenceCommandCfg

    def __init__(self, cfg: RobotReferenceCommandCfg, env) -> None:
        super().__init__(cfg, env)
        if cfg.load_source not in LOAD_SOURCES:
            raise ValueError(f"load_source must be one of {LOAD_SOURCES}, not {cfg.load_source!r}")
        if cfg.load_source.startswith("encoder") and self.encoder is None:
            raise ValueError("an encoder's loads need use_encoder=True and a contract with one")
        robot = self.robot
        spec = body_model(
            list(robot.data.body_names),
            list(robot.data.joint_names),
            cfg.sole_points,
            cfg.sole_corners if cfg.sole_corners is not None else [[p] for p in cfg.sole_points],
            cfg.leg_length,
            base_body=cfg.base_body,
            foot_bodies=cfg.foot_bodies,
        )
        self.body_model_spec = spec
        """The robot's body model, as :func:`~.body_model.body_model` built it here."""
        self.robot_model = BodyModel(spec)
        names = list(robot.data.joint_names)
        self.measured_ids = torch.tensor(
            [names.index(n) for n in self.robot_model.joint_names], device=self.device
        )
        # The robot's NumPy estimator, transcribed to torch for every environment at once
        # (tests/test_nao_estimator_torch.py holds the two together).
        self.estimator = TorchEstimator(self.robot_model, self.num_envs, env.step_dt, self.device)
        self._robot_z = None
        self._robot_frame = None

    # -- the robot's sensors ------------------------------------------------

    def foot_loads(self) -> torch.Tensor:
        """``(N, 2)`` each foot's load in units of ``m g``, from ``load_source``."""
        source = self.cfg.load_source
        if source.startswith("encoder"):
            reading = self.encoder_input()
            if source == "encoder_no_vel":
                reading = reading.clone()
                reading[:, :3] = 0.0
            with torch.no_grad():
                learned = self.encoder(reading)
            # z* in the body's units: the loads are already in units of m g.
            return self.generator.skill_norm.inverse(learned)[:, _LOAD]
        weight = self.units.mass * self.units.gravity
        force = self.sensor.data.net_forces_w[:, self.sensor_ids, 2] / weight
        if source == "contact":
            return force
        errors = robot_errors(self._env)
        noise = self.cfg.fsr_noise * torch.randn_like(force)
        return (force * errors.fsr_scale + errors.fsr_offset + noise).clamp(min=0.0)

    def _robot_update(self) -> None:
        """One step of the robot's estimator, for every environment."""
        env = self._env
        q = measured_joint_pos(env)[:, self.measured_ids]
        gyro = measured_gyro(env)
        if self.cfg.gyro_noise > 0.0:
            gyro = gyro + self.cfg.gyro_noise * torch.randn_like(gyro)
        self.estimator.reset(self.just_reset)
        inputs = (q, measured_gravity(env), gyro, self.foot_loads())
        self.robot_inputs = inputs
        """What the estimator read this step: joints, gravity, gyroscope, loads."""
        self._robot_z = self.estimator.update(*inputs)
        position, yaw = self.estimator.position, self.estimator.yaw
        self._robot_frame = (position, yaw)

    # -- hooks into the plain command ----------------------------------------

    def _update_command(self) -> None:
        if self.cfg.robot_estimator or self.cfg.robot_odometry:
            self._robot_update()
        super()._update_command()

    def frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.robot_odometry and self._robot_frame is not None:
            return self._robot_frame
        return super().frame()

    def _estimate(self) -> torch.Tensor:
        if not self.cfg.robot_estimator:
            return super()._estimate()
        estimate = self.generator.skill_norm(self._robot_z)
        if self.cfg.estimate_noise > 0.0:
            estimate = estimate + self.precision * self.cfg.estimate_noise * torch.randn_like(
                estimate
            )
        return estimate


@configclass
class RobotReferenceCommandCfg(ReferenceCommandCfg):
    class_type: type = RobotReferenceCommand

    robot_estimator: bool = False
    """``ẑ*`` from the robot's estimator on the simulated sensors."""

    robot_odometry: bool = False
    """The reference path placed from the robot's leg odometry and integrated yaw."""

    load_source: str = "fsr"
    """Where the robot's estimator takes the feet's loads from (:data:`LOAD_SOURCES`)."""

    fsr_noise: float = 0.02
    """Standard deviation of each foot's force-sensor reading, units of ``m g``."""

    gyro_noise: float = 0.01
    """Standard deviation of the gyroscope's reading the estimator integrates, rad/s."""


def robot_command(cfg: ReferenceCommandCfg, **options) -> RobotReferenceCommandCfg:
    """``cfg`` as a :class:`RobotReferenceCommandCfg`, every field kept, ``options`` set."""
    robot = RobotReferenceCommandCfg()
    for key, value in vars(cfg).items():
        setattr(robot, key, value)
    robot.class_type = RobotReferenceCommand
    for key, value in options.items():
        if not hasattr(robot, key):
            raise KeyError(f"no command option {key!r}")
        setattr(robot, key, value)
    return robot


__all__ = ["LOAD_SOURCES", "RobotReferenceCommand", "RobotReferenceCommandCfg", "robot_command"]
