"""The robot's estimate of ``z*`` and of where it is, from its own sensors only.

Inputs, once per control step:

* the measured joint angles (encoders), in the body model's joint order;
* the IMU's gravity direction in the base frame (its tilt) and its gyroscope;
* each foot's load, in units of ``m g`` (the feet's force sensors, summed per foot).

From them:

* the base's orientation: roll and pitch from gravity, yaw integrated from the
  gyroscope (``z*`` does not depend on yaw; its rate does);
* every body's pose by forward kinematics, in a frame turned as the world and
  centred on the base;
* velocities by finite differences over one step: the base's translation is the
  same for every body and cancels once velocities are measured from a stance
  foot's sole, taken to stand still (leg odometry) -- one loaded at both ends of
  the step (:func:`~.skill.stance_velocity`);
* ``z*`` from those (:func:`~.skill.proprioceptive_skill_state`);
* leg odometry of the whole-body centre of mass in the plane (its velocity from
  the stance foot, integrated) and the yaw: where the robot is relative to where
  it started, for the reference path.

The same object runs in the simulator, batched over environments, fed by the
simulated sensors (:mod:`humanoid_transfer.nao.sim2real`): what the policy reads
in training is what this code computes on the robot.
"""

from __future__ import annotations

import numpy as np

from .kinematics import BodyModel
from .skill import proprioceptive_skill_state, stance_velocity


def tilt_from_gravity(gravity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll and pitch ``(...)`` from the gravity direction in the base frame ``(..., 3)``.

    With ``R = Rz(yaw) Ry(pitch) Rx(roll)`` the base's orientation, gravity reads
    ``R^T (0, 0, -1) = (sin pitch, -sin roll cos pitch, -cos roll cos pitch)``.
    """
    g = gravity / np.linalg.norm(gravity, axis=-1, keepdims=True)
    pitch = np.arcsin(np.clip(g[..., 0], -1.0, 1.0))
    roll = np.arctan2(-g[..., 1], -g[..., 2])
    return roll, pitch


def rpy_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack(
        [
            np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
            np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
            np.stack([-sp, cp * sr, cp * cr], -1),
        ],
        axis=-2,
    )


def yaw_rate(roll: np.ndarray, pitch: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """The rate of the base's yaw ``(...)`` from its body-frame angular velocity ``(..., 3)``."""
    return (np.sin(roll) * gyro[..., 1] + np.cos(roll) * gyro[..., 2]) / np.cos(pitch)


def rotation_rate(now: np.ndarray, before: np.ndarray, dt: float) -> np.ndarray:
    """``(..., 3)`` the world-frame angular velocity taking ``before`` to ``now`` in ``dt``."""
    w = now @ np.swapaxes(before, -1, -2)
    skew = 0.5 * (w - np.swapaxes(w, -1, -2))
    return np.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], axis=-1) / dt


class Estimator:
    """``z*`` and leg odometry for ``count`` robots, one control step at a time."""

    def __init__(self, model: BodyModel, count: int, dt: float, pivot: str = "corner") -> None:
        self.model = model
        self.count = count
        self.dt = float(dt)
        bodies = len(model.names)
        self.yaw = np.zeros(count)
        self.rot = np.tile(np.eye(3), (count, bodies, 1, 1))
        self.com = np.zeros((count, bodies, 3))
        self.pivot = pivot
        """Which point of the stance foot is taken to stand still (:data:`~.skill.PIVOTS`)."""
        self.load = np.zeros((count, 2))
        """Each foot's load at the previous step."""
        self.fresh = np.ones(count, dtype=bool)
        """Robots whose next step starts their history: no velocity, odometry at the origin."""
        self.position = np.zeros((count, 2))
        """The whole-body centre of mass in the odometry frame, metres."""

    def reset(self, which: np.ndarray | None = None) -> None:
        """Start again, for ``which`` robots (a boolean mask) or all."""
        self.fresh[slice(None) if which is None else which] = True

    def update(
        self, q: np.ndarray, gravity: np.ndarray, gyro: np.ndarray, foot_load: np.ndarray
    ) -> np.ndarray:
        """Read one step's sensors ``(count, ...)``; return ``z*`` ``(count, size)`` in the
        body's units (before the contract's normalisation)."""
        model, fresh = self.model, self.fresh
        roll, pitch = tilt_from_gravity(gravity)
        yaw = self.yaw + self.dt * yaw_rate(roll, pitch, gyro)
        yaw = np.where(fresh, 0.0, yaw)
        base = rpy_matrix(roll, pitch, yaw)
        rot_b, pos_b = model.forward(q)
        rot = base[:, None] @ rot_b
        com = np.einsum("nij,nbj->nbi", base, model.centres(rot_b, pos_b))
        velocity = np.where(fresh[:, None, None], 0.0, (com - self.com) / self.dt)
        spin = np.where(fresh[:, None, None], 0.0, rotation_rate(rot, self.rot, self.dt))
        # The base turns as its gyroscope says, not as two filtered tilts differ.
        spin[:, model.base] = np.einsum("nij,nj->ni", base, gyro)
        previous = np.where(fresh[:, None], foot_load, self.load)
        z = proprioceptive_skill_state(
            model, rot, com, velocity, spin, foot_load, self.pivot, previous
        )

        # Leg odometry: the whole-body centre of mass moves at its velocity measured from the
        # stance foot, integrated; it starts where the kinematics put it relative to the base.
        still = stance_velocity(model, rot, com, velocity, spin, foot_load, self.pivot, previous)
        centre = np.einsum("b,nbi->ni", model.mass, com) / model.total_mass
        moving = np.einsum("b,nbi->ni", model.mass, velocity) / model.total_mass - still
        self.position = np.where(
            fresh[:, None], centre[:, :2], self.position + self.dt * moving[:, :2]
        )

        self.yaw, self.rot, self.com, self.load = (
            yaw,
            rot,
            com,
            np.array(foot_load, dtype=np.float64),
        )
        self.fresh = np.zeros(self.count, dtype=bool)
        return z


__all__ = ["Estimator", "rotation_rate", "rpy_matrix", "tilt_from_gravity", "yaw_rate"]
