"""Fit a NAO joint's drive to how it followed its targets, hanging in a harness.

One joint moves its distal subtree while the rest of the robot holds still, so
the joint is a single degree of freedom:

``(I + A) q'' = clip(K (q*(t - d) - q) - D q', ±τ_max) - τ_f sign(q') + τ_g(q)``

with ``I`` the subtree's inertia about the axis and ``τ_g`` its gravity torque
(both from the URDF at the recorded posture, gravity linearised about it), ``A``
the reflected rotor inertia, ``K`` and ``D`` the drive's gains, ``d`` the latency,
``τ_max`` the torque ceiling and ``τ_f`` the gearbox's dry friction. The targets
``q*`` are those sent, held over each recorded cycle. :func:`fit_joint` finds
``(K, D, d, A, τ_max, τ_f)`` minimising the squared error between the simulated
and measured angles over every excitation of the joint; ``tests/test_actuator_fit.py``
recovers known parameters from synthetic records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..assets import nao_kinematics as nk

SUBSTEP = 0.0005
"""Integration step, s."""


@dataclass
class Plant:
    inertia: float
    """The distal subtree's inertia about the joint axis, kg m^2."""
    gravity: float
    """Gravity torque at the reference angle, N m."""
    gravity_slope: float
    """Its change per radian, N m / rad."""
    reference: float
    """The angle the gravity torque is linearised about, rad."""


@dataclass
class Drive:
    stiffness: float
    damping: float
    delay: float
    armature: float
    ceiling: float
    friction: float


def plant_for(joint: str, posture: dict[str, float], step: float = 1e-4) -> Plant:
    """The hanging joint's inertia and gravity torque from the URDF at ``posture``."""
    inertia = nk.composite_inertia_about_joint(joint, posture)
    zero = np.zeros(3)

    def gravity(angle: float) -> float:
        pose = {**posture, joint: angle}
        # static_joint_torque holds the subtree against gravity; gravity's torque is its negative.
        return -nk.static_joint_torque(joint, zero, zero, pose)

    q0 = float(posture.get(joint, 0.0))
    slope = (gravity(q0 + step) - gravity(q0 - step)) / (2 * step)
    return Plant(inertia=inertia, gravity=gravity(q0), gravity_slope=slope, reference=q0)


def simulate(
    plant: Plant, drive: Drive, time: np.ndarray, target: np.ndarray, q0: float, v0: float = 0.0
):
    """The joint's angle at each recorded ``time``, its targets held over each interval."""
    t_end = time[-1]
    steps = int(np.ceil((t_end - time[0]) / SUBSTEP)) + 1
    grid = time[0] + SUBSTEP * np.arange(steps)
    # The target ``delay`` earlier, interpolated between the recorded cycles: the drive
    # low-passes far below the 83 Hz cycle, and the fit needs the error smooth in ``delay``.
    held = np.interp(grid - drive.delay, time, target)
    q, v = q0, v0
    out = np.empty(steps)
    mass = plant.inertia + drive.armature
    for k in range(steps):
        out[k] = q
        torque = drive.stiffness * (held[k] - q) - drive.damping * v
        torque = min(max(torque, -drive.ceiling), drive.ceiling)
        torque += plant.gravity + plant.gravity_slope * (q - plant.reference)
        if v != 0.0:
            torque -= drive.friction * np.sign(v)
        elif abs(torque) <= drive.friction:
            torque = 0.0
        else:
            torque -= drive.friction * np.sign(torque)
        v += SUBSTEP * torque / mass
        q += SUBSTEP * v
    return np.interp(time, grid, out)


NAMES = ("stiffness", "damping", "delay", "armature", "ceiling", "friction")
LOWER = np.array([0.5, 0.01, 0.0, 0.0, 0.3, 0.0])
UPPER = np.array([200.0, 20.0, 0.12, 0.05, 15.0, 1.0])


def fit_joint(
    plant: Plant, records: list[dict], start: Drive, iterations: int = 200
) -> tuple[Drive, float]:
    """Fit a drive to ``records`` (each: ``time``, ``target``, ``position`` of the joint).

    Least squares on the angle error, in the logarithm of every parameter (all positive,
    bounded by :data:`LOWER` and :data:`UPPER`), from ``start``. Returns the drive and the
    RMS angle error, rad.
    """

    def unpack(x: np.ndarray) -> Drive:
        return Drive(**dict(zip(NAMES, (float(v) for v in np.exp(x)), strict=True)))

    def residuals(x: np.ndarray) -> np.ndarray:
        drive = unpack(x)
        return np.concatenate(
            [
                simulate(plant, drive, r["time"], r["target"], float(r["position"][0]))
                - r["position"]
                for r in records
            ]
        )

    low, high = np.log(LOWER + 1e-6), np.log(UPPER)
    x0 = np.clip(
        np.log(np.maximum([getattr(start, n) for n in NAMES], 1e-6)), low + 1e-6, high - 1e-6
    )
    result = least_squares(residuals, x0, bounds=(low, high), max_nfev=iterations, diff_step=1e-3)
    drive = unpack(result.x)
    return drive, float(np.sqrt(np.mean(residuals(result.x) ** 2)))


__all__ = ["Drive", "Plant", "fit_joint", "plant_for", "simulate"]
