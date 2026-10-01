"""Centroidal quantities of a multibody system, from per-body state.

A simulator reports bodies; balance and gait are about the whole. This module
does the aggregation once, for any tree of rigid bodies, so that neither robot
package nor any script carries its own copy -- the G1 rollout capture had one,
and it dropped a term its own docstring listed.

The quantities, for bodies ``i`` with mass ``m_i``, centre of mass ``r_i``,
velocity ``v_i`` (at ``r_i``), angular velocity ``w_i`` and inertia ``I_i``
about ``r_i``:

.. math::

    c = \\frac{1}{M}\\sum_i m_i r_i, \\qquad
    \\dot c = \\frac{1}{M}\\sum_i m_i v_i,

    k_G = \\underbrace{\\sum_i (r_i - c) \\times m_i (v_i - \\dot c)}_{\\text{orbital}}
        + \\underbrace{\\sum_i R_i I_i R_i^\\top \\, w_i}_{\\text{spin}}

``k_G`` is the angular part of the centroidal momentum (Orin, Goswami and Lee,
2013). The linear part is ``M c_dot`` and carries nothing ``c_dot`` does not.

Inertia convention
------------------
``I_i`` is taken about the body's centre of mass and expressed in the body's
**link** frame, with ``R_i`` the world-from-link rotation. That is what Isaac
Lab's ``default_inertia`` holds ("with respect to the center of mass, expressed
in the articulation links' actor frame", ``articulation_data.py``), so ``R_i``
must come from ``body_link_quat_w``. ``body_com_quat_w`` composes the link
rotation with the principal-axes rotation and would rotate the tensor twice.
Nothing raises if that is done; the spin term is just wrong.

Pure NumPy, batched over any leading dimensions: bodies are axis ``-1`` of
``masses`` and axis ``-2`` of every per-body vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _weights(masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(masses, dtype=float)
    total = m.sum(axis=-1)
    if np.any(total <= 0.0):
        raise ValueError("total mass must be positive")
    return m, total


def center_of_mass(masses: np.ndarray, com_pos: np.ndarray) -> np.ndarray:
    """Whole-body centre of mass, shape ``(..., 3)``."""
    m, total = _weights(masses)
    return np.einsum("...b,...bi->...i", m, np.asarray(com_pos, dtype=float)) / total[..., None]


def com_velocity(masses: np.ndarray, com_lin_vel: np.ndarray) -> np.ndarray:
    """Velocity of the whole-body centre of mass, shape ``(..., 3)``.

    ``com_lin_vel`` must be each body's velocity *at its own centre of mass*
    (Isaac Lab ``body_com_lin_vel_w``), not at the link origin.
    """
    return center_of_mass(masses, com_lin_vel)


def com_acceleration(masses: np.ndarray, com_lin_acc: np.ndarray) -> np.ndarray:
    """Acceleration of the whole-body centre of mass, shape ``(..., 3)``."""
    return center_of_mass(masses, com_lin_acc)


def world_inertia(inertia_local: np.ndarray, link_rot: np.ndarray) -> np.ndarray:
    """Each body's inertia about its centre of mass, in the world frame.

    Args:
        inertia_local: ``(..., B, 3, 3)``, about the body CoM, in the link frame.
        link_rot: ``(..., B, 3, 3)``, world-from-link rotations.
    """
    r = np.asarray(link_rot, dtype=float)
    return r @ np.asarray(inertia_local, dtype=float) @ np.swapaxes(r, -1, -2)


@dataclass(frozen=True)
class CentroidalAngularMomentum:
    """Angular momentum about the whole-body centre of mass, split by origin."""

    orbital: np.ndarray
    """``sum (r_i - c) x m_i (v_i - c_dot)``: bodies moving relative to the CoM."""

    spin: np.ndarray
    """``sum R_i I_i R_i^T w_i``: bodies rotating about their own centres."""

    @property
    def total(self) -> np.ndarray:
        return self.orbital + self.spin


def angular_momentum(
    masses: np.ndarray,
    inertia_local: np.ndarray,
    link_rot: np.ndarray,
    com_pos: np.ndarray,
    com_lin_vel: np.ndarray,
    ang_vel: np.ndarray,
) -> CentroidalAngularMomentum:
    """Centroidal angular momentum ``k_G``, orbital and spin parts, in the world frame.

    Both parts are returned rather than only their sum because which one
    dominates is a measurement, not an assumption: the G1 capture once
    documented the spin term and silently omitted it on the grounds that "the
    translational term dominates", which nobody had checked.
    """
    m, _ = _weights(masses)
    positions = np.asarray(com_pos, dtype=float)
    velocities = np.asarray(com_lin_vel, dtype=float)
    c = center_of_mass(m, positions)
    c_dot = com_velocity(m, velocities)

    relative_pos = positions - c[..., None, :]
    relative_momentum = m[..., None] * (velocities - c_dot[..., None, :])
    orbital = np.cross(relative_pos, relative_momentum).sum(axis=-2)

    inertia = world_inertia(inertia_local, link_rot)
    spin = np.einsum("...bij,...bj->...i", inertia, np.asarray(ang_vel, dtype=float))
    return CentroidalAngularMomentum(orbital=orbital, spin=spin)


def composite_inertia(
    masses: np.ndarray,
    inertia_local: np.ndarray,
    link_rot: np.ndarray,
    com_pos: np.ndarray,
) -> np.ndarray:
    """Inertia of the whole body, frozen in its current posture, about its CoM.

    The parallel-axis (Steiner) sum of every body's world-frame inertia. It is
    the tensor that maps a rigid rotation of the whole posture to ``k_G``, so a
    system turning as one body has ``k_G = I_c w`` exactly -- which is how the
    tests check :func:`angular_momentum`. Shape ``(..., 3, 3)``.
    """
    m, _ = _weights(masses)
    positions = np.asarray(com_pos, dtype=float)
    offset = positions - center_of_mass(m, positions)[..., None, :]
    squared = np.einsum("...bi,...bi->...b", offset, offset)
    steiner = m[..., None, None] * (
        squared[..., None, None] * np.eye(3) - np.einsum("...bi,...bj->...bij", offset, offset)
    )
    return (world_inertia(inertia_local, link_rot) + steiner).sum(axis=-3)


__all__ = [
    "CentroidalAngularMomentum",
    "angular_momentum",
    "center_of_mass",
    "com_acceleration",
    "com_velocity",
    "composite_inertia",
    "world_inertia",
]
