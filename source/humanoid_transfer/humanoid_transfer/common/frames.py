"""Rotations and the heading frame, for any floating-base robot.

Two conventions are fixed here because getting either wrong raises nothing:

* Quaternions are **w-x-y-z**, which is what Isaac Lab reports
  (``root_quat_w``, ``body_link_quat_w``). A quaternion read in x-y-z-w order
  is still a unit quaternion and still produces a rotation matrix -- just the
  wrong one.
* The **heading frame** is the world frame turned about the vertical by the
  yaw of a chosen body. Expressing a state in it removes where the robot is
  and which way it faces, which is what makes two walks comparable at all.

Batched throughout: every function accepts leading dimensions and keeps them.
"""

from __future__ import annotations

import numpy as np


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix of a w-x-y-z quaternion, shape ``(..., 4) -> (..., 3, 3)``.

    The quaternion is normalised first. Simulator output is unit to float
    precision; a test building one by hand is not always.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def axis_angle_to_matrix(axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """Rodrigues rotation of ``angle`` radians about ``axis``. Batched over both."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    theta = np.asarray(angle, dtype=float)[..., None, None]
    x, y, z = np.moveaxis(a, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def yaw_of(rotation: np.ndarray) -> np.ndarray:
    """Heading angle of a body, in radians: where its x axis points, seen from above.

    Defined from the projection of the body x axis onto the ground plane, which
    is the only definition that does not depend on an Euler-angle order. It is
    singular when the x axis points straight up or down -- a body lying on its
    back has no heading -- and a walking robot is never there.
    """
    r = np.asarray(rotation, dtype=float)
    return np.arctan2(r[..., 1, 0], r[..., 0, 0])


def yaw_matrix(yaw: np.ndarray | float) -> np.ndarray:
    """Rotation about the world vertical by ``yaw`` radians, ``(...) -> (..., 3, 3)``."""
    angle = np.asarray(yaw, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(c), np.ones_like(c)
    return np.stack(
        [
            np.stack([c, -s, zero], axis=-1),
            np.stack([s, c, zero], axis=-1),
            np.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )


def rotate(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply ``rotation`` (..., 3, 3) to ``vectors`` (..., 3)."""
    return np.einsum("...ij,...j->...i", rotation, vectors)


def rotate_inverse(rotation: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Apply the transpose of ``rotation``: world vectors into the rotated frame."""
    return np.einsum("...ji,...j->...i", rotation, vectors)


__all__ = [
    "axis_angle_to_matrix",
    "quat_to_matrix",
    "rotate",
    "rotate_inverse",
    "yaw_matrix",
    "yaw_of",
]
