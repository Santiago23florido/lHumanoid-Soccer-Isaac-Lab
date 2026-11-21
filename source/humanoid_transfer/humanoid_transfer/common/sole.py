"""Where a foot meets the ground: the centre of its contact patch in a standing posture.

``z*`` places each foot at its sole point, and the leg length is measured to it, so the
sole point is part of every body's units. The rule that placed it at the centre of the
bottom of the collision bounding box *in the foot link's frame* assumed the link's ``z``
axis points up. On the G1 and the H1 it does; on Cassie the toe link's ``z`` is lateral
(both toes' ``z`` point to the robot's right) and its collision shape is an upright disc,
so that rule put both sole points on a side face of the disc: 2 cm to the robot's left of
the feet, 4-8 cm above the ground, a symmetric stance reading as an asymmetric one.

The rule here asks the geometry, not the frame: sample the foot's collision surface, turn
it into the world with the link at a standing posture, and take the points within a thin
band above the lowest one -- the patch that touches flat ground. The sole point is the
centre of the patch's extent in the ground plane, at the lowest height, expressed back in
the link frame. A flat sole yields its bottom face's centre (the old rule, where that rule
was right); a disc on its rim yields the bottom of the rim; a sphere its lowest point.
"""

from __future__ import annotations

import numpy as np

PATCH_BAND = 0.02
"""Height of the contact patch, as a fraction of the foot's largest extent: points this close
to the lowest one touch flat ground. A flat sole pitched by 0.01 rad over 0.2 m rises 2 mm
heel to toe, inside the band of a 0.2 m foot (4 mm); a 6 cm disc's patch is a 3.4 cm chord
of its rim, centred on the bottom."""

_AXES = {"X": 0, "Y": 1, "Z": 2}


def _circle(radius: float, count: int) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)


def _on_axis(planar: np.ndarray, along: np.ndarray, axis: str) -> np.ndarray:
    """Points with ``planar`` (..., 2) across ``axis`` and ``along`` (...) on it."""
    k = _AXES[axis]
    across = [i for i in range(3) if i != k]
    points = np.zeros((*along.shape, 3))
    points[..., across[0]] = planar[..., 0]
    points[..., across[1]] = planar[..., 1]
    points[..., k] = along
    return points


def cylinder_points(radius: float, height: float, axis: str = "Z", count: int = 180) -> np.ndarray:
    """Surface samples of a USD ``Cylinder``: its rim at several heights and both caps."""
    rim = _circle(radius, count)
    levels = np.linspace(-height / 2.0, height / 2.0, 9)
    side = _on_axis(np.repeat(rim[None], len(levels), 0), np.repeat(levels[:, None], count, 1), axis)
    caps = []
    for fraction in np.linspace(0.0, 1.0, 8)[1:]:
        ring = _circle(radius * fraction, count)
        for end in (-height / 2.0, height / 2.0):
            caps.append(_on_axis(ring, np.full(count, end), axis))
    return np.concatenate([side.reshape(-1, 3), *caps], axis=0)


def sphere_points(radius: float, count: int = 2000) -> np.ndarray:
    """Surface samples of a sphere (a Fibonacci lattice)."""
    i = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    return radius * np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=-1
    )


def capsule_points(radius: float, height: float, axis: str = "Z", count: int = 180) -> np.ndarray:
    """Surface samples of a USD ``Capsule``: a cylinder of ``height`` with hemispherical ends."""
    ball = sphere_points(radius, 4 * count)
    k = _AXES[axis]
    ends = [ball[ball[:, k] >= 0.0] + _unit(k) * height / 2.0, ball[ball[:, k] < 0.0] - _unit(k) * height / 2.0]
    return np.concatenate([cylinder_points(radius, height, axis, count), *ends], axis=0)


def cube_points(size: float, count: int = 21) -> np.ndarray:
    """Surface samples of a USD ``Cube`` of edge ``size``: a grid on each face."""
    grid = np.linspace(-size / 2.0, size / 2.0, count)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    faces = []
    for k in range(3):
        for end in (-size / 2.0, size / 2.0):
            face = np.zeros((a.size, 3))
            others = [i for i in range(3) if i != k]
            face[:, others[0]], face[:, others[1]], face[:, k] = a.ravel(), b.ravel(), end
            faces.append(face)
    return np.concatenate(faces, axis=0)


def _unit(k: int) -> np.ndarray:
    e = np.zeros(3)
    e[k] = 1.0
    return e


def contact_patch(points_link: np.ndarray, rotation: np.ndarray, band: float = PATCH_BAND) -> dict:
    """The sole of a foot whose collision surface is ``points_link`` (K, 3), in its link frame,
    with the link turned by ``rotation`` (3, 3; link to world) as it stands.

    Returns the sole point in the link frame, the foot's length and width (its extent along the
    world's x and y as it stands) and the patch's extent (``patch_length``, ``patch_width``).
    """
    world = points_link @ rotation.T
    low = world[:, 2].min()
    size = float(np.ptp(world, axis=0).max())
    patch = world[world[:, 2] <= low + band * size]
    centre = np.array(
        [
            (patch[:, 0].min() + patch[:, 0].max()) / 2.0,
            (patch[:, 1].min() + patch[:, 1].max()) / 2.0,
            low,
        ]
    )
    return {
        "sole_point_link": (rotation.T @ centre).tolist(),
        "length": float(np.ptp(world[:, 0])),
        "width": float(np.ptp(world[:, 1])),
        "patch_length": float(np.ptp(patch[:, 0])),
        "patch_width": float(np.ptp(patch[:, 1])),
    }


__all__ = [
    "PATCH_BAND",
    "capsule_points",
    "contact_patch",
    "cube_points",
    "cylinder_points",
    "sphere_points",
]
