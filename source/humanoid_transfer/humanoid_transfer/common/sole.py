"""Where a foot meets the ground: the centre of its contact patch in a standing posture.

``z*`` places each foot at its sole point, and the leg length is measured to it, so the
sole point is part of every body's units. The rule that placed it at the centre of the
bottom of the collision bounding box *in the foot link's frame* assumed the link's ``z``
axis points up. On the G1 and the H1 it does; on Cassie the toe link's ``z`` is lateral
(both toes' ``z`` point to the robot's right) and its collision shape is an upright disc,
so that rule put both sole points on a side face of the disc: 2 cm to the robot's left of
the feet, 4-8 cm above the ground, a symmetric stance reading as an asymmetric one.

The rule here asks the geometry, not the frame: sample the foot's collision surface, turn
it into the world with the link at a standing posture, lay it on its support (the posture's
small tilt taken up, as a free-standing biped's base takes it up), and take the points within
a thin band above the lowest one -- the patch that touches flat ground. The sole point is the
centre of the patch's extent in the ground plane, on the foot's surface there, expressed back
in the link frame. A flat sole yields its bottom face's centre (the old rule, where that rule
was right); a rod lying forward, the middle of its bottom line; a disc on its rim, the bottom
of the rim; a sphere, its lowest point.
"""

from __future__ import annotations

import numpy as np

PATCH_BAND = 0.02
"""Height of the contact patch, as a fraction of the foot's largest extent: points this close
to the lowest one touch flat ground (4 mm on a 0.2 m foot); a 6 cm disc's patch is a 3.4 cm
chord of its rim, centred on the bottom."""

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
    side = _on_axis(
        np.repeat(rim[None], len(levels), 0), np.repeat(levels[:, None], count, 1), axis
    )
    caps = []
    for fraction in np.linspace(0.0, 1.0, 8)[1:]:
        ring = _circle(radius * fraction, count)
        for end in (-height / 2.0, height / 2.0):
            caps.append(_on_axis(ring, np.full(count, end), axis))
    return np.concatenate([side.reshape(-1, 3), *caps], axis=0)


def sphere_points(radius: float, count: int = 20000) -> np.ndarray:
    """Surface samples of a sphere (a Fibonacci lattice). A sphere has no support plane, and
    laid on its sampled hull it settles within about half a facet of its lowest point: 0.25 mm
    on a 5 cm sphere at this count, 2 mm at 2000."""
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
    ends = [
        ball[ball[:, k] >= 0.0] + _unit(k) * height / 2.0,
        ball[ball[:, k] < 0.0] - _unit(k) * height / 2.0,
    ]
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


def _hull(points: np.ndarray):
    """The convex hull of ``points`` (K, 3): its facets' outward unit normals and offsets
    (``n . p + d <= 0`` inside), vertex indices and areas."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    corners = points[hull.simplices]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1
    )
    return hull.equations[:, :3], hull.equations[:, 3], hull.simplices, areas


def _turn_onto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation taking unit vector ``a`` to unit vector ``b`` about their common normal."""
    axis = np.cross(a, b)
    sine, cosine = float(np.linalg.norm(axis)), float(np.dot(a, b))
    if sine < 1e-15:
        return np.eye(3)
    k = axis / sine
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + sine * cross + (1.0 - cosine) * cross @ cross


LEVEL_TOLERANCE = 1e-6
"""The support plane is level when the last turn is below this (rad): it moves the end of a
0.2 m foot by 0.2 micrometres."""


def contact_patch(
    points_link: np.ndarray,
    rotation: np.ndarray,
    band: float = PATCH_BAND,
    iterations: int = 50,
) -> dict:
    """The sole of a foot whose collision surface is ``points_link`` (K, 3), in its link frame,
    with the link turned by ``rotation`` (3, 3; link to world) as it stands.

    A posture's foot is rarely exactly flat (a few tenths of a degree on the G1 and the H1,
    3 degrees on Cassie's rod-shaped toe), and on flat ground a free-standing biped's feet lie
    on their support, its base taking up the tilt. So the foot is first laid on its support:
    the mean normal of the facets of its convex hull (PhysX collides a convex shape) that
    reach into the patch, each weighted by its footprint on the ground, is turned straight
    down and the patch re-taken, until the turn is below ``LEVEL_TOLERANCE``. A flat sole's
    patch is its bottom face and a rod's its bottom line; on a curved surface the patch is a
    cap around the lowest point, whose mean normal is already vertical, so a sphere or a
    disc's rim is not turned. The sole is then the centre of the patch's extent in the ground
    plane, on the hull's bottom there. A flat sole yields its bottom face's centre at any
    small tilt; a rod, the middle of the line it rests on; a disc on its rim, the bottom of
    the rim; a sphere, its lowest point.

    Returns the sole point in the link frame, the angle the foot was turned to lie on its support
    (``support_tilt``, rad), the foot's length and width (its extent along the world's x and y,
    lying on it) and the patch's extent (``patch_length``, ``patch_width``).
    """
    points_link = np.asarray(points_link, dtype=float)
    normals, offsets, simplices, areas = _hull(points_link)
    down = np.array([0.0, 0.0, -1.0])
    turn = np.asarray(rotation, dtype=float)
    for _ in range(iterations):
        world = points_link @ turn.T
        size = float(np.ptp(world, axis=0).max())
        top = world[:, 2].min() + band * size
        facing = normals @ turn.T
        inside = (world[simplices, 2].min(axis=1) <= top) & (facing[:, 2] < 0.0)
        # Each facet weighs its footprint on the ground (area times the downward part of its
        # normal), so a side face that dips into the band while the foot is tilted, and that
        # bears nothing once it lies flat, drops out as it levels.
        weight = areas[inside] * -facing[inside, 2]
        mean = (facing[inside] * weight[:, None]).sum(axis=0)
        step = _turn_onto(mean / np.linalg.norm(mean), down)
        turn = step @ turn
        if np.arccos(np.clip(step[2, 2], -1.0, 1.0)) < LEVEL_TOLERANCE:
            break
    else:
        raise RuntimeError(f"the foot's support did not settle in {iterations} turns")
    world = points_link @ turn.T
    size = float(np.ptp(world, axis=0).max())
    patch = world[world[:, 2] <= world[:, 2].min() + band * size]
    x = (patch[:, 0].min() + patch[:, 0].max()) / 2.0
    y = (patch[:, 1].min() + patch[:, 1].max()) / 2.0
    facing = normals @ turn.T
    below = facing[:, 2] < -1e-9
    n, d = facing[below], offsets[below]
    height = float((-(n[:, 0] * x + n[:, 1] * y + d) / n[:, 2]).max())
    relative = turn @ np.asarray(rotation, dtype=float).T
    return {
        "sole_point_link": (turn.T @ np.array([x, y, height])).tolist(),
        "support_tilt": float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))),
        "length": float(np.ptp(world[:, 0])),
        "width": float(np.ptp(world[:, 1])),
        "patch_length": float(np.ptp(patch[:, 0])),
        "patch_width": float(np.ptp(patch[:, 1])),
    }


__all__ = [
    "LEVEL_TOLERANCE",
    "PATCH_BAND",
    "capsule_points",
    "contact_patch",
    "cube_points",
    "cylinder_points",
    "sphere_points",
]
