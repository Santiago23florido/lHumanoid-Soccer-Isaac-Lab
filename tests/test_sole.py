"""The sole point as the centre of a foot's contact patch (``common/sole.py``)."""

from __future__ import annotations

import numpy as np

from humanoid_transfer.common.sole import (
    contact_patch,
    cube_points,
    cylinder_points,
    sphere_points,
)


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def _plate(length=0.22, width=0.08, thick=0.02, offset=(0.03, 0.0, -0.05)) -> np.ndarray:
    """A flat foot plate, its link origin above and behind its centre (as on the G1/H1)."""
    return cube_points(1.0) * np.array([length, width, thick]) + np.array(offset)


def test_flat_sole_is_the_bottom_face_centre():
    """Upright link (z up), sole pitched 0.01 rad: the old rule's point, within 0.1 mm."""
    plate = _plate()
    out = contact_patch(plate, _rot("y", -0.01))
    old = [0.03, 0.0, -0.05 - 0.01]
    assert np.allclose(out["sole_point_link"], old, atol=2e-3)
    assert abs(out["sole_point_link"][0] - 0.03) < 1e-3 and abs(out["sole_point_link"][1]) < 1e-9
    assert out["patch_length"] > 0.2


def test_upright_disc_with_a_lateral_axis_touches_at_the_rim_bottom():
    """Cassie's toe: a disc whose axis is the link's z, the link's z pointing to the world's -y.

    The bounding-box rule put the sole on the disc's face, at the link's z = -half thickness:
    in the world, 2 cm to the side and a radius above the ground. The patch is the rim's
    bottom, in the disc's mid-plane.
    """
    radius, thick, centre = 0.058, 0.04, np.array([0.03, 0.03, 0.0])
    disc = cylinder_points(radius, thick, "Z") + centre
    turn = _rot("x", np.pi / 2) @ _rot("z", np.deg2rad(40))  # link z -> world -y, tilted in-plane
    out = contact_patch(disc, turn)
    world = turn @ np.array(out["sole_point_link"])
    centre_world = turn @ centre
    assert abs(world[1] - centre_world[1]) < 1e-6  # in the disc's mid-plane: no lateral offset
    assert abs(world[0] - centre_world[0]) < 2e-3  # under the centre
    assert abs(world[2] - (centre_world[2] - radius)) < 1e-4  # on the ground
    old = turn @ np.array([0.03, 0.03, -thick / 2])  # the bounding-box rule's point
    assert abs(old[1] - centre_world[1]) > 0.019 and old[2] - world[2] > 0.05


def test_mirrored_feet_give_mirrored_soles():
    """Two feet that are mirror images in the world have mirror-image sole points there."""
    disc = cylinder_points(0.058, 0.04, "Z") + np.array([0.03, 0.03, 0.0])
    turn = _rot("x", np.pi / 2) @ _rot("z", np.deg2rad(40))
    mirror = np.diag([1.0, -1.0, 1.0])
    left = turn @ np.array(contact_patch(disc, turn)["sole_point_link"])
    right_points = (disc @ turn.T) @ mirror  # the mirror image, in the world
    right = np.array(contact_patch(right_points, np.eye(3))["sole_point_link"])
    assert np.allclose(right, mirror @ left, atol=1e-6)


def test_sphere_touches_at_its_lowest_point():
    out = contact_patch(sphere_points(0.05) + np.array([0.0, 0.0, -0.1]), _rot("y", 0.3))
    world = _rot("y", 0.3) @ np.array(out["sole_point_link"])
    centre = _rot("y", 0.3) @ np.array([0.0, 0.0, -0.1])
    assert np.allclose(world[:2], centre[:2], atol=2e-3)
    assert abs(world[2] - (centre[2] - 0.05)) < 1e-4
