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


def test_flat_sole_is_the_bottom_face_centre_at_any_small_tilt():
    """Upright link (z up), sole pitched 0.01 and 0.05 rad, rolled 0.03: the bottom face's
    centre, on the face (the old rule's point where that rule was right), and the turn that
    laid it flat."""
    plate = _plate()
    for pitch in (-0.01, 0.05):
        out = contact_patch(plate, _rot("x", 0.03) @ _rot("y", pitch))
        assert np.allclose(out["sole_point_link"], [0.03, 0.0, -0.06], atol=1e-6)
        assert abs(out["support_tilt"] - np.hypot(pitch, 0.03)) < 1e-3
        # Extents along the world's x: the levelling turns leave the plate yawed by ~1e-4 rad.
        assert abs(out["patch_length"] - 0.22) < 1e-4 and abs(out["length"] - 0.22) < 1e-4


def test_a_tilted_rod_rests_on_the_middle_of_its_bottom_line():
    """Cassie's toe: a rod along the link's x, its link turned so the rod's axis is the world's
    x tilted 0.05 rad. Lying flat it rests on its whole bottom line, whose middle is the sole;
    taken as it stands, the patch would be the lower half of the line."""
    radius, length, centre = 0.02, 0.125, np.array([0.04, 0.01, 0.0])
    rod = cylinder_points(radius, length, "X") + centre
    turn = _rot("y", 0.05) @ _rot("x", 0.4)
    out = contact_patch(rod, turn)
    sole = np.array(out["sole_point_link"])
    below = _rot("x", 0.4).T @ np.array([0.0, 0.0, -radius])  # lying flat, the link's down
    # The sampled rod is a 180-sided prism: it settles on a side within half a side (1 degree)
    # of the round rod's bottom, 0.35 mm along its rim.
    assert np.allclose(sole, centre + below, atol=4e-4)
    assert abs(np.linalg.norm(sole[1:] - centre[1:]) - radius) < 1e-4
    assert abs(out["support_tilt"] - 0.05) < 1e-3 and out["patch_length"] > 0.12


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
