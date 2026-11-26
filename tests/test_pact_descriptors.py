"""Body descriptors: Pi groups that dynamically similar bodies share exactly."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from humanoid_transfer.transfer.latent.pact.descriptors import (
    BODY_FEATURES,
    EFFORT_LIMIT_CLIP,
    JOINT_FEATURES,
    describe,
)

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
BODIES = [name for name in REGISTRY if not name.startswith("$")]


def _card(body: str) -> dict:
    return json.loads((REPO / REGISTRY[body]["card"]).read_text("utf-8"))


def _scaled(card: dict, s: float, k: float) -> dict:
    """The card of a dynamically similar body: lengths x s, masses x k, same g.

    Time scales as sqrt(s); torque as k s; stiffness (torque per rad) as k s;
    damping (torque per rad/s) as k s^1.5; inertia and armature as k s^2.
    """
    t = np.sqrt(s)
    out = copy.deepcopy(card)

    def scale(value, factor):
        return (np.asarray(value, dtype=float) * factor).tolist()

    out["leg_length"] = card["leg_length"] * s
    out["total_mass"] = card["total_mass"] * k
    bodies = out["bodies"]
    bodies["mass"] = scale(card["bodies"]["mass"], k)
    bodies["inertia_link"] = scale(card["bodies"]["inertia_link"], k * s**2)
    bodies["com_offset_link"] = scale(card["bodies"]["com_offset_link"], s)
    for posture in out["postures"].values():
        for key in ("link_pos", "com_pos", "soles", "com"):
            posture[key] = scale(posture[key], s)
        posture["com_height_above_soles"] *= s
        posture["sole_lateral_separation"] *= s
    for row in out["joint_geometry"]:
        row["position_root"] = scale(row["position_root"], s)
    for foot in out["foot"].values():
        for key in ("bbox_min_link", "bbox_max_link", "sole_point_link"):
            foot[key] = scale(foot[key], s)
        foot["length"] *= s
        foot["width"] *= s
    out["joints"]["effort_limits"] = scale(card["joints"]["effort_limits"], k * s)
    out["joints"]["vel_limits"] = scale(card["joints"]["vel_limits"], 1 / t)
    for row in out["drives"]:
        row["effort_limit"] *= k * s
        row["velocity_limit"] /= t
        row["stiffness"] *= k * s
        row["damping"] *= k * s * t
        if row["armature"] is not None:
            row["armature"] *= k * s**2
        if row["saturation_effort"] is not None:
            row["saturation_effort"] *= k * s
    out["control"]["physics_dt"] *= t
    out["control"]["control_period"] *= t
    return out


@pytest.mark.parametrize("body", BODIES)
def test_every_registered_body_has_a_finite_descriptor(body):
    descriptor = describe(_card(body))
    assert descriptor.joints.shape == (len(descriptor.joint_names), len(JOINT_FEATURES))
    assert descriptor.body.shape == (len(BODY_FEATURES),)
    assert np.isfinite(descriptor.joints).all()
    assert np.isfinite(descriptor.body).all()
    effort = descriptor.joints[:, JOINT_FEATURES.index("effort_limit")]
    assert (effort <= EFFORT_LIMIT_CLIP + 1e-9).all()
    axes = descriptor.joints[:, [JOINT_FEATURES.index(f"axis.{a}") for a in "xyz"]]
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("body", BODIES)
def test_tree_matrices_are_consistent(body):
    descriptor = describe(_card(body))
    distance, ancestor = descriptor.tree_distance, descriptor.ancestor
    np.testing.assert_array_equal(distance, distance.T)
    assert (np.diag(distance) == 0).all()
    assert not (ancestor & ancestor.T).any()
    # an ancestor is always at a positive distance
    assert (distance[ancestor] > 0).all()


@pytest.mark.parametrize("s, k", [(0.38, 0.165), (2.0, 7.0)])
def test_dynamically_similar_bodies_share_their_descriptor(s, k):
    card = _card("g1")
    original = describe(card)
    similar = describe(_scaled(card, s, k))
    np.testing.assert_allclose(similar.joints, original.joints, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(similar.body, original.body, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(similar.tree_distance, original.tree_distance)


def test_a_different_body_has_a_different_descriptor():
    g1, h1 = describe(_card("g1")), describe(_card("h1"))
    assert g1.joints.shape[0] != h1.joints.shape[0] or not np.allclose(g1.joints, h1.joints)
    assert not np.allclose(g1.body, h1.body)


def test_the_hip_pitch_moves_a_leg_and_mirrors_across_the_sagittal_plane():
    descriptor = describe(_card("g1"))
    left, right = descriptor.index(["left_hip_pitch_joint", "right_hip_pitch_joint"])
    y = JOINT_FEATURES.index("position.y")
    assert descriptor.joints[left, y] == pytest.approx(-descriptor.joints[right, y], abs=1e-6)
    mass = JOINT_FEATURES.index("subtree_mass")
    assert 0.05 < descriptor.joints[left, mass] < 0.3
    assert descriptor.joints[left, mass] == pytest.approx(descriptor.joints[right, mass], rel=1e-3)


def test_leg_length_override_keeps_a_trained_unit():
    card = _card("nao")
    restated = describe(card, leg_length=0.24801)
    assert restated.leg_length == 0.24801
    assert describe(card).leg_length == pytest.approx(card["leg_length"])
