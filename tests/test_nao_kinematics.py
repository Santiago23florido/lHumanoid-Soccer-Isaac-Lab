"""Checks on the kinematic quantities the balance controller is sized from.

These run without Isaac Sim. They exist because every gain, reward scale and
termination threshold in the standing task is derived from the numbers in
:mod:`humanoid_transfer.nao.assets.nao_kinematics`; if one of them silently
drifted from the URDF the whole controller would be detuned with no error.

Tests that need the CC BY-NC-ND geometry skip themselves when it has not been
fetched, following the convention in ``tests/test_nao_assets.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.assets import nao_paths  # noqa: E402

requires_meshes = pytest.mark.skipif(
    not nao_paths.meshes_are_available(),
    reason="NAO meshes not fetched; run scripts/fetch_nao_meshes.py",
)


# --------------------------------------------------------------------------
# Model parsing
# --------------------------------------------------------------------------


def test_model_matches_the_urdf_the_project_targets() -> None:
    model = nk.load_model()
    assert model.name == "NaoH25V50"
    assert len(model.links) == 79
    assert len(model.joints) == 78
    assert model.total_mass == pytest.approx(5.3054, abs=1e-4)


def test_forward_kinematics_reaches_every_link() -> None:
    """A disconnected link would silently drop out of the centre-of-mass sum."""
    model = nk.load_model()
    poses = nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)
    missing = sorted(set(model.links) - set(poses))
    assert missing == [], f"links unreachable from base_link: {missing}"


# --------------------------------------------------------------------------
# Nominal posture
# --------------------------------------------------------------------------


def test_nominal_posture_lies_inside_every_urdf_joint_limit() -> None:
    limits = nao_paths.joint_limits_from_urdf()
    for name, value in nk.NOMINAL_STAND_JOINT_POS.items():
        lower, upper = limits[name]
        assert lower <= value <= upper, f"{name}={value} outside [{lower}, {upper}]"


def test_nominal_posture_only_moves_joints_the_policy_or_the_hold_owns() -> None:
    commanded = set(nk.ACTUATED_JOINTS) | set(nk.HELD_JOINTS)
    assert set(nk.NOMINAL_STAND_JOINT_POS) <= commanded


def test_sagittal_pitch_angles_sum_to_zero_so_the_torso_stays_vertical() -> None:
    """hip + ankle = -knee is the flat-foot, vertical-torso condition."""
    pose = nk.NOMINAL_STAND_JOINT_POS
    for side in ("L", "R"):
        total = pose[f"{side}HipPitch"] + pose[f"{side}KneePitch"] + pose[f"{side}AnklePitch"]
        assert total == pytest.approx(0.0, abs=1e-9)


def test_the_crouch_keeps_both_feet_flat_and_level_under_the_hips() -> None:
    """Independent confirmation of the sum-of-angles condition, via kinematics."""
    poses = nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)
    left, right = poses["l_sole"], poses["r_sole"]

    # Both soles at the same height, so the stance is symmetric.
    assert left[2, 3] == pytest.approx(right[2, 3], abs=1e-9)
    # Soles stay under the hips rather than swinging forward.
    assert abs(left[0, 3]) < 0.01
    # Sole frames keep their z axis vertical: the foot has not pitched over.
    assert left[2, 2] == pytest.approx(1.0, abs=1e-6)
    assert right[2, 2] == pytest.approx(1.0, abs=1e-6)


def test_the_crouch_costs_little_standing_height() -> None:
    """The knee bend buys Jacobian conditioning; it should not cost much height."""
    upright = nk.sole_height_below_base({})
    crouched = nk.sole_height_below_base(nk.NOMINAL_STAND_JOINT_POS)
    assert 0.0 < upright - crouched < 0.02


# --------------------------------------------------------------------------
# Inverted pendulum quantities
# --------------------------------------------------------------------------


def test_com_height_is_consistent_with_a_058_m_robot() -> None:
    height = nk.com_height_above_soles(nk.NOMINAL_STAND_JOINT_POS)
    assert 0.20 < height < 0.35, f"z_c={height} is not plausible for a NAO"


def test_lipm_frequency_and_doubling_time() -> None:
    """The control rate must beat this; it is why the policy runs at 100 Hz."""
    height = nk.com_height_above_soles(nk.NOMINAL_STAND_JOINT_POS)
    omega = nk.lipm_omega(height)
    assert omega == pytest.approx(math.sqrt(nk.GRAVITY / height))
    doubling_ms = math.log(2.0) / omega * 1e3
    # Roughly an eighth of a second: a 100 Hz policy gets ~11 actions per doubling.
    assert 90.0 < doubling_ms < 150.0


def test_the_standing_ankle_carries_the_whole_body_not_its_distal_subtree() -> None:
    """The load-bearing correction: with the foot planted the chain inverts.

    Sizing ankle gains from the open-chain subtree inertia would under-shoot by
    more than two orders of magnitude.
    """
    pose = nk.NOMINAL_STAND_JOINT_POS
    distal = nk.composite_inertia_about_joint("LAnklePitch", pose)
    whole, lever = nk.whole_body_inertia_about_ankle(joint_pos=pose)
    assert whole / distal > 100.0
    assert 0.15 < lever < 0.35


def test_gravity_sets_a_hard_lower_bound_on_ankle_stiffness() -> None:
    """M*g*l is a negative stiffness: below it the closed loop cannot be stable."""
    model = nk.load_model()
    _, lever = nk.whole_body_inertia_about_ankle(joint_pos=nk.NOMINAL_STAND_JOINT_POS)
    destabilising = model.total_mass * nk.GRAVITY * lever
    assert 8.0 < destabilising < 16.0


# --------------------------------------------------------------------------
# Support polygon and capturability
# --------------------------------------------------------------------------


def test_support_polygon_is_a_nao_sized_foot() -> None:
    x_min, x_max = nk.SUPPORT_POLYGON_X
    y_min, y_max = nk.SUPPORT_POLYGON_Y_SINGLE
    assert 0.14 < x_max - x_min < 0.19
    assert 0.06 < y_max - y_min < 0.12


def test_double_stance_polygon_bridges_the_gap_between_the_feet() -> None:
    (_, _), (y_min, y_max) = nk.support_polygon_double_stance()
    # The hull spans from one foot's outer edge to the other's, not just the soles.
    assert y_max > nk.STANCE_HALF_WIDTH
    assert y_min == pytest.approx(-y_max)


def test_the_nominal_com_sits_inside_the_support_polygon_with_margin() -> None:
    """If it did not, the robot would be falling before the policy acted."""
    com = nk.center_of_mass(nk.NOMINAL_STAND_JOINT_POS)
    (x_min, x_max), (y_min, y_max) = nk.support_polygon_double_stance()
    assert x_min + 0.02 < com[0] < x_max - 0.02
    assert y_min + 0.02 < com[1] < y_max - 0.02


def test_capturable_velocity_bounds_the_perturbation_curriculum() -> None:
    """Pushes past this cannot be recovered without stepping."""
    bounds = nk.capturable_com_velocity()
    for direction in ("forward", "backward", "lateral"):
        assert 0.2 < bounds[direction] < 1.0
    # Backward margin is larger than forward, because the CoM sits ahead of
    # the ankle axis and the heel is nearer than the toe.
    assert bounds["backward"] < bounds["forward"]
    assert bounds["forward_impulse"] == pytest.approx(
        nk.load_model().total_mass * bounds["forward"], rel=1e-9
    )


def test_capturability_uses_com_and_polygon_in_the_same_frame() -> None:
    pose = nk.NOMINAL_STAND_JOINT_POS
    frames = nk.forward_kinematics(pose)
    com_x = nk.center_of_mass(pose)[0]
    foot_mid_x = (frames["l_sole"][0, 3] + frames["r_sole"][0, 3]) / 2
    omega = nk.lipm_omega(nk.com_height_above_soles(pose))
    bounds = nk.capturable_com_velocity(pose)
    assert bounds["forward"] == pytest.approx(
        omega * (foot_mid_x + nk.SUPPORT_POLYGON_X[1] - com_x)
    )
    assert bounds["backward"] == pytest.approx(
        omega * (com_x - foot_mid_x - nk.SUPPORT_POLYGON_X[0])
    )


# --------------------------------------------------------------------------
# The constants must match the geometry they claim to describe
# --------------------------------------------------------------------------


@requires_meshes
def test_support_polygon_constants_match_the_collision_mesh() -> None:
    """Re-derives the hard-coded polygon from the STL, so it cannot go stale."""
    low, high = nk.foot_collision_extent()
    assert low[0] == pytest.approx(nk.SUPPORT_POLYGON_X[0], abs=5e-4)
    assert high[0] == pytest.approx(nk.SUPPORT_POLYGON_X[1], abs=5e-4)
    assert low[1] == pytest.approx(nk.SUPPORT_POLYGON_Y_SINGLE[0], abs=5e-4)
    assert high[1] == pytest.approx(nk.SUPPORT_POLYGON_Y_SINGLE[1], abs=5e-4)


@requires_meshes
def test_the_sole_frame_sits_on_the_bottom_of_the_foot_mesh() -> None:
    """Confirms the sole frame really is the ground contact plane."""
    low, _ = nk.foot_collision_extent()
    assert low[2] == pytest.approx(0.0, abs=2e-3)


# --------------------------------------------------------------------------
# Joint groups
# --------------------------------------------------------------------------


def test_the_policy_commands_nineteen_joints() -> None:
    assert len(nk.ACTUATED_JOINTS) == 19
    assert len(nk.LEG_ACTUATED_JOINTS) == 11
    assert len(nk.ARM_ACTUATED_JOINTS) == 8
    assert len(set(nk.ACTUATED_JOINTS)) == 19


def test_no_mimic_joint_is_ever_commanded() -> None:
    """RHipYawPitch shares one motor with LHipYawPitch on the real robot."""
    commanded = set(nk.ACTUATED_JOINTS) | set(nk.HELD_JOINTS)
    assert commanded.isdisjoint(nk.MIMIC_DRIVEN_JOINTS)
    assert "LHipYawPitch" in nk.ACTUATED_JOINTS
    assert "RHipYawPitch" not in nk.ACTUATED_JOINTS


def test_mimic_joint_list_matches_the_urdf() -> None:
    model = nk.load_model()
    from_urdf = {joint.name for joint in model.joints if joint.mimic is not None}
    assert from_urdf == set(nk.MIMIC_DRIVEN_JOINTS)


def test_every_named_joint_exists_and_the_groups_partition_the_dofs() -> None:
    model = nk.load_model()
    movable = {joint.name for joint in model.joints if joint.type != "fixed"}
    commanded = set(nk.ACTUATED_JOINTS) | set(nk.HELD_JOINTS)
    assert commanded <= movable
    assert set(nk.ACTUATED_JOINTS).isdisjoint(nk.HELD_JOINTS)
    # Every movable joint is either commanded by us or driven by a mimic tag.
    assert movable == commanded | set(nk.MIMIC_DRIVEN_JOINTS)


def test_foot_bodies_are_the_links_that_survive_the_fixed_frame_merge() -> None:
    """The soles are merged away, so contact sensing must target the ankles."""
    model = nk.load_model()
    for body in nk.FOOT_BODIES:
        assert model.links[body].mass > 0.0
    for sole in nk.SOLE_FRAMES:
        assert model.links[sole].mass == 0.0
