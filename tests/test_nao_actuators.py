"""Checks on the NAO actuator model.

``humanoid_soccer_lab.nao.assets.nao`` imports ``isaaclab``, which cannot load
outside the Isaac Sim Kit application because it needs ``pxr``. These tests
therefore read the configuration out of the module source rather than importing
it, which keeps them runnable anywhere — the same constraint the rest of the
suite works under.

What they protect is the link between the shipped gains and the derivation in
``scripts/derive_gains.py``. The gains are not tuned values, so if the two ever
disagree the shipped numbers have lost their justification.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_soccer_lab"))
sys.path.insert(0, str(ROOT / "scripts"))

import derive_gains  # noqa: E402
from humanoid_soccer_lab.nao.assets import nao_kinematics as nk  # noqa: E402

NAO_SOURCE = (
    ROOT / "source" / "humanoid_soccer_lab" / "humanoid_soccer_lab" / "nao" / "assets" / "nao.py"
).read_text(encoding="utf-8")


def _literal(name: str) -> dict[str, float]:
    """Read a module-level dict literal out of ``nao.py`` without importing it."""
    match = re.search(rf"^{name}: dict\[str, float\] = (\{{.*?^\}})", NAO_SOURCE, re.S | re.M)
    assert match is not None, f"{name} not found in nao.py"
    return ast.literal_eval(match.group(1))


STIFFNESS = _literal("NAO_STIFFNESS")
DAMPING = _literal("NAO_DAMPING")


# --------------------------------------------------------------------------
# Coverage: every degree of freedom is accounted for exactly once
# --------------------------------------------------------------------------


def test_stiffness_and_damping_cover_the_same_joints() -> None:
    assert set(STIFFNESS) == set(DAMPING)


def test_every_commanded_joint_has_a_gain() -> None:
    missing = [j for j in nk.ACTUATED_JOINTS + nk.HELD_JOINTS if j not in STIFFNESS]
    assert missing == [], f"joints with no gain: {missing}"


def test_no_mimic_joint_is_given_a_drive() -> None:
    """A drive on a mimic joint fights the PhysX constraint that drives it."""
    offenders = [j for j in nk.MIMIC_DRIVEN_JOINTS if j in STIFFNESS]
    assert offenders == [], f"mimic joints given gains: {offenders}"


def test_the_actuator_groups_partition_all_42_dofs() -> None:
    """Legs, arms, held and mimic must tile the articulation with no overlap."""
    legs = {n for n in STIFFNESS if n[1:].startswith(("Hip", "Knee", "Ankle"))}
    arms = {n for n in STIFFNESS if n[1:].startswith(("Shoulder", "Elbow"))}
    held = {n for n in STIFFNESS if n in nk.HELD_JOINTS and n not in nk.ACTUATED_JOINTS}
    mimic = set(nk.MIMIC_DRIVEN_JOINTS)

    assert len(legs) == 11
    assert len(arms) == 8
    assert len(held) == 6
    assert legs.isdisjoint(arms) and arms.isdisjoint(held) and legs.isdisjoint(held)
    assert (legs | arms | held).isdisjoint(mimic)
    # 11 + 8 + 6 + 17 is the full 42-DOF articulation reported by the viewer.
    assert len(legs | arms | held | mimic) == 42


# --------------------------------------------------------------------------
# The shipped gains must equal the derivation
# --------------------------------------------------------------------------


def _expected_gains(name: str) -> tuple[float, float]:
    """Recompute one joint's gains the way ``scripts/derive_gains.py`` does."""
    pose = nk.NOMINAL_STAND_JOINT_POS
    if name in nk.LEG_ACTUATED_JOINTS:
        inertia, lever = nk.whole_body_inertia_about_ankle(name, pose)
        gravity = nk.load_model().total_mass * nk.GRAVITY * lever
        parallel = 1 if name == "LHipYawPitch" else derive_gains.PARALLEL_LEG_JOINTS
        frequency = derive_gains.LEG_NATURAL_FREQUENCY
    else:
        inertia = nk.composite_inertia_about_joint(name, pose)
        gravity = 0.0
        parallel = 1
        frequency = (
            derive_gains.ARM_NATURAL_FREQUENCY
            if name in nk.ARM_ACTUATED_JOINTS
            else derive_gains.HELD_NATURAL_FREQUENCY
        )
    return derive_gains.critically_damped_gains(inertia, gravity, frequency, parallel)


@pytest.mark.parametrize("joint", sorted(set(nk.ACTUATED_JOINTS)))
def test_shipped_gains_match_the_derivation(joint: str) -> None:
    """The gains are derived, so a hand edit that breaks that must fail here."""
    stiffness, damping = _expected_gains(joint)
    assert STIFFNESS[joint] == pytest.approx(stiffness, abs=0.06), (
        f"{joint} stiffness {STIFFNESS[joint]} does not match the derived {stiffness:.3f}; "
        "re-run scripts/derive_gains.py"
    )
    assert DAMPING[joint] == pytest.approx(damping, abs=0.06)


# --------------------------------------------------------------------------
# Physical admissibility
# --------------------------------------------------------------------------


def test_ankle_stiffness_exceeds_the_gravitational_negative_stiffness() -> None:
    """The hard bound: below M*g*l no amount of damping stabilises the loop.

    Both ankles act on the same body inertia in double support, so it is their
    sum that has to clear the bound.
    """
    model = nk.load_model()
    _, lever = nk.whole_body_inertia_about_ankle(joint_pos=nk.NOMINAL_STAND_JOINT_POS)
    destabilising = model.total_mass * nk.GRAVITY * lever
    total = STIFFNESS["LAnklePitch"] + STIFFNESS["RAnklePitch"]
    assert total > destabilising, f"{total} N m/rad does not clear M g l = {destabilising:.2f}"
    # And with real margin, not marginally.
    assert total > 3.0 * destabilising


def test_gains_stay_inside_the_urdf_torque_budget_for_normal_excursions() -> None:
    """A joint must not saturate before it has left the balance regime.

    Saturation error is tau_max/K_p. For the legs this should sit well outside
    the few degrees a standing robot actually moves through.
    """
    efforts = {joint.name: joint.effort for joint in nk.load_model().joints if joint.effort}
    for joint in nk.LEG_ACTUATED_JOINTS:
        saturation = efforts[joint] / STIFFNESS[joint]
        assert saturation > 0.08, f"{joint} saturates at only {saturation:.3f} rad"


def test_every_gain_is_non_negative_and_damped() -> None:
    for joint, stiffness in STIFFNESS.items():
        assert stiffness >= 0.0
        assert DAMPING[joint] >= 0.0
        if stiffness > 0.0:
            assert DAMPING[joint] > 0.0, f"{joint} has stiffness but no damping"


# --------------------------------------------------------------------------
# The two configurations must stay distinct
# --------------------------------------------------------------------------


def test_the_phase_one_config_keeps_its_passive_drives() -> None:
    """``view_nao.py`` depends on the robot still collapsing under gravity."""
    passive = re.search(r'"all_joints": ImplicitActuatorCfg\((.*?)\)', NAO_SOURCE, re.S)
    assert passive is not None
    assert "stiffness=None" in passive.group(1)
    assert "damping=None" in passive.group(1)


def test_contact_reporting_is_on_only_for_the_standing_config() -> None:
    """The ContactSensor raises without it, but Phase 1 never needed it."""
    assert "activate_contact_sensors=False" in NAO_SOURCE
    assert "activate_contact_sensors=True" in NAO_SOURCE


def test_soft_joint_limits_leave_margin_for_the_limit_penalty() -> None:
    """Isaac Lab defaults this to 1.0, which leaves the penalty nothing to bite."""
    match = re.search(r"NAO_SOFT_JOINT_POS_LIMIT_FACTOR = ([0-9.]+)", NAO_SOURCE)
    assert match is not None
    assert 0.8 <= float(match.group(1)) < 1.0
