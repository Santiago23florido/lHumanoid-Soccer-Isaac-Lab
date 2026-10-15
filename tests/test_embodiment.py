"""Checks on the dimensionless description of each body.

The constants in ``transfer.embodiment`` are restated, not computed, so these
tests re-derive every one from where it came from: the NAO's URDF and the G1's
measured body card. A drifted constant fails here instead of quietly changing
the size of the embodiment gap the study reports.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from humanoid_transfer.common.centroidal import composite_inertia
from humanoid_transfer.common.frames import quat_to_matrix
from humanoid_transfer.common.similarity import UnitSystem
from humanoid_transfer.nao.assets import nao_kinematics as nk
from humanoid_transfer.transfer.embodiment import (
    G1,
    G1_TEACHER_TASK,
    LEG_ROLES,
    NAO,
    EmbodimentSpec,
    pi_groups,
    similarity_defect,
    task_groups,
)

ROOT = Path(__file__).resolve().parents[1]
BODY_CARD = ROOT / "g1" / "body" / "body_card.json"

NAO_ROLE_JOINTS = {
    "hip_pitch": "LHipPitch",
    "hip_roll": "LHipRoll",
    "knee": "LKneePitch",
    "ankle_pitch": "LAnklePitch",
    "ankle_roll": "LAnkleRoll",
}
G1_ROLE_JOINTS = {
    "hip_pitch": "left_hip_pitch_joint",
    "hip_roll": "left_hip_roll_joint",
    "knee": "left_knee_joint",
    "ankle_pitch": "left_ankle_pitch_joint",
    "ankle_roll": "left_ankle_roll_joint",
}

NAO_REFERENCE_POSTURE = {
    "LShoulderPitch": math.pi / 2,
    "RShoulderPitch": math.pi / 2,
    "LElbowRoll": -0.0349066,
    "RElbowRoll": 0.0349066,
}
"""Legs straight, arms hanging. The NAO's zero pose points the arms forward."""


# --- the student, from its URDF ---------------------------------------------


def test_the_nao_spec_is_what_its_urdf_says() -> None:
    model = nk.load_model()
    straight = nk.forward_kinematics({})
    poses = nk.forward_kinematics(NAO_REFERENCE_POSTURE)
    names = [n for n, link in model.links.items() if link.mass > 0.0 and n in poses]

    inertia = composite_inertia(
        np.array([model.links[n].mass for n in names]),
        np.stack(
            [
                model.links[n].inertia_rot @ model.links[n].inertia @ model.links[n].inertia_rot.T
                for n in names
            ]
        ),
        np.stack([poses[n][:3, :3] for n in names]),
        np.stack([(poses[n] @ np.append(model.links[n].com, 1.0))[:3] for n in names]),
    )

    leg = np.linalg.norm(straight["LThigh"][:3, 3] - straight["l_sole"][:3, 3])
    assert NAO.leg_length == pytest.approx(leg, abs=1e-5)
    assert NAO.mass == pytest.approx(model.total_mass, abs=1e-5)
    assert NAO.com_height == pytest.approx(
        nk.com_height_above_soles(NAO_REFERENCE_POSTURE), abs=1e-5
    )
    assert NAO.composite_inertia == pytest.approx(np.diag(inertia), abs=1e-6)
    assert NAO.foot_length == pytest.approx(np.diff(nk.SUPPORT_POLYGON_X)[0], abs=1e-6)
    assert NAO.foot_width == pytest.approx(np.diff(nk.SUPPORT_POLYGON_Y_SINGLE)[0], abs=1e-6)
    assert NAO.sole_separation == pytest.approx(
        abs(poses["l_sole"][1, 3] - poses["r_sole"][1, 3]), abs=1e-6
    )

    joints = {j.name: j for j in model.joints}
    for role, joint in NAO_ROLE_JOINTS.items():
        assert NAO.joint_effort[role] == pytest.approx(joints[joint].effort)
        assert NAO.joint_velocity[role] == pytest.approx(joints[joint].velocity)


# --- the teacher, from its measured body ------------------------------------


@pytest.fixture(scope="module")
def card() -> dict:
    return json.loads(BODY_CARD.read_text(encoding="utf-8"))


def test_the_g1_spec_is_what_its_body_card_says(card: dict) -> None:
    zero = card["postures"]["zero"]
    bodies = card["bodies"]
    inertia = composite_inertia(
        np.array(bodies["mass"]),
        np.array(bodies["inertia_link"]),
        quat_to_matrix(np.array(zero["link_quat_wxyz"])),
        np.array(zero["com_pos"]),
    )
    foot = card["foot"]["left_ankle_roll_link"]

    assert G1.leg_length == pytest.approx(card["leg_length"], abs=1e-5)
    assert G1.mass == pytest.approx(card["total_mass"], abs=1e-4)
    assert G1.com_height == pytest.approx(zero["com_height_above_soles"], abs=1e-5)
    assert G1.composite_inertia == pytest.approx(np.diag(inertia), abs=1e-5)
    assert G1.foot_length == pytest.approx(foot["length"], abs=1e-5)
    assert G1.foot_width == pytest.approx(foot["width"], abs=1e-5)
    assert G1.sole_separation == pytest.approx(zero["sole_lateral_separation"], abs=1e-5)

    joints = card["joints"]
    index = {name: i for i, name in enumerate(joints["names"])}
    for role, joint in G1_ROLE_JOINTS.items():
        assert G1.joint_effort[role] == pytest.approx(joints["effort_limits"][index[joint]])
        assert G1.joint_velocity[role] == pytest.approx(joints["vel_limits"][index[joint]])


def test_the_g1_feet_are_mirror_images(card: dict) -> None:
    """One foot's numbers stand for both only if the two collision boxes agree."""
    left, right = (card["foot"][name] for name in card["foot_bodies"])
    assert left["length"] == pytest.approx(right["length"])
    assert left["width"] == pytest.approx(right["width"])
    assert left["sole_point_link"] == pytest.approx(right["sole_point_link"])


def test_the_g1_task_constants_are_the_teachers(card: dict) -> None:
    text = (ROOT / "g1" / "policy" / "teacher_env.yaml").read_text(encoding="utf-8")
    assert "dt: 0.005" in text
    assert "decimation: 4" in text
    assert G1_TEACHER_TASK["control_dt"] == pytest.approx(0.005 * 4)
    assert "static_friction_range: !!python/tuple\n      - 0.8\n      - 0.8" in text


# --- the defect -------------------------------------------------------------


def _rescaled(spec: EmbodimentSpec, lam: float, mu: float) -> EmbodimentSpec:
    """A geometrically and dynamically similar body: length x lam, mass x mu, gravity fixed."""
    return replace(
        spec,
        leg_length=spec.leg_length * lam,
        mass=spec.mass * mu,
        com_height=spec.com_height * lam,
        composite_inertia=tuple(i * mu * lam**2 for i in spec.composite_inertia),
        foot_length=spec.foot_length * lam,
        foot_width=spec.foot_width * lam,
        sole_separation=spec.sole_separation * lam,
        joint_effort={k: v * mu * lam for k, v in spec.joint_effort.items()},
        joint_velocity={k: v / math.sqrt(lam) for k, v in spec.joint_velocity.items()},
    )


def test_a_scaled_copy_of_a_body_has_no_similarity_defect() -> None:
    defect = similarity_defect(G1, _rescaled(G1, 0.38, 0.165))
    assert defect == pytest.approx(dict.fromkeys(defect, 0.0), abs=1e-12)


def test_the_defect_is_antisymmetric() -> None:
    forward = similarity_defect(G1, NAO)
    backward = similarity_defect(NAO, G1)
    assert forward == pytest.approx({k: -v for k, v in backward.items()})


def test_every_leg_role_has_a_torque_and_a_speed_group() -> None:
    groups = pi_groups(NAO)
    for role in LEG_ROLES:
        assert f"torque_{role}" in groups
        assert f"speed_{role}" in groups


def test_the_student_is_weaker_at_hip_and_knee_but_stronger_at_the_ankle() -> None:
    """A finding, pinned so a changed constant that reverses it is noticed.

    In body units the NAO has about a sixth of the G1's hip and knee torque but
    more than twice its ankle torque: the simulated G1's ankle group is capped
    at 20 N m against 300 N m elsewhere. "The weak robot" is weak in some
    directions and not others, which is what a single strength ratio would hide.
    """
    defect = similarity_defect(G1, NAO)
    assert defect["torque_hip_pitch"] < math.log(0.2)
    assert defect["torque_knee"] < math.log(0.2)
    assert defect["torque_ankle_pitch"] > math.log(2.0)
    assert defect["foot_length"] > math.log(2.0)


def test_task_groups_put_the_control_period_in_body_time() -> None:
    units = UnitSystem(length=0.65457, mass=32.2389)
    groups = task_groups(units, **G1_TEACHER_TASK)
    assert groups["control_period"] == pytest.approx(0.02 / math.sqrt(0.65457 / 9.81))
    assert groups["static_friction"] == 0.8
