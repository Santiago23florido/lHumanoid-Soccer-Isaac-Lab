"""The static torque a NAO joint holds, checked by virtual work.

:func:`~humanoid_transfer.nao.assets.nao_kinematics.static_joint_torque` sums
moments about the joint axis. Virtual work gives the same torque without any
cross product: at equilibrium ``tau = dV/dq - (dx/dq) . F``, where ``V`` is the
potential energy of the links and ``x`` the point the force acts on, both
differentiated through forward kinematics by central differences.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402

CROUCH = {
    **nk.NOMINAL_STAND_JOINT_POS,
    "LHipPitch": -0.6,
    "LKneePitch": 1.2,
    "LAnklePitch": -0.6,
    "LHipRoll": 0.05,
    "LAnkleRoll": -0.05,
}
FORCE = np.array([1.5, -2.0, 26.0])


def _potential_and_sole(joint_pos: dict[str, float]) -> tuple[float, np.ndarray]:
    model = nk.load_model()
    poses = nk.forward_kinematics(joint_pos)
    energy = 0.0
    for name, link in model.links.items():
        if link.mass == 0.0 or name not in poses:
            continue
        energy += link.mass * nk.GRAVITY * (poses[name] @ np.append(link.com, 1.0))[2]
    return energy, poses["l_sole"][:3, 3]


@pytest.mark.parametrize(
    "joint", ["LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"]
)
def test_static_torque_matches_virtual_work(joint: str) -> None:
    step = 1e-6
    ahead = {**CROUCH, joint: CROUCH[joint] + step}
    behind = {**CROUCH, joint: CROUCH[joint] - step}
    energy_ahead, sole_ahead = _potential_and_sole(ahead)
    energy_behind, sole_behind = _potential_and_sole(behind)
    expected = (energy_ahead - energy_behind) / (2 * step) - FORCE @ (sole_ahead - sole_behind) / (
        2 * step
    )
    sole = nk.forward_kinematics(CROUCH)["l_sole"][:3, 3]
    assert nk.static_joint_torque(joint, FORCE, sole, CROUCH) == pytest.approx(expected, abs=1e-6)


def test_a_knee_carrying_the_body_holds_the_weight_times_its_lever() -> None:
    """With the whole weight under the CoM, the knee's torque is that weight times the
    horizontal distance from the knee axis, less what the shank and foot weigh."""
    weight = nk.load_model().total_mass * nk.GRAVITY
    com = nk.center_of_mass(CROUCH)
    poses = nk.forward_kinematics(CROUCH)
    sole = poses["l_sole"][:3, 3]
    cop = np.array([com[0], sole[1], sole[2]])
    knee = poses["LTibia"][:3, 3]  # the knee joint's child frame sits on its axis
    torque = nk.static_joint_torque("LKneePitch", np.array([0.0, 0.0, weight]), cop, CROUCH)
    assert abs(torque) == pytest.approx(weight * abs(cop[0] - knee[0]), rel=0.1)
