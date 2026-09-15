"""Checks on the capture-point balance controller.

Pure tensor arithmetic, so these run without Isaac Sim. They protect the parts
of the controller that are easy to get backwards and expensive to debug in
simulation: the sign of the feedback, when the angular-momentum strategy is
allowed to engage, and the fact that a foot is never asked to carry pressure
outside its own sole.

Two of these tests exist because the mistake they catch was actually made. See
``docs/stand_nao_dcm.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_soccer_lab"))

from humanoid_soccer_lab.assets import nao_kinematics as nk  # noqa: E402
from humanoid_soccer_lab.controllers import DcmBalanceCfg, DcmBalanceController  # noqa: E402

JOINTS = (
    "LAnklePitch",
    "RAnklePitch",
    "LAnkleRoll",
    "RAnkleRoll",
    "LHipPitch",
    "RHipPitch",
    "LShoulderPitch",
    "RShoulderPitch",
)
INDEX = {name: i for i, name in enumerate(JOINTS)}

OMEGA = 6.04
STANCE = 0.05
FOOT_HEIGHT = 0.047
NOMINAL_COM_X = 0.0126
"""Forward offset of the nominal centre of mass from the sole midpoint, in metres."""


def _controller(cfg: DcmBalanceCfg | None = None, reference: float = 0.0):
    return DcmBalanceController(
        omega=OMEGA,
        nominal_joint_pos=torch.zeros(1, len(JOINTS)),
        joint_index=INDEX,
        polygon_x=nk.SUPPORT_POLYGON_X,
        polygon_y=nk.support_polygon_double_stance()[1],
        foot_polygon_y=nk.SUPPORT_POLYGON_Y_SINGLE,
        dcm_reference=(reference, 0.0),
        cfg=cfg,
    )


def _run(controller, com_xy=(0.0, 0.0), vel_xy=(0.0, 0.0), load=26.0, stiffness=34.0):
    """Drive the controller once and return its joint targets."""
    count = len(JOINTS)
    return controller.compute(
        com_pos_w=torch.tensor([[com_xy[0], com_xy[1], 0.27]]),
        com_vel_w=torch.tensor([[vel_xy[0], vel_xy[1], 0.0]]),
        foot_pos_w=torch.tensor(
            [[[0.0, STANCE, FOOT_HEIGHT], [0.0, -STANCE, FOOT_HEIGHT]]]
        ),
        foot_normal_force=torch.full((1, 2), load),
        joint_pos=torch.zeros(1, count),
        joint_vel=torch.zeros(1, count),
        stiffness=torch.full((1, count), stiffness),
        damping=torch.full((1, count), 4.7),
    )


# --------------------------------------------------------------------------
# The feedback law
# --------------------------------------------------------------------------


def test_a_gain_of_one_or_less_is_rejected() -> None:
    """At k <= 1 the pressure centre never overtakes, so the loop diverges."""
    for gain in (0.5, 1.0):
        with pytest.raises(ValueError, match="dcm_gain must exceed 1"):
            _controller(DcmBalanceCfg(dcm_gain=gain))


def test_the_divergent_component_is_position_plus_scaled_velocity() -> None:
    controller = _controller()
    _run(controller, com_xy=(0.02, 0.0), vel_xy=(0.3, 0.0))
    assert float(controller.last_dcm[0, 0]) == pytest.approx(0.02 + 0.3 / OMEGA)


def test_the_pressure_centre_overtakes_the_divergent_component() -> None:
    """The whole design: p must pass xi, because xidot only turns negative then."""
    controller = _controller()
    _run(controller, vel_xy=(0.2, 0.0))
    dcm = float(controller.last_dcm[0, 0])
    command = float(controller.last_cop_command[0, 0])
    assert command > dcm > 0.0
    assert command == pytest.approx(controller.cfg.dcm_gain * dcm)


def test_at_the_reference_the_controller_asks_for_nothing_extra() -> None:
    """A reference that is not the posture's equilibrium demands standing torque.

    The first version used the polygon centre, which sits 12.6 mm behind where
    the nominal posture actually balances. That asked for a permanent ankle
    torque the posture loops fought, and the robot fell backwards in 0.76 s with
    no push at all.
    """
    controller = _controller(reference=NOMINAL_COM_X)
    _run(controller, com_xy=(NOMINAL_COM_X, 0.0))
    assert float(controller.last_cop_command[0, 0]) == pytest.approx(NOMINAL_COM_X)
    assert float(controller.last_saturation[0, 0]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Saturation and the angular-momentum strategy
# --------------------------------------------------------------------------


def test_the_command_is_clipped_into_the_support_polygon() -> None:
    """Unilateral contact: a foot can push on the ground but never pull."""
    controller = _controller()
    _run(controller, vel_xy=(1.5, 0.0))
    limit = nk.SUPPORT_POLYGON_X[1] - controller.cfg.polygon_margin
    assert float(controller.last_cop_clipped[0, 0]) == pytest.approx(limit)
    assert float(controller.last_cop_command[0, 0]) > limit


def test_hip_and_arms_stay_still_until_the_ankle_saturates() -> None:
    """Below saturation the ankle can do the job, so nothing else should move."""
    controller = _controller()
    targets = _run(controller, vel_xy=(0.2, 0.0))
    assert float(controller.last_saturation[0, 0]) == pytest.approx(0.0, abs=1e-9)
    for name in ("LHipPitch", "RHipPitch", "LShoulderPitch", "RShoulderPitch"):
        assert float(targets[0, INDEX[name]]) == pytest.approx(0.0, abs=1e-9)


def test_hip_and_arms_engage_once_the_ankle_has_run_out_of_polygon() -> None:
    """Past the polygon the only remaining lever is centroidal angular momentum."""
    controller = _controller()
    targets = _run(controller, vel_xy=(0.6, 0.0))
    assert float(controller.last_saturation[0, 0]) > 0.0
    assert float(targets[0, INDEX["LHipPitch"]]) > 0.05
    # The shoulders swing the other way from the hips, which is what makes the
    # pair a momentum exchange rather than a single lean.
    assert float(targets[0, INDEX["LShoulderPitch"]]) < -0.05


def test_the_momentum_strategy_is_bounded() -> None:
    controller = _controller()
    targets = _run(controller, vel_xy=(5.0, 0.0))
    assert abs(float(targets[0, INDEX["LHipPitch"]])) <= controller.cfg.hip_limit + 1e-9
    assert abs(float(targets[0, INDEX["LShoulderPitch"]])) <= controller.cfg.arm_limit + 1e-9


# --------------------------------------------------------------------------
# A foot can only carry pressure inside its own sole
# --------------------------------------------------------------------------


def test_neither_foot_is_asked_for_pressure_outside_itself() -> None:
    """The feet are 100 mm apart and 92 mm wide, so the midline is outside both.

    The first version asked each ankle to place its own pressure centre on the
    midline, which is physically impossible for either foot alone. Lateral
    authority in double support comes from how the load divides between them.
    """
    controller = _controller()
    stiffness, load = 34.0, 26.0
    targets = _run(controller, vel_xy=(0.0, 0.3), load=load, stiffness=stiffness)

    right_low, right_high = nk.SUPPORT_POLYGON_Y_SINGLE
    bounds = {
        "LAnkleRoll": (-right_high, -right_low),
        "RAnkleRoll": (right_low, right_high),
    }
    for name, (low, high) in bounds.items():
        # Invert the position-target trick to recover the commanded torque, then
        # the lever it corresponds to: tau = -e F, so e = -tau / F.
        torque = stiffness * float(targets[0, INDEX[name]])
        lever = -torque / load
        assert low - 1e-6 <= lever <= high + 1e-6, (
            f"{name} was asked for pressure at {lever * 1e3:.1f} mm, outside its own"
            f" sole [{low * 1e3:.1f}, {high * 1e3:.1f}] mm"
        )


def test_an_unloaded_foot_contributes_no_torque() -> None:
    """Zero normal force means zero achievable ankle moment."""
    controller = _controller()
    targets = _run(controller, vel_xy=(0.3, 0.0), load=0.0)
    for name in ("LAnklePitch", "RAnklePitch", "LAnkleRoll", "RAnkleRoll"):
        assert float(targets[0, INDEX[name]]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Turning a position loop into a torque source
# --------------------------------------------------------------------------


def test_the_target_offset_reproduces_the_requested_torque() -> None:
    """q_des = q + (tau + Kd qdot)/Kp must invert tau = Kp (q_des - q) - Kd qdot."""
    controller = _controller()
    stiffness, load = 34.0, 26.0
    targets = _run(controller, vel_xy=(0.2, 0.0), load=load, stiffness=stiffness)

    lever = float(controller.last_cop_clipped[0, 0])
    expected = lever * load
    # Joint velocity is zero in this fixture, so the damping term drops out.
    produced = stiffness * float(targets[0, INDEX["LAnklePitch"]])
    assert produced == pytest.approx(expected, rel=1e-6)


def test_a_forward_lean_plantarflexes_both_ankles() -> None:
    """Falling forward must drive the pressure centre toward the toes."""
    controller = _controller()
    targets = _run(controller, vel_xy=(0.2, 0.0))
    assert float(targets[0, INDEX["LAnklePitch"]]) > 0.0
    assert float(targets[0, INDEX["RAnklePitch"]]) > 0.0


def test_the_response_is_invariant_to_the_robot_s_heading() -> None:
    """A yawed robot leaning the same way needs the same ankle command.

    The ankle pitch and roll axes live in the body, not the world. Computing the
    pressure-centre command in world frame sends a pitch correction to the roll
    joint as soon as the robot is not facing +x. With yaw randomised over the
    full circle at reset that is most episodes, and it took the controller from
    45% fall-free to 5% once the randomisation actually reached the simulation.
    """
    import math

    controller = _controller()
    count = len(JOINTS)
    facing_x = _run(controller, vel_xy=(0.25, 0.0))

    # The same robot, yawed 90 degrees, leaning the same way relative to itself:
    # its forward direction is now +y, so the world-frame velocity rotates too.
    half = math.pi / 4.0  # half of 90 degrees, as a quaternion needs
    quat = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
    yawed = controller.compute(
        com_pos_w=torch.tensor([[0.0, 0.0, 0.27]]),
        com_vel_w=torch.tensor([[0.0, 0.25, 0.0]]),
        foot_pos_w=torch.tensor([[[-STANCE, 0.0, FOOT_HEIGHT], [STANCE, 0.0, FOOT_HEIGHT]]]),
        foot_normal_force=torch.full((1, 2), 26.0),
        joint_pos=torch.zeros(1, count),
        joint_vel=torch.zeros(1, count),
        stiffness=torch.full((1, count), 34.0),
        damping=torch.full((1, count), 4.7),
        base_quat_w=quat,
    )

    assert float(yawed[0, INDEX["LAnklePitch"]]) == pytest.approx(
        float(facing_x[0, INDEX["LAnklePitch"]]), abs=1e-5
    ), "the ankle pitch command changed with heading alone"


def test_the_response_is_symmetric_between_left_and_right() -> None:
    """A purely sagittal disturbance must not favour one leg."""
    controller = _controller()
    targets = _run(controller, vel_xy=(0.25, 0.0))
    assert float(targets[0, INDEX["LAnklePitch"]]) == pytest.approx(
        float(targets[0, INDEX["RAnklePitch"]]), abs=1e-9
    )
    assert float(targets[0, INDEX["LAnkleRoll"]]) == pytest.approx(
        -float(targets[0, INDEX["RAnkleRoll"]]), abs=1e-9
    )
