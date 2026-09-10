"""Capture-point balance controller for the NAO.

The joint PD in ``scripts/stand_nao.py`` holds a posture. It reacts to joint
error and has no representation of whether the robot is falling, so the only
thing keeping it upright is that the nominal posture happens to be an
equilibrium. Push it hard enough and it holds the wrong posture very accurately
all the way to the floor.

This controller closes a loop on the quantity that actually decides the outcome.

Theory
------

Under the linear inverted pendulum model the horizontal centre of mass obeys

.. math:: \\ddot{x} = \\omega_0^2 (x - p), \\qquad \\omega_0 = \\sqrt{g/z_c}

where :math:`p` is the centre of pressure. This second-order system factors into
two first-order ones. Defining the **divergent component of motion**

.. math:: \\xi = x + \\dot{x}/\\omega_0

gives the pair

.. math:: \\dot{\\xi} = \\omega_0(\\xi - p), \\qquad \\dot{x} = -\\omega_0(x - \\xi)

The second is stable: the centre of mass always converges to :math:`\\xi`. The
first is unstable and is the entire problem. Balance is therefore not about
where the centre of mass *is*, it is about where :math:`\\xi` is going, which is
why a robot standing perfectly still with the wrong velocity is already falling
and a joint-space controller cannot tell.

The control law follows directly. Place the centre of pressure at

.. math:: p = \\xi_{ref} + k(\\xi - \\xi_{ref}), \\qquad k > 1

and the closed loop becomes

.. math:: \\dot{\\xi} = \\omega_0 (1 - k)(\\xi - \\xi_{ref})

which is stable with rate :math:`\\omega_0(k-1)`. The condition :math:`k>1` is
the whole design: the pressure centre has to move *past* the divergent
component, not merely toward it. That is why a controller that chases the centre
of mass fails -- it never overtakes.

From pressure centre to joint targets
-------------------------------------

The commanded :math:`p` is realised through ankle torque. Static equilibrium of
one foot, with normal load :math:`F` and the pressure centre a distance
:math:`d` ahead of the ankle axis, gives the moment the shank must transmit:

.. math:: m_y = d\\,F, \\qquad m_x = -e\\,F

Both signs were checked against the simulator's own reported joint reaction
wrench rather than derived and hoped for; see ``docs/stand_nao_dcm.md``.

The actuators are implicit position PDs, so a torque is requested by displacing
the target:

.. math:: q_{des} = q + (\\tau_{des} + K_d \\dot{q}) / K_p

Saturation, and what happens after it
-------------------------------------

Unilateral contact confines :math:`p` to the support polygon: a foot can push on
the ground but it cannot pull. Once the commanded pressure centre hits the toe
edge the ankle strategy is finished, and this is not a rare corner case -- for
the NAO the foot runs out before the ankle motors do, at 103.3 mm against
116.2 mm of torque authority.

Past that point the only remaining source of horizontal force is a change of
centroidal angular momentum: accelerate the trunk and arms one way and the
ground pushes the body the other. The controller therefore keeps the *unclipped*
command and feeds the part it could not realise into hip and shoulder pitch.
This is the hip and arm strategy, and it is the piece a fixed joint PD
structurally cannot express, because a position loop's whole purpose is to hold
the arms still.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["DcmBalanceCfg", "DcmBalanceController"]


@dataclass
class DcmBalanceCfg:
    """Tuning for :class:`DcmBalanceController`."""

    dcm_gain: float = 2.5
    r"""Feedback gain :math:`k` on the divergent component.

    Must exceed 1 or the closed loop is unstable: the pressure centre has to
    overtake :math:`\xi`, not follow it. The closed-loop rate is
    :math:`\omega_0(k-1)`, so 2.5 gives 9.1 rad/s against an open-loop
    divergence of 6.04 rad/s -- the error is pulled in about 1.5 times faster
    than it grows. Higher values reach the polygon edge sooner and saturate.
    """

    polygon_margin: float = 0.015
    """Metres of support polygon held in reserve.

    The commanded pressure centre is clipped this far inside the geometric
    edge. The foot is 164 mm long, so 15 mm costs little authority and keeps the
    contact from going critical on the edge of a convex hull that is itself an
    approximation of the real sole.
    """

    hip_gain: float = 1.2
    """Radians of hip pitch per metre of unrealised pressure-centre command.

    Only active once the ankle has saturated. The sign is such that the trunk
    rotates *into* the fall, which drives the ground reaction the other way.
    """

    arm_gain: float = 2.5
    """Radians of shoulder pitch per metre of unrealised command.

    Larger than the hip gain because the arms are light -- about 0.0086 kg m^2
    each against the trunk -- so they have to move further for the same change
    in angular momentum. They are also free to move, which the trunk is not.
    """

    hip_limit: float = 0.35
    """Maximum hip pitch deviation, in radians."""

    arm_limit: float = 1.0
    """Maximum shoulder pitch deviation, in radians."""

    ankle_limit: float = 0.40
    """Maximum ankle target deviation, in radians.

    Guards the position-target trick: a large normal load with a small stiffness
    can ask for a displacement that would otherwise leave the joint range.
    """

    min_normal_force: float = 2.0
    """Newtons below which a foot is treated as unloaded and contributes nothing."""


class DcmBalanceController:
    """Capture-point feedback on top of the nominal standing posture.

    Batched over environments, so the same controller can drive one robot in the
    baseline script or every robot in a vectorised environment for comparison
    against a learned policy.
    """

    def __init__(
        self,
        omega: float,
        nominal_joint_pos: torch.Tensor,
        joint_index: dict[str, int],
        polygon_x: tuple[float, float],
        polygon_y: tuple[float, float],
        foot_polygon_y: tuple[float, float],
        dcm_reference: tuple[float, float] = (0.0, 0.0),
        cfg: DcmBalanceCfg | None = None,
    ) -> None:
        """
        Args:
            omega: Inverted-pendulum frequency, in rad/s.
            nominal_joint_pos: Full nominal joint vector, shape (num_envs, num_joints).
            joint_index: Map from joint name to column in that vector.
            polygon_x: Forward support bounds relative to the sole midpoint, in metres.
            polygon_y: Lateral support bounds relative to the sole midpoint, in metres.
            foot_polygon_y: Lateral bounds of the **right** foot about its own
                ankle, as ``(low, high)``. The left foot is the mirror image,
                ``(-high, -low)``. Needed because a foot can only carry pressure
                inside its own sole, so a command on the midline is unreachable
                for either foot on its own.
            dcm_reference: Where the divergent component should settle, relative to
                the sole midpoint, in metres. This is **not** the centre of the
                support polygon. The nominal posture puts the centre of mass
                12.6 mm ahead of it, and that is the configuration the rest of the
                body is held at by its own position loops. Asking for the
                geometric centre instead demands a standing ankle torque that the
                posture loops fight, and the ankle saturates trying to win.
            cfg: Tuning. Defaults to :class:`DcmBalanceCfg`.
        """
        self.cfg = cfg or DcmBalanceCfg()
        self.omega = omega
        self.nominal = nominal_joint_pos
        self.index = joint_index
        self.polygon_x = polygon_x
        self.polygon_y = polygon_y
        self.foot_polygon_y = foot_polygon_y
        self.dcm_reference = dcm_reference

        if self.cfg.dcm_gain <= 1.0:
            raise ValueError(
                f"dcm_gain must exceed 1 for the closed loop to be stable, got {self.cfg.dcm_gain}."
                " At k <= 1 the pressure centre never overtakes the divergent component."
            )

        self._ankle_pitch = [joint_index["LAnklePitch"], joint_index["RAnklePitch"]]
        self._ankle_roll = [joint_index["LAnkleRoll"], joint_index["RAnkleRoll"]]
        self._hip_pitch = [joint_index["LHipPitch"], joint_index["RHipPitch"]]
        self._shoulder_pitch = [joint_index["LShoulderPitch"], joint_index["RShoulderPitch"]]

        # Filled in by compute(), for logging and for the report.
        self.last_dcm: torch.Tensor | None = None
        self.last_cop_command: torch.Tensor | None = None
        self.last_cop_clipped: torch.Tensor | None = None
        self.last_saturation: torch.Tensor | None = None

    def compute(
        self,
        com_pos_w: torch.Tensor,
        com_vel_w: torch.Tensor,
        foot_pos_w: torch.Tensor,
        foot_normal_force: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
    ) -> torch.Tensor:
        """Return full joint position targets.

        Args:
            com_pos_w: Whole-body centre of mass, shape (N, 3).
            com_vel_w: Its velocity, shape (N, 3).
            foot_pos_w: Foot body positions, shape (N, 2, 3), left then right.
            foot_normal_force: Upward load carried by each foot, shape (N, 2).
            joint_pos: Measured joint positions, shape (N, J).
            joint_vel: Measured joint velocities, shape (N, J).
            stiffness: Per-joint drive stiffness, shape (N, J).
            damping: Per-joint drive damping, shape (N, J).

        Returns:
            Joint position targets, shape (N, J).
        """
        cfg = self.cfg
        support_center = foot_pos_w.mean(dim=1)

        # -- divergent component, relative to the middle of the support polygon
        offset = com_pos_w[:, :2] - support_center[:, :2]
        velocity = com_vel_w[:, :2]
        dcm = offset + velocity / self.omega
        self.last_dcm = dcm

        # -- desired pressure centre: overtake the divergent component
        # p = xi_ref + k (xi - xi_ref), so at the reference the command is the
        # reference itself and the standing ankle torque is whatever static
        # equilibrium already needed -- not a permanent correction.
        reference = torch.tensor(self.dcm_reference, device=dcm.device, dtype=dcm.dtype)
        command = reference + cfg.dcm_gain * (dcm - reference)
        self.last_cop_command = command

        clipped = torch.stack(
            [
                command[:, 0].clamp(
                    self.polygon_x[0] + cfg.polygon_margin,
                    self.polygon_x[1] - cfg.polygon_margin,
                ),
                command[:, 1].clamp(
                    self.polygon_y[0] + cfg.polygon_margin,
                    self.polygon_y[1] - cfg.polygon_margin,
                ),
            ],
            dim=-1,
        )
        self.last_cop_clipped = clipped

        # What the ankles could not deliver has to come from angular momentum.
        saturation = command - clipped
        self.last_saturation = saturation

        targets = self.nominal.clone()

        # -- ankle strategy ------------------------------------------------
        # Per foot, the pressure centre is placed relative to that foot's own
        # ankle axis, and the load it carries sets the torque needed.
        loaded = foot_normal_force > cfg.min_normal_force
        force = torch.where(loaded, foot_normal_force, torch.zeros_like(foot_normal_force))

        right_low, right_high = self.foot_polygon_y
        for slot, (pitch_col, roll_col) in enumerate(
            zip(self._ankle_pitch, self._ankle_roll, strict=True)
        ):
            foot_offset = foot_pos_w[:, slot, :2] - support_center[:, :2]

            # A foot can only carry pressure inside its own sole. Asking both
            # ankles to put their pressure centre on the midline would be asking
            # for a point 50 mm outside each foot: the feet are 100 mm apart and
            # only 92 mm wide. The global command is therefore clamped into each
            # foot's own reachable range, and the lateral component of the
            # commanded pressure centre is realised mostly by how the load
            # divides between the two feet rather than by either ankle alone.
            lever_x = (clipped[:, 0] - foot_offset[:, 0]).clamp(
                self.polygon_x[0], self.polygon_x[1]
            )
            # Slot 0 is the left foot, on +y. Its sole is the mirror of the
            # right one, so the bounds negate and swap.
            low, high = (-right_high, -right_low) if slot == 0 else (right_low, right_high)
            lever_y = (clipped[:, 1] - foot_offset[:, 1]).clamp(low, high)

            # m_y = +d F and m_x = -e F, both checked against the simulator's
            # own reported joint reaction wrench.
            self._request_torque(targets, joint_pos, joint_vel, stiffness, damping,
                                 pitch_col, lever_x * force[:, slot])
            self._request_torque(targets, joint_pos, joint_vel, stiffness, damping,
                                 roll_col, -lever_y * force[:, slot])

        # -- hip and arm strategy -----------------------------------------
        # Only non-zero once the ankle has run out of polygon, and only for a
        # *forward* overrun.
        #
        # The sign of this strategy was verified against forward pushes and
        # nothing else, and measurement showed that mattered: with it active in
        # both directions the controller fell to a 0.30 m/s backward push that
        # the plain joint PD survives. Falling backwards is also the more
        # dangerous failure, since the heel is only 60.7 mm from the ankle
        # against 103.3 mm of toe.
        #
        # Gating to the validated direction leaves backward recovery to the
        # ankle alone, which is worse than a correct two-sided strategy would be
        # but better than an actively wrong one. Deriving and checking the
        # backward and lateral signs is left as real work, not a sign flip.
        forward_saturation = saturation[:, 0].clamp(min=0.0)
        hip = (cfg.hip_gain * forward_saturation).clamp(-cfg.hip_limit, cfg.hip_limit)
        arm = (cfg.arm_gain * forward_saturation).clamp(-cfg.arm_limit, cfg.arm_limit)
        for column in self._hip_pitch:
            targets[:, column] = self.nominal[:, column] + hip
        for column in self._shoulder_pitch:
            targets[:, column] = self.nominal[:, column] - arm

        return targets

    def _request_torque(
        self,
        targets: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        stiffness: torch.Tensor,
        damping: torch.Tensor,
        column: int,
        torque: torch.Tensor,
    ) -> None:
        """Displace one joint's target so its PD produces ``torque``.

        The drive computes ``tau = Kp (q_des - q) - Kd qdot``. Inverting that
        for ``q_des`` turns a position loop into a torque source without
        replacing the actuator model, which matters because the implicit drive
        is integrated by PhysX at the full physics rate rather than held across
        the control period.
        """
        gain = stiffness[:, column].clamp(min=1e-6)
        offset = (torque + damping[:, column] * joint_vel[:, column]) / gain
        offset = offset.clamp(-self.cfg.ankle_limit, self.cfg.ankle_limit)
        targets[:, column] = joint_pos[:, column] + offset
