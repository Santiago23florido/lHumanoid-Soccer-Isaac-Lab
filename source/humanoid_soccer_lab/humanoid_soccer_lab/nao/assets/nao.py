"""Isaac Lab articulation configurations for the SoftBank / Aldebaran NAO H25 V5.0.

Two configurations live here, and the difference between them is deliberate:

* :data:`NAO_CFG` is the Phase 1 asset. Its joint drives are zero, so the robot
  collapses under gravity. That fall is what validates the articulation, and
  ``scripts/view_nao.py`` still depends on it behaving exactly that way.
* :data:`NAO_STAND_CFG` adds the actuator model, the nominal standing posture
  and contact reporting. It is the configuration any balance or locomotion work
  should build on.

Nothing here knows about tasks, rewards, observations or actions. Environments
import one of these configurations and adjust ``prim_path`` and ``init_state``
rather than redefining the robot.

The USD both point at is a build artifact generated from the vendored upstream
URDF; see :mod:`humanoid_soccer_lab.nao.assets.nao_usd` and
``third_party/nao/README.md``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .nao_kinematics import (
    ACTUATED_JOINTS,
    HELD_JOINTS,
    MIMIC_DRIVEN_JOINTS,
    NOMINAL_STAND_JOINT_POS,
    com_height_above_soles,
    lipm_omega,
    sole_height_below_base,
)
from .nao_paths import NAO_USD_PATH, default_joint_positions
from .nao_usd import ensure_nao_usd

__all__ = [
    "NAO_CFG",
    "NAO_STAND_CFG",
    "get_nao_cfg",
    "get_nao_stand_cfg",
    "NAO_SPAWN_HEIGHT",
    "NAO_SOLE_OFFSET_Z",
    "NAO_EXPECTED_JOINTS",
    "NAO_DEFAULT_JOINT_POS",
    "NAO_STAND_JOINT_POS",
    "NAO_STAND_BASE_HEIGHT",
    "NAO_STAND_SPAWN_HEIGHT",
    "NAO_STAND_COM_HEIGHT",
    "NAO_STAND_LIPM_OMEGA",
    "NAO_STIFFNESS",
    "NAO_DAMPING",
    "NAO_SOFT_JOINT_POS_LIMIT_FACTOR",
]

NAO_DEFAULT_JOINT_POS = default_joint_positions()
"""Joints whose upstream zero angle lies outside their own URDF limits.

Read from the URDF at import time rather than hard-coded. On the H25 V5.0 this
is only ``LElbowRoll`` and ``RElbowRoll``, whose ranges exclude zero because a
NAO elbow cannot fully straighten. Every other joint keeps the upstream zero.
"""

NAO_SOLE_OFFSET_Z = 0.33301
"""Height of ``base_link`` above the foot soles at the zero joint pose, in metres.

Derived from the URDF kinematic chain ``base_link -> ... -> l_sole`` with every
joint at zero; it is not a tuned value.
"""

NAO_SPAWN_HEIGHT = 0.36
"""Default spawn height of the articulation root, in metres.

``NAO_SOLE_OFFSET_Z`` plus roughly 2.7 cm of clearance, so the robot starts
just above the ground plane instead of interpenetrating it. The robot is free
floating and is expected to settle or fall: no controller exists in this phase.
"""

NAO_EXPECTED_JOINTS: tuple[str, ...] = (
    # head
    "HeadYaw",
    "HeadPitch",
    # left leg
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    # right leg
    "RHipYawPitch",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
    # left arm
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
    # right arm
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "RHand",
)
"""Joints the H25 V5.0 description is expected to expose.

Sanity-check list for the smoke test, taken from the URDF itself. ``RHipYawPitch``
and the finger/thumb joints are ``mimic`` joints upstream, so whether they appear
as independent degrees of freedom depends on how the importer resolves the mimic
tags. The smoke test reports what the articulation actually contains and never
assumes a degree-of-freedom count.
"""


NAO_CFG = ArticulationCfg(
    prim_path="/World/Robots/Nao",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(NAO_USD_PATH),
        activate_contact_sensors=False,
        # Solver and safety settings only. Every mass, inertia, joint limit and
        # joint axis comes from the upstream URDF and is left untouched.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, NAO_SPAWN_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        # Joints default to the upstream URDF zero pose. Only the joints whose
        # limits exclude zero are overridden, with the nearest valid angle from
        # the URDF itself. No posture is invented here.
        joint_pos=NAO_DEFAULT_JOINT_POS,
        joint_vel={},
    ),
    actuators={
        # stiffness/damping left as None so the values parsed from the URDF and
        # written into the USD are used verbatim. The upstream URDF declares no
        # <dynamics>, so the drives are effectively passive and the robot falls
        # under gravity. That is intended for this phase: it validates the
        # articulation rather than any controller.
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
    },
)
"""Reusable NAO articulation configuration, with passive joints.

The ``usd_path`` refers to a generated artifact. Use :func:`get_nao_cfg` to have
it built on demand, or run the conversion once beforehand.
"""


def get_nao_cfg(
    prim_path: str = "/World/Robots/Nao",
    spawn_height: float | None = None,
    force_conversion: bool = False,
) -> ArticulationCfg:
    """Return a :data:`NAO_CFG` copy with the USD guaranteed to exist.

    Generates the derived URDF and converts it to USD on first use. Must be
    called after the Isaac Sim application has been launched.

    Args:
        prim_path: Prim path for the articulation. Regex forms such as
            ``/World/Envs/Env_.*/Robot`` are supported by Isaac Lab.
        spawn_height: Root height in metres. Defaults to :data:`NAO_SPAWN_HEIGHT`.
        force_conversion: Rebuild the derived URDF and USD even if they exist.

    Returns:
        A configured :class:`ArticulationCfg` for the NAO.
    """
    usd_path = ensure_nao_usd(force=force_conversion)

    cfg = NAO_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn.usd_path = str(usd_path)
    height = NAO_SPAWN_HEIGHT if spawn_height is None else spawn_height
    cfg.init_state.pos = (0.0, 0.0, height)
    return cfg


# ---------------------------------------------------------------------------
# Standing configuration: actuators, posture and contact reporting
# ---------------------------------------------------------------------------

NAO_STAND_JOINT_POS: dict[str, float] = {**NAO_DEFAULT_JOINT_POS, **NOMINAL_STAND_JOINT_POS}
"""Nominal standing posture, with the unreachable elbow zeros still corrected.

``NOMINAL_STAND_JOINT_POS`` already sets both elbow rolls, so the merge only
matters if that posture ever stops naming them.
"""

NAO_STAND_BASE_HEIGHT = sole_height_below_base(NOMINAL_STAND_JOINT_POS)
"""Height of ``base_link`` above the soles in the nominal posture, in metres.

Computed from the URDF, not measured in simulation: 0.3207 m. The shallow crouch
costs about 12 mm against the 0.3330 m of the fully extended zero pose.
"""

NAO_STAND_SPAWN_HEIGHT = NAO_STAND_BASE_HEIGHT + 0.005
"""Root spawn height for the standing posture, in metres.

Five millimetres of clearance: enough that the soles do not start interpenetrating
the ground plane, small enough that the robot does not acquire a drop velocity
before the controller sees it.
"""

NAO_STAND_COM_HEIGHT = com_height_above_soles(NOMINAL_STAND_JOINT_POS)
"""Centre-of-mass height above the soles in the nominal posture, in metres.

0.2689 m. This is the length scale of the linear inverted pendulum and therefore
the single number the whole balance problem is timed by.
"""

NAO_STAND_LIPM_OMEGA = lipm_omega(NAO_STAND_COM_HEIGHT)
r"""Inverted-pendulum frequency :math:`\omega_0=\sqrt{g/z_c}`, in rad/s.

6.04 rad/s. The divergent mode grows as :math:`e^{\omega_0 t}`, so an
uncorrected balance error doubles every :math:`\ln 2/\omega_0` = 115 ms. Any
controller has to act well inside that.
"""

NAO_SOFT_JOINT_POS_LIMIT_FACTOR = 0.9
"""Fraction of each joint's range exposed as a soft limit.

Isaac Lab defaults this to 1.0, which makes the soft limits identical to the
hard ones and leaves a joint-limit penalty nothing to bite on before the
simulator clamps. Ten per cent of the range is enough margin for the penalty to
steer the policy away from the stops without shrinking the usable workspace.
"""


# The gains below are derived, not tuned. For each joint the closed loop is
# placed at a chosen natural frequency against the inertia that joint actually
# carries, with critical damping:
#
#     K_p = I * w_n^2 + M*g*l        K_d = 2 * zeta * sqrt(I * I * w_n^2)
#
# Two things make this concrete rather than generic.
#
# First, the inertia that matters for a leg joint is *not* the inertia of its
# distal subtree. With the foot planted the kinematic chain inverts and the
# joint carries the whole body: the ankle sees 0.390 kg m^2 rather than the
# 0.00072 kg m^2 of the foot alone, a factor of 543. Gains sized from the
# open-chain value would be two orders of magnitude too soft. Leg gains are
# therefore computed against the whole-body inertia about each joint axis, and
# halved where the left and right joints act on that inertia in parallel.
#
# Second, gravity contributes a *negative* stiffness M*g*l about the ankle. At
# the nominal posture that is 11.95 N m/rad, and a joint stiffness below it
# cannot be stabilised by any amount of damping. Adding it back into K_p is what
# makes the ankle gain a lower bound rather than a preference.
#
# w_n = 12 rad/s for the legs, roughly twice the open-loop unstable pole of
# 5.53 rad/s and comfortably inside the 100 Hz control loop. Arms and head are
# open chains carrying only themselves, so they take their own subtree inertia
# at 20 and 25 rad/s respectively. ``scripts/derive_gains.py`` recomputes the
# whole table from the URDF.

NAO_STIFFNESS: dict[str, float] = {
    # legs, against the whole-body inertia about each axis
    "LAnklePitch": 34.0,
    "RAnklePitch": 34.0,
    "LAnkleRoll": 36.3,
    "RAnkleRoll": 36.3,
    "LKneePitch": 18.9,
    "RKneePitch": 18.9,
    "LHipPitch": 11.0,
    "RHipPitch": 11.0,
    "LHipRoll": 13.2,
    "RHipRoll": 13.2,
    "LHipYawPitch": 14.4,
    # arms, open chains carrying only themselves
    "LShoulderPitch": 3.4,
    "RShoulderPitch": 3.4,
    "LShoulderRoll": 3.4,
    "RShoulderRoll": 3.4,
    "LElbowYaw": 0.20,
    "RElbowYaw": 0.20,
    "LElbowRoll": 0.69,
    "RElbowRoll": 0.69,
    # head, wrists and hands: held, never commanded
    "HeadYaw": 0.62,
    "HeadPitch": 1.56,
    "LWristYaw": 0.05,
    "RWristYaw": 0.05,
    "LHand": 0.05,
    "RHand": 0.05,
}
"""Per-joint position gain in N m/rad, derived as described above."""

NAO_DAMPING: dict[str, float] = {
    "LAnklePitch": 4.7,
    "RAnklePitch": 4.7,
    "LAnkleRoll": 5.1,
    "RAnkleRoll": 5.1,
    "LKneePitch": 2.5,
    "RKneePitch": 2.5,
    "LHipPitch": 1.6,
    "RHipPitch": 1.6,
    "LHipRoll": 1.9,
    "RHipRoll": 1.9,
    "LHipYawPitch": 1.9,
    "LShoulderPitch": 0.34,
    "RShoulderPitch": 0.34,
    "LShoulderRoll": 0.34,
    "RShoulderRoll": 0.34,
    "LElbowYaw": 0.02,
    "RElbowYaw": 0.02,
    "LElbowRoll": 0.07,
    "RElbowRoll": 0.07,
    "HeadYaw": 0.05,
    "HeadPitch": 0.12,
    "LWristYaw": 0.005,
    "RWristYaw": 0.005,
    "LHand": 0.005,
    "RHand": 0.005,
}
"""Per-joint velocity gain in N m s/rad, critically damped against :data:`NAO_STIFFNESS`."""

_MIMIC_ZERO_GAINS: dict[str, float] = dict.fromkeys(MIMIC_DRIVEN_JOINTS, 0.0)
"""Mimic joints are driven by a PhysX constraint, so their own drives stay off.

``RHipYawPitch`` follows ``LHipYawPitch`` because a single motor drives both on
the real robot, and the sixteen finger and thumb joints follow ``LHand`` and
``RHand``. Giving any of them a nonzero drive would have the actuator fight the
constraint, and on the finger links -- 2e-06 kg with 1.1e-09 inertia upstream --
that diverges quickly.
"""


def _stand_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Build the actuator groups for the standing configuration.

    Implicit rather than explicit actuators: PhysX then integrates the PD law at
    the 200 Hz physics rate instead of holding one torque across the whole 10 ms
    control period, which is both more accurate and closer to the real NAO,
    whose joints are position controlled by the DCM underneath whatever sends
    the targets. ``effort_limit_sim`` is left as None so the per-joint torque
    ceilings parsed from the URDF are used verbatim.
    """
    groups = {
        "legs": tuple(
            name
            for name in NAO_STIFFNESS
            if name[1:].startswith(("Hip", "Knee", "Ankle"))
        ),
        "arms": tuple(
            name
            for name in NAO_STIFFNESS
            if name[1:].startswith(("Shoulder", "Elbow"))
        ),
        "head_and_hands": tuple(
            name for name in NAO_STIFFNESS if name in HELD_JOINTS and name not in ACTUATED_JOINTS
        ),
    }
    actuators = {
        key: ImplicitActuatorCfg(
            joint_names_expr=list(names),
            stiffness={name: NAO_STIFFNESS[name] for name in names},
            damping={name: NAO_DAMPING[name] for name in names},
            effort_limit_sim=None,
            velocity_limit_sim=None,
        )
        for key, names in groups.items()
    }
    actuators["mimic_driven"] = ImplicitActuatorCfg(
        joint_names_expr=list(MIMIC_DRIVEN_JOINTS),
        stiffness=dict(_MIMIC_ZERO_GAINS),
        damping=dict(_MIMIC_ZERO_GAINS),
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    return actuators


NAO_STAND_CFG = ArticulationCfg(
    prim_path="/World/Robots/Nao",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(NAO_USD_PATH),
        # Required before a ContactSensor can read the feet: without it the
        # sensor finds no body carrying PhysxContactReportAPI and raises.
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, NAO_STAND_SPAWN_HEIGHT),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=NAO_STAND_JOINT_POS,
        joint_vel={},
    ),
    soft_joint_pos_limit_factor=NAO_SOFT_JOINT_POS_LIMIT_FACTOR,
    actuators=_stand_actuators(),
)
"""NAO configured to hold a posture: actuator gains, nominal crouch, contact reporting.

Unlike :data:`NAO_CFG` this robot does not collapse. With zero commanded offset
the drives hold the nominal posture, which makes it both the baseline controller
and a stabilising starting point for a learned policy: an action of zero is
already a working controller rather than a torque-free fall.
"""


def get_nao_stand_cfg(
    prim_path: str = "/World/Robots/Nao",
    spawn_height: float | None = None,
    force_conversion: bool = False,
) -> ArticulationCfg:
    """Return a :data:`NAO_STAND_CFG` copy with the USD guaranteed to exist.

    Args:
        prim_path: Prim path for the articulation. Regex forms such as
            ``/World/envs/env_.*/Robot`` are supported by Isaac Lab.
        spawn_height: Root height in metres. Defaults to
            :data:`NAO_STAND_SPAWN_HEIGHT`.
        force_conversion: Rebuild the derived URDF and USD even if they exist.

    Returns:
        A configured :class:`ArticulationCfg` for the NAO holding its posture.
    """
    usd_path = ensure_nao_usd(force=force_conversion)

    cfg = NAO_STAND_CFG.copy()
    cfg.prim_path = prim_path
    cfg.spawn.usd_path = str(usd_path)
    height = NAO_STAND_SPAWN_HEIGHT if spawn_height is None else spawn_height
    cfg.init_state.pos = (0.0, 0.0, height)
    return cfg
