"""Isaac Lab articulation configuration for the SoftBank / Aldebaran NAO H25 V5.0.

Phase 1 asset definition only: it describes the robot, and nothing about tasks,
rewards, observations, actions or policies. RL environments added later should
import :data:`NAO_CFG` (or call :func:`get_nao_cfg`) and adjust ``prim_path``
and ``init_state`` rather than redefining the robot.

The USD this configuration points at is a build artifact generated from the
vendored upstream URDF; see :mod:`humanoid_soccer_lab.assets.nao_usd` and
``third_party/nao/README.md``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .nao_paths import NAO_USD_PATH, default_joint_positions
from .nao_usd import ensure_nao_usd

__all__ = [
    "NAO_CFG",
    "get_nao_cfg",
    "NAO_SPAWN_HEIGHT",
    "NAO_SOLE_OFFSET_Z",
    "NAO_EXPECTED_JOINTS",
    "NAO_DEFAULT_JOINT_POS",
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
"""Reusable NAO articulation configuration.

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
