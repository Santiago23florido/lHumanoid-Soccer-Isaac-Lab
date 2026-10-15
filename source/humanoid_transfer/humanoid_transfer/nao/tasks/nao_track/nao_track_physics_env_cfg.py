"""The NAO tracker with its body's physics in the action (decoder v2, skill_space.md §9).

:class:`~.nao_track_env_cfg.NaoTrackEnvCfg` with the action of
:class:`~humanoid_transfer.transfer.tasks.tracking.physics_prior.PhysicsPriorAction`:
the legs' targets from a least-squares step towards the plan's feet and the
plan's leg forces and gravity as feed-forward effort, the policy's output a
residual on them. The NAO's two hip yaw-pitch joints are one motor
(``RHipYawPitch`` mimics ``LHipYawPitch``), so its legs give eleven columns for
the twelve rows of the two feet's tasks, solved in the least-squares sense.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from ....transfer.tasks.tracking.physics_cfg import add_physics_observations, prior_switches
from ....transfer.tasks.tracking.physics_prior import PhysicsPriorActionCfg
from ...assets import nao_kinematics as nk
from .nao_track_env_cfg import (
    BASE_BODY,
    COMMAND,
    FOOT_BODIES,
    NaoTrackEnvCfg,
    NaoTrackEnvCfg_PLAY,
    sole_points,
)

LEG_JOINTS = ["LHipYawPitch", "[LR]HipRoll", "[LR]HipPitch", "[LR]KneePitch", "[LR]AnklePitch",
              "[LR]AnkleRoll"]  # fmt: skip
"""The joints the kinematic prior moves; ``RHipYawPitch`` follows ``LHipYawPitch``."""

RESIDUAL_SCALE = float(os.environ.get("HUMANOID_RESIDUAL_SCALE", "0.125"))
"""Half the plain tracker's 0.25 rad per unit action, since the prior carries the legs."""


def _prior_action() -> PhysicsPriorActionCfg:
    return PhysicsPriorActionCfg(
        asset_name="robot",
        joint_names=list(nk.ACTUATED_JOINTS),
        scale=RESIDUAL_SCALE,
        use_default_offset=True,
        preserve_order=True,
        command_name=COMMAND,
        ik_joint_names=list(LEG_JOINTS),
        mimic={"RHipYawPitch": "LHipYawPitch"},
        foot_bodies=list(FOOT_BODIES),
        sole_points=sole_points(),
        base_body=BASE_BODY,
        **prior_switches(),
    )


@configclass
class NaoTrackPhysicsEnvCfg(NaoTrackEnvCfg):
    """The NAO tracker whose action is a residual on its body's physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos = _prior_action()
        add_physics_observations(self.observations, COMMAND)


@configclass
class NaoTrackPhysicsEnvCfg_PLAY(NaoTrackEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos = _prior_action()
        add_physics_observations(self.observations, COMMAND)
