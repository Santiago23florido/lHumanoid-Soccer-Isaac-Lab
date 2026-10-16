"""The G1 tracker with its body's physics in the action (decoder v2, skill_space.md §9).

Everything of :class:`~.g1_track_env_cfg.G1TrackEnvCfg` -- command, rewards,
terminations, randomisation -- with one change: the policy's output is a
residual on the physics prior of
:class:`~humanoid_transfer.transfer.tasks.tracking.physics_prior.PhysicsPriorAction`
(the legs' targets from a least-squares step towards the plan's feet, the
plan's leg forces and gravity as feed-forward effort), and the policy and its
critic see how far that prior is from the joints. ``HUMANOID_HEADS=1`` adds the
physics heads' targets and the critic's balance
(:mod:`~humanoid_transfer.transfer.tasks.tracking.physics_cfg`).
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from ....transfer.tasks.tracking.physics_cfg import (
    add_physics_observations,
    prior_switches,
    trust_plans,
)
from ....transfer.tasks.tracking.physics_prior import PhysicsPriorActionCfg
from .g1_track_env_cfg import (
    BASE_BODY,
    COMMAND,
    FOOT_BODIES,
    G1TrackEnvCfg,
    G1TrackEnvCfg_PLAY,
    _body_card,
)

LEG_JOINTS = [".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint",
              ".*_ankle_pitch_joint", ".*_ankle_roll_joint"]  # fmt: skip
"""The joints the kinematic prior moves: six per leg, a square task per foot."""

RESIDUAL_SCALE = float(os.environ.get("HUMANOID_RESIDUAL_SCALE", "0.25"))
"""The policy's residual in radians per unit action: half the plain tracker's 0.5, since the
prior carries the legs' motion."""


def _prior_action() -> PhysicsPriorActionCfg:
    card = _body_card()
    return PhysicsPriorActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=RESIDUAL_SCALE,
        use_default_offset=True,
        command_name=COMMAND,
        ik_joint_names=list(LEG_JOINTS),
        foot_bodies=list(FOOT_BODIES),
        sole_points=[list(card["foot"][name]["sole_point_link"]) for name in FOOT_BODIES],
        base_body=BASE_BODY,
        **prior_switches(),
    )


@configclass
class G1TrackPhysicsEnvCfg(G1TrackEnvCfg):
    """The G1 tracker whose action is a residual on its body's physics."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos = _prior_action()
        add_physics_observations(self.observations, COMMAND)
        trust_plans(self.commands.base_velocity)


@configclass
class G1TrackPhysicsEnvCfg_PLAY(G1TrackEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos = _prior_action()
        add_physics_observations(self.observations, COMMAND)
        trust_plans(self.commands.base_velocity)
