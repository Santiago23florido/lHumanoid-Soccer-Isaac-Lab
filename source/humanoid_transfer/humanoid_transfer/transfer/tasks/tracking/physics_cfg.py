"""What any body's tracking task adds to become decoder v2 (skill_space.md §9, F4).

A body's physics variant replaces its action term with a
:class:`~.physics_prior.PhysicsPriorActionCfg` and calls :func:`add_physics_observations`:
the policy and its critic see how far the prior is from the joints, and with the heads
(``HUMANOID_HEADS=1``) the critic also reads the measured balance and a ``physics`` group
carries it as the actor's heads' targets.
"""

from __future__ import annotations

import os

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .balance import balance_state
from .physics_prior import joint_target_rate_l2, prior_feedforward, prior_offset

HEADS = os.environ.get("HUMANOID_HEADS", "0") == "1"
"""The second layer: the critic reads the body's measured balance, and the actor's features
must predict it (:class:`~humanoid_transfer.transfer.latent.physics_model.PhysicsActor`)."""

PRIOR = os.environ.get("HUMANOID_PRIOR", "full")
"""Ablations: ``full``; ``kinematic`` (no feed-forward); ``static`` (feed-forward only)."""

PLAN_TRUST = os.environ.get("HUMANOID_PLAN_TRUST", "0") == "1"
PLAN_FUSION = os.environ.get("HUMANOID_PLAN_FUSION", "shrink")
"""How a trusted plan is fused (``PhysicsPriorActionCfg.plan_fusion``): ``shrink`` or ``mean``."""
"""The prior trusts a plan by its measured sampling noise and aims the trunk at its slow tilt;
the biped's plans carry the generator's measured noise and the gait's trunk sway (§9.1)."""

TARGET_RATE = os.environ.get("HUMANOID_TARGET_RATE", "0") == "1"
"""The smoothness cost on the joint targets, in the plain tracker's units, instead of on the
residual (:func:`~.physics_prior.joint_target_rate_l2`). Off, the inherited ``action_rate_l2``
reads the residual: the runs of §9 and §9.1 before the warm study's second round."""

FEEDFORWARD_OBS = os.environ.get("HUMANOID_FEEDFORWARD_OBS", "0") == "1"
"""The policy and critic also see the prior's feed-forward (over each joint's stiffness), not
only its targets' offset: a residual that must correct the prior then sees all of it. Off,
the earlier physics trackers' observation (a distilled student had to infer the feed-forward
the teacher's labels were compensated for, §9.1)."""

SWAY = 0.035
"""Largest procedural trunk sway, radians: twice the walk teacher's (0.015 pitch, 0.017 roll),
as the family's other ranges reach beyond the one task they were checked against."""

PHYSICS_ACTOR = "humanoid_transfer.transfer.latent.physics_model:PhysicsActor"
PHYSICS_PPO = "humanoid_transfer.transfer.latent.physics_model:PhysicsPPO"


@configclass
class PhysicsTargetsCfg(ObsGroup):
    """What the actor's physics heads predict: each foot's centre of pressure and ground force."""

    balance = ObsTerm(func=balance_state, params={"command_name": "base_velocity"})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


def add_physics_observations(observations, command_name: str = "base_velocity") -> None:
    """The prior's offset for the policy and critic; with the heads, the balance and targets."""
    prior = ObsTerm(func=prior_offset, params={"action_name": "joint_pos"})
    observations.policy.prior = prior
    observations.critic.prior = prior
    if FEEDFORWARD_OBS:
        feedforward = ObsTerm(func=prior_feedforward, params={"action_name": "joint_pos"})
        observations.policy.feedforward = feedforward
        observations.critic.feedforward = feedforward
    if HEADS:
        observations.critic.balance = ObsTerm(
            func=balance_state, params={"command_name": command_name}
        )
        observations.physics = PhysicsTargetsCfg()


def prior_switches() -> dict[str, bool]:
    """The action term's flags for ``HUMANOID_PRIOR`` and ``HUMANOID_PLAN_TRUST``."""
    return {
        "kinematic": PRIOR in ("full", "kinematic"),
        "feedforward": PRIOR in ("full", "static"),
        "trust_plan": PLAN_TRUST,
        "slow_tilt": PLAN_TRUST,
        "plan_fusion": PLAN_FUSION,
    }


def rate_on_targets(rewards, plain_scale: float) -> None:
    """With ``HUMANOID_TARGET_RATE=1``: ``action_rate_l2`` becomes the joint targets' rate, at
    its weight, in units of the plain tracker's action scale ``plain_scale``."""
    if not TARGET_RATE or rewards.action_rate_l2 is None:
        return
    weight = rewards.action_rate_l2.weight
    rewards.action_rate_l2 = None
    rewards.joint_target_rate_l2 = RewTerm(
        func=joint_target_rate_l2,
        weight=weight,
        params={"action_name": "joint_pos", "scale": float(plain_scale)},
    )


def trust_plans(command) -> None:
    """With ``HUMANOID_PLAN_TRUST=1``: the command measures its plans' noise, and its biped's
    plans carry that noise and the gait's trunk sway."""
    if not PLAN_TRUST:
        return
    command.plan_trust = True
    command.procedural_plan_noise = True
    command.procedural_sway = SWAY


@configclass
class PhysicsActorCfg(RslRlMLPModelCfg):
    """An MLP actor whose last hidden layer also predicts the ``physics`` observation group."""

    class_name: str = PHYSICS_ACTOR
    physics_group: str = "physics"
    head_dims: list[int] = [128]


def with_physics_heads(runner) -> None:
    """With ``HUMANOID_HEADS=1``, the runner's actor gets the heads and PPO reports their loss."""
    if not HEADS:
        return
    actor = runner.actor
    runner.actor = PhysicsActorCfg(
        hidden_dims=list(actor.hidden_dims),
        activation=actor.activation,
        obs_normalization=actor.obs_normalization,
        distribution_cfg=actor.distribution_cfg,
    )
    runner.algorithm.class_name = PHYSICS_PPO


__all__ = [
    "HEADS",
    "PLAN_TRUST",
    "SWAY",
    "FEEDFORWARD_OBS",
    "TARGET_RATE",
    "rate_on_targets",
    "trust_plans",
    "PhysicsActorCfg",
    "with_physics_heads",
    "PHYSICS_ACTOR",
    "PHYSICS_PPO",
    "PRIOR",
    "PhysicsTargetsCfg",
    "add_physics_observations",
    "prior_switches",
]
