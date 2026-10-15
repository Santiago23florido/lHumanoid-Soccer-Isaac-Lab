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
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .balance import balance_state
from .physics_prior import prior_offset

HEADS = os.environ.get("HUMANOID_HEADS", "0") == "1"
"""The second layer: the critic reads the body's measured balance, and the actor's features
must predict it (:class:`~humanoid_transfer.transfer.latent.physics_model.PhysicsActor`)."""

PRIOR = os.environ.get("HUMANOID_PRIOR", "full")
"""Ablations: ``full``; ``kinematic`` (no feed-forward); ``static`` (feed-forward only)."""

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
    if HEADS:
        observations.critic.balance = ObsTerm(
            func=balance_state, params={"command_name": command_name}
        )
        observations.physics = PhysicsTargetsCfg()


def prior_switches() -> dict[str, bool]:
    """The action term's ``kinematic`` and ``feedforward`` flags for ``HUMANOID_PRIOR``."""
    return {
        "kinematic": PRIOR in ("full", "kinematic"),
        "feedforward": PRIOR in ("full", "static"),
    }


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
    "PhysicsActorCfg",
    "with_physics_heads",
    "PHYSICS_ACTOR",
    "PHYSICS_PPO",
    "PRIOR",
    "PhysicsTargetsCfg",
    "add_physics_observations",
    "prior_switches",
]
