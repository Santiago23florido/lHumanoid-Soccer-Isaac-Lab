"""What any body's tracking task adds to read its plan with decoder v2's core (§9, F4).

The core's actor (:class:`~humanoid_transfer.transfer.latent.core_net.PlanCoreNet` in a
:class:`~humanoid_transfer.transfer.latent.physics_model.PhysicsActor`) reads one policy
group laid out for it:

* the plan as tokens, first: each row of the view's ``z*`` relative to ``ẑ*`` and where the
  reference will be then (:func:`~.mdp.contract_tokens`);
* the body's proprioception with a short history (:data:`HISTORY` steps, oldest first);
* ``ẑ*`` (:func:`~.mdp.contract_estimate`), and whatever the task adds after (the prior's
  offset in the physics task).

The plain tracker's own policy group stays, as ``plain``: an incumbent distilled into the
core reads it (:class:`~.teacher_view.TeacherView`), and no learner does.
"""

from __future__ import annotations

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .mdp import contract_estimate, contract_tokens
from .physics_cfg import HEADS, PHYSICS_ACTOR, PHYSICS_PPO, PhysicsActorCfg

HISTORY = 4
"""Steps of proprioception the core sees (pre-registered with F4)."""

HISTORY_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
)
"""The plain group's terms read with a history; the plan's are replaced by the tokens."""

PLAN_TERMS = ("contract", "contract_path")

TOKEN_WIDTH = 30
"""A token: one row's 27 numbers of ``z*`` and its 3 of path."""


@configclass
class CorePolicyCfg(ObsGroup):
    """The core's policy group: the plan's tokens first, then the rest of the plain group."""

    tokens = ObsTerm(func=contract_tokens, params={"command_name": "base_velocity"})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


def core_policy(plain: ObsGroup, command_name: str, history: int = HISTORY) -> CorePolicyCfg:
    """The core's group from the plain one: its proprioception with a history, its plan as
    tokens, ``ẑ*`` after them; any other term kept as it is, in its order."""
    group = CorePolicyCfg()
    group.tokens.params = {"command_name": command_name}
    group.enable_corruption = plain.enable_corruption
    for name, term in vars(plain).items():
        if not isinstance(term, ObsTerm) or name in PLAN_TERMS:
            continue
        term = copy.deepcopy(term)
        if name in HISTORY_TERMS:
            term.history_length = history
            term.flatten_history_dim = True
        setattr(group, name, term)
    group.estimate = ObsTerm(func=contract_estimate, params={"command_name": command_name})
    return group


def add_core_observations(observations, command_name: str = "base_velocity") -> None:
    """The plain policy group kept as ``plain``; the policy group becomes the core's."""
    observations.plain = copy.deepcopy(observations.policy)
    observations.policy = core_policy(observations.policy, command_name)


@configclass
class CoreActorCfg(PhysicsActorCfg):
    """The core's actor: the plan's temporal encoder and a residual trunk, heads optional."""

    class_name: str = PHYSICS_ACTOR
    physics_group: str | None = None
    network: str = "plan_core"
    network_cfg: dict = {"tokens": 8, "token_width": TOKEN_WIDTH}


def with_core(runner, view: int = 8) -> None:
    """The runner's actor becomes the core's ([512, 512, 256], SiLU); with ``HUMANOID_HEADS=1``
    in a physics task it also carries the heads, and PPO reports their loss."""
    actor = runner.actor
    heads = HEADS and getattr(actor, "physics_group", None) is not None
    runner.actor = CoreActorCfg(
        hidden_dims=[512, 512, 256],
        activation="swish",
        obs_normalization=actor.obs_normalization,
        distribution_cfg=actor.distribution_cfg,
        physics_group="physics" if heads else None,
        network_cfg={"tokens": view, "token_width": TOKEN_WIDTH},
    )
    if heads:
        runner.algorithm.class_name = PHYSICS_PPO


__all__ = [
    "HISTORY",
    "HISTORY_TERMS",
    "TOKEN_WIDTH",
    "CoreActorCfg",
    "CorePolicyCfg",
    "add_core_observations",
    "core_policy",
    "with_core",
]
