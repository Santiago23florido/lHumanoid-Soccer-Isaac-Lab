"""The body side of the contract: following a plan of ``z*`` on any robot.

A task hands a body the next ``K`` values of the dimensionless skill state
(:mod:`humanoid_transfer.transfer.latent.contract`). These are the pieces any
robot's tracking environment is built from:

:class:`ReferenceCommand`
    Samples the velocity command as Isaac Lab's own term does, and on top of
    it keeps the body's history of ``ẑ*``, draws plans from the task's
    generator, holds each for its commitment and redraws it from the history
    that actually happened. It computes the true ``z*`` on the GPU
    (:mod:`~humanoid_transfer.transfer.latent.skill_torch`) for the reward.
:func:`contract_view` / :func:`contract_path` / :func:`contract_privileged`
    What the tracker and its critic see of the plan: the states it asks for,
    and where it is going.
:func:`contract_tracking`
    The only task-shaped reward a tracker gets: how likely the state it
    reached is under the plan, at the plan's own tolerance.

The tracker never sees the command or the task: the plan is the only thing
that says what to do, so it cannot be ignored, and nothing a tracker learns is
specific to the task whose generator trained it.
"""

from .command import ReferenceCommand, ReferenceCommandCfg
from .mdp import (
    contract_left,
    contract_path,
    contract_privileged,
    contract_tracking,
    contract_view,
)

__all__ = [
    "ReferenceCommand",
    "ReferenceCommandCfg",
    "contract_left",
    "contract_path",
    "contract_privileged",
    "contract_tracking",
    "contract_view",
]
