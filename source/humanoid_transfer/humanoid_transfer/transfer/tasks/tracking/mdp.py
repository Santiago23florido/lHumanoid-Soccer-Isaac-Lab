"""Observation and reward terms that read a :class:`~.command.ReferenceCommand`.

Thin on purpose: the term owns the plan, the history and the true state, so
that the tracker, its critic and its reward all read one consistent copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command import ReferenceCommand


def _term(env: ManagerBasedRLEnv, command_name: str) -> ReferenceCommand:
    return env.command_manager.get_term(command_name)


def contract_view(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The tracker's view: the held plan relative to its ``ẑ*``, and ``ẑ*``."""
    return _term(env, command_name).observation()


def contract_privileged(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The critic's view: the plan relative to the true ``z*``, the true ``z*``, the plan's age."""
    return _term(env, command_name).privileged()


def contract_tracking(
    env: ManagerBasedRLEnv, command_name: str, kernel: float = 1.0
) -> torch.Tensor:
    """Mean over the fields of ``z*`` of ``exp(-e_f / (2 kernel^2))``.

    ``e_f`` is the field's mean squared distance to the plan in units of the
    plan's tolerance -- the generator's own spread, floored at the encoder's
    error, so no scale is chosen by hand. With ``kernel = 1`` each field's term
    is its Gaussian likelihood under the plan.

    That alone is too sharp to learn from: the teacher's walk is so regular
    that the tolerance on height and on the feet's height is one or two
    millimetres on the G1, and a body that stands slightly differently scores
    hundreds of tolerances and a reward of zero, with no gradient. Summing the
    term at kernels 1, 10 and 100 makes a scale mixture -- a heavy-tailed
    likelihood -- that still peaks at the plan but rewards getting closer from
    anywhere; scoring fields apart gives credit for each quantity reached.
    Zero on the first step after a reset, which has no plan yet.
    """
    return _term(env, command_name).tracking_reward(kernel)


__all__ = ["contract_privileged", "contract_tracking", "contract_view"]
