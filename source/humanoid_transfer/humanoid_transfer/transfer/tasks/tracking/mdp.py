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
    """``exp(-e / (2 kernel^2))``, ``e`` the mean squared distance to the plan in tolerances.

    With ``kernel = 1`` this is the Gaussian likelihood of the reached state
    under the plan at the plan's own tolerance (the generator's spread, floored
    at the encoder's error): no scale is chosen by hand. A wider kernel keeps a
    gradient while the body is still far from any plan. Zero on the first step
    after a reset, which has no plan yet.
    """
    return _term(env, command_name).tracking_reward(kernel)


__all__ = ["contract_privileged", "contract_tracking", "contract_view"]
