"""The oracle's fall, measured from the ground under the robot, and its guard against a
diverged simulation.

Kept apart from the environment configuration, which needs the Kit application, so the
tests can run it on a stand-in scene.
"""

from __future__ import annotations

import functools

import torch

DIVERGED = 1.0e3
"""A bound no state of the NAO's reaches (m/s, rad, rad/s): its joints turn at most ~10
rad/s and it walks below 1 m/s. PhysX's divergences on the rough terrains reached 1e5 to
1e10 (``logs/contract/train_oracle_rough.log``) and then NaN."""


def base_below(
    env, minimum_height: float, scanner: str | None = None, robot: str = "robot"
) -> torch.Tensor:
    """The trackers' fall: the base below ``minimum_height`` above the ground under it.

    Isaac Lab's ``root_height_below_minimum`` measures from the world's zero, which is the
    ground only on a plane: on the rough terrains a robot spawned in a sunken sub-terrain
    fell on its first step. With a height ``scanner`` the ground is its rays' mean, as
    Isaac Lab's ``base_height_l2`` reads it; on a plane the two agree.
    """
    height = env.scene[robot].data.root_pos_w[:, 2]
    if scanner is not None:
        height = height - env.scene[scanner].data.ray_hits_w[..., 2].mean(dim=1)
    return height < minimum_height


def diverged(env, robot: str = "robot") -> torch.Tensor:
    """Environments whose simulation has diverged: a base or joint state not finite, or
    beyond :data:`DIVERGED`. Ended as a time-out: it is the simulator's failure, not the
    policy's fall, and its next observation is the reset's."""
    data = env.scene[robot].data
    state = torch.cat(
        [data.root_lin_vel_w, data.root_ang_vel_w, data.joint_pos, data.joint_vel], dim=-1
    )
    return ~torch.isfinite(state).all(dim=-1) | (state.abs() > DIVERGED).any(dim=-1)


def finite(func):
    """``func``, a reward term, with a diverged environment's value set to zero.

    Rewards are computed before the step's resets, so a diverged state would reach PPO
    once, as NaN or ~1e18, before :func:`diverged` resets it. Every other value is
    ``func``'s; ``functools.wraps`` keeps its signature, which Isaac Lab checks against the
    term's parameters.
    """

    @functools.wraps(func)
    def term(env, **params):
        value = func(env, **params)
        return torch.where(diverged(env), torch.zeros_like(value), value)

    # Hydra carries the configuration through "module:name" strings and back (Isaac Lab's
    # update_class_from_dict): the wrapper has to resolve to itself, not to ``func``.
    term.__module__ = __name__
    term.__name__ = term.__qualname__ = f"finite_{func.__name__}"
    globals()[term.__name__] = term
    return term
