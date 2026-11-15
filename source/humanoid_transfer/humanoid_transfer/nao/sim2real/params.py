"""The robot's parameters as a teacher reads them and its student estimates them: ``robot_params``.

Five numbers describe how a simulated NAO's drives differ from the nominal model
-- its latency and its leg drives' stiffness, damping, torque ceiling and reflected
rotor inertia, each as the factor drawn for the whole robot (the joints scatter
around it, :mod:`.events`) -- and are what a harness identification of the real
robot measures (``nao/scripts/fit_actuators.py``). A teacher policy reads them;
its student estimates them from its own history, and on the robot they can be
given measured values instead.

Each is normalised to ``[-1, 1]`` over a fixed range (:data:`PARAMS`' ranges in
``NaoTrack-Sim2Real-v0``'s ``ROBOT_RANGES``), and clipped there.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

PARAMS = ("delay_ms", "stiffness_scale", "damping_scale", "effort_scale", "armature_scale")
"""The parameters, in the order of ``robot_params``' columns."""


def robot_draws(env) -> dict[str, torch.Tensor]:
    """Each robot's drawn factor per drive parameter (``(N,)``, 1 before any draw)."""
    draws = getattr(env, "_robot_draws", None)
    if draws is None:
        draws = {key: torch.ones(env.num_envs, device=env.device) for key in PARAMS[1:]}
        env._robot_draws = draws
    return draws


def robot_param_values(env, action_name: str = "joint_pos") -> torch.Tensor:
    """``(N, 5)`` each robot's parameters in their own units (ms, factors)."""
    term = env.action_manager.get_term(action_name)
    delay = getattr(term, "delay", None)
    if delay is None:
        delay_ms = torch.zeros(env.num_envs, device=env.device)
    else:
        delay_ms = delay.float() * (env.physics_dt * 1000.0)
    draws = robot_draws(env)
    return torch.stack([delay_ms, *(draws[key] for key in PARAMS[1:])], dim=-1)


def draw_factors(
    count: int,
    width: int,
    scale_range: tuple[float, float],
    spread: float | None,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(count, width)`` per-joint factors and ``(count,)`` the robot's own (:mod:`.events`).

    Without ``spread`` the joints are drawn independently and the robot's factor is their
    mean; with it, the robot's factor is drawn and each joint is it times ``1 ± spread``.
    """
    low, high = scale_range
    if spread is None:
        per_joint = low + (high - low) * torch.rand(count, width, device=device)
        return per_joint, per_joint.mean(dim=-1)
    robot = low + (high - low) * torch.rand(count, 1, device=device)
    scatter = 1.0 + spread * (2.0 * torch.rand(count, width, device=device) - 1.0)
    return robot * scatter, robot[:, 0]


def normalise(values: torch.Tensor, ranges: Sequence[Sequence[float]]) -> torch.Tensor:
    """Each column mapped from its ``[low, high]`` to ``[-1, 1]``, and clipped there."""
    bounds = torch.as_tensor(ranges, dtype=values.dtype, device=values.device)
    low, high = bounds[:, 0], bounds[:, 1]
    return (2.0 * (values - low) / (high - low) - 1.0).clamp(-1.0, 1.0)


def robot_params(
    env, ranges: Sequence[Sequence[float]], action_name: str = "joint_pos"
) -> torch.Tensor:
    """Observation term: ``(N, 5)`` the robot's parameters, normalised (see the module)."""
    return normalise(robot_param_values(env, action_name), ranges)


__all__ = [
    "PARAMS",
    "draw_factors",
    "normalise",
    "robot_draws",
    "robot_param_values",
    "robot_params",
]
