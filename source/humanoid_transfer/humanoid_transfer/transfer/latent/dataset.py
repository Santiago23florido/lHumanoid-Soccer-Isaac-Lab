"""Teacher rollouts as the transitions the latent model trains on.

A transition is ``(o_t, z*_t, c*_t, a_t) -> (o_{t+1}, z*_{t+1})`` inside one
episode. Two kinds of step are dropped, both because of resets:

* the last step of an episode, whose successor belongs to a new one;
* the first step of an episode, whose per-body state can be read before the
  simulator has stepped the reset pose, so its ``z*`` is not trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...common.similarity import UnitSystem


@dataclass
class Transitions:
    obs: torch.Tensor
    next_obs: torch.Tensor
    skill: torch.Tensor
    next_skill: torch.Tensor
    command: torch.Tensor
    """Dimensionless: ``(vx / V, vy / V, wz T)``."""

    action: torch.Tensor
    """The teacher's clean action -- the label, whatever was executed."""

    env: torch.Tensor

    falling: torch.Tensor
    """True within :data:`FALL_HORIZON` steps before the episode terminates (not times out)."""

    def __len__(self) -> int:
        return self.obs.shape[0]

    def subset(self, mask: torch.Tensor) -> Transitions:
        return Transitions(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})

    def to(self, device: torch.device | str) -> Transitions:
        return Transitions(**{k: getattr(self, k).to(device) for k in self.__dataclass_fields__})


FALL_HORIZON = 50
"""One second at 50 Hz. States this close to a fall are reported apart from walking."""


def falling_steps(terminated: np.ndarray, horizon: int = FALL_HORIZON) -> np.ndarray:
    """``(T, N)`` True at the ``horizon`` steps leading up to a termination, inclusive."""
    falling = np.zeros_like(terminated, dtype=bool)
    for t, n in zip(*np.nonzero(terminated), strict=True):
        falling[max(0, t - horizon + 1) : t + 1, n] = True
    return falling


def dimensionless_command(command: np.ndarray, units: UnitSystem) -> np.ndarray:
    """``(vx, vy, wz)`` in SI to ``(vx / V, vy / V, wz T)``."""
    scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
    return np.asarray(command, dtype=float) * scale


def valid_steps(done: np.ndarray) -> np.ndarray:
    """``(T, N)``: steps ``t`` and ``t + 1`` share an episode, and ``t`` is not its first."""
    valid = np.zeros_like(done, dtype=bool)
    valid[1:-1] = ~done[1:-1] & ~done[:-2]
    return valid


def load_transitions(path: Path) -> Transitions:
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    t, n = np.nonzero(valid_steps(data["done"]))

    def take(key: str, offset: int = 0) -> torch.Tensor:
        return torch.as_tensor(data[key][t + offset, n], dtype=torch.float32)

    command = dimensionless_command(data["command"][t, n], units)
    terminated = data["terminated"] if "terminated" in data.files else np.zeros_like(data["done"])
    return Transitions(
        obs=take("obs"),
        next_obs=take("obs", 1),
        skill=take("skill_state"),
        next_skill=take("skill_state", 1),
        command=torch.as_tensor(command, dtype=torch.float32),
        action=take("action_teacher"),
        env=torch.as_tensor(n, dtype=torch.long),
        falling=torch.as_tensor(falling_steps(terminated)[t, n]),
    )


__all__ = [
    "FALL_HORIZON",
    "Transitions",
    "dimensionless_command",
    "falling_steps",
    "load_transitions",
    "valid_steps",
]
