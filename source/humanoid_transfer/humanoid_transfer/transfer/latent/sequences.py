"""Teacher rollouts as streams a memory can read without crossing a reset.

:mod:`.dataset` cuts rollouts into single transitions, which is all a
memoryless model needs. A model with a memory needs the *history* of each step
and, for multi-step prediction, its next few successors -- and neither may
reach into another episode, because the simulator teleports the robot at a
reset and the "history" across it is two unrelated bodies.

Every ``(capture, environment)`` column of ``(T, N, ...)`` arrays is laid end
to end into one flat stream, and every step records where its episode's
trustworthy part begins and where the episode ends:

``start[s]``
    The first step of the episode whose skill state can be trusted -- the one
    after the reset step, for the reason :mod:`.dataset` drops reset steps
    (the per-body state can be read before the simulator steps the reset pose).
``end[s]``
    The episode's last recorded step.

A history window is then ``max(s - H + 1 + j, start[s])`` for ``j < H``: steps
before the episode's start repeat its first trustworthy state, which is what
the policy's buffer does at run time when it is filled at a reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...common.similarity import UnitSystem
from .dataset import dimensionless_command, falling_steps


@dataclass
class Streams:
    obs: torch.Tensor
    skill: torch.Tensor
    command: torch.Tensor
    """Dimensionless: ``(vx / V, vy / V, wz T)``."""

    action: torch.Tensor
    """The teacher's clean action -- the label, whatever was executed."""

    falling: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    capture: torch.Tensor
    env: torch.Tensor

    def __len__(self) -> int:
        return self.obs.shape[0]

    def to(self, device: torch.device | str) -> Streams:
        return Streams(**{k: getattr(self, k).to(device) for k in self.__dataclass_fields__})

    def samples(self, future: int = 0) -> torch.Tensor:
        """Steps that are trustworthy and keep ``future`` successors inside their episode."""
        steps = torch.arange(len(self), device=self.start.device)
        return steps[(steps >= self.start) & (steps + future <= self.end)]

    def windows(self, steps: torch.Tensor, length: int) -> torch.Tensor:
        """``(B, length)`` indices of each step's history, oldest first, padded at the start."""
        offsets = torch.arange(length - 1, -1, -1, device=steps.device)
        return torch.maximum(steps[:, None] - offsets, self.start[steps][:, None])

    def futures(self, steps: torch.Tensor, count: int) -> torch.Tensor:
        """``(B, count)`` indices of each step's next ``count`` successors."""
        return steps[:, None] + torch.arange(1, count + 1, device=steps.device)


def _column_bounds(done: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per step of one column: index of its episode's trustworthy start, and of its end."""
    length = done.shape[0]
    first = np.zeros(length, dtype=bool)
    first[0] = True
    first[1:] = done[:-1]
    episode = np.cumsum(first) - 1
    begins = np.flatnonzero(first)
    ends = np.append(begins[1:] - 1, length - 1)
    return begins[episode] + 1, ends[episode]


def load_streams(paths: list[Path]) -> Streams:
    """Every column of every capture, one after another, in the capture's own units."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in Streams.__dataclass_fields__}
    offset = 0
    for capture, path in enumerate(paths):
        data = np.load(path)
        units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
        done = data["done"]
        steps, envs = done.shape
        terminated = data["terminated"] if "terminated" in data.files else np.zeros_like(done)
        falling = falling_steps(terminated)
        command = dimensionless_command(data["command"], units)
        for env in range(envs):
            start, end = _column_bounds(done[:, env])
            parts["obs"].append(data["obs"][:, env])
            parts["skill"].append(data["skill_state"][:, env])
            parts["command"].append(command[:, env])
            parts["action"].append(data["action_teacher"][:, env])
            parts["falling"].append(falling[:, env])
            parts["start"].append(start + offset)
            parts["end"].append(end + offset)
            parts["capture"].append(np.full(steps, capture))
            parts["env"].append(np.full(steps, env))
            offset += steps

    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.concatenate(parts[key]), dtype=dtype)

    return Streams(
        obs=stack("obs", torch.float32),
        skill=stack("skill", torch.float32),
        command=stack("command", torch.float32),
        action=stack("action", torch.float32),
        falling=stack("falling", torch.bool),
        start=stack("start", torch.long),
        end=stack("end", torch.long),
        capture=stack("capture", torch.long),
        env=stack("env", torch.long),
    )


__all__ = ["Streams", "load_streams"]
