"""Captures of several bodies as windows a PACT encoder reads.

A capture (``capture_velocity.py --joints``) holds, per control step and
environment, the joints, the base's angular velocity and projected gravity, and the
true ``z*``. Here each becomes, per sample:

- ``joint_obs (J, history, 2)``: joint angle (rad) and joint speed times
  ``sqrt(L/g)``, the body's own time unit;
- ``imu (history, 6)``: projected gravity and base angular velocity times
  ``sqrt(L/g)``;
- ``z (27)``: the true skill state at the window's last step.

Windows never cross an episode boundary. Environments are split, not steps, so a
test window never shares an episode with a training one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_transfer.transfer.latent.pact.descriptors import GRAVITY


@dataclass
class BodyWindows:
    """Every window of one body's captures, as arrays indexed by sample."""

    body: str
    joint_obs: np.ndarray
    """``(N, J, history, 2)`` float32."""
    imu: np.ndarray
    """``(N, history, 6)`` float32."""
    z: np.ndarray
    """``(N, 27)`` float32."""
    group: np.ndarray
    """``(N,)`` int: capture index * 1000 + environment, the unit of the test split
    and of the bootstrap."""


def windows_of_capture(
    path: Path, history: int, leg_length: float, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(joint_obs, imu, z, env)`` of every complete window of one capture."""
    data = np.load(path)
    names = tuple(str(n) for n in data["joint_names"])
    if names != tuple(joint_names):
        raise ValueError(f"{path}: joints {names[:4]}... differ from the card's")
    time = np.sqrt(leg_length / GRAVITY)
    q = data["joint_pos"]
    qd = data["joint_vel"] * time
    imu = np.concatenate([data["projected_gravity"], data["base_ang_vel"] * time], axis=-1)
    z = data["skill_state"]
    done = data["done"]
    steps, envs = done.shape

    # A window ending at t is valid when no step in [t - history + 1, t - 1] ended an episode.
    ended = np.zeros((steps + 1, envs), dtype=np.int64)
    ended[1:] = np.cumsum(done, axis=0)
    t = np.arange(history - 1, steps)
    clean = (ended[t] - ended[t - history + 1]) == 0
    t_idx, e_idx = np.nonzero(clean)
    t_end = t[t_idx]
    offsets = np.arange(-history + 1, 1)
    rows = t_end[:, None] + offsets[None, :]
    joint_obs = np.stack([q[rows, e_idx[:, None]], qd[rows, e_idx[:, None]]], axis=-1)
    return (
        joint_obs.transpose(0, 2, 1, 3).astype(np.float32),
        imu[rows, e_idx[:, None]].astype(np.float32),
        z[t_end, e_idx].astype(np.float32),
        e_idx.astype(np.int64),
    )


def load_body(
    body: str,
    paths: list[Path],
    history: int,
    leg_length: float,
    joint_names: tuple[str, ...],
) -> BodyWindows:
    """All windows of a body's captures."""
    parts = [windows_of_capture(p, history, leg_length, joint_names) for p in sorted(paths)]
    return BodyWindows(
        body=body,
        joint_obs=np.concatenate([p[0] for p in parts]),
        imu=np.concatenate([p[1] for p in parts]),
        z=np.concatenate([p[2] for p in parts]),
        group=np.concatenate([i * 1000 + p[3] for i, p in enumerate(parts)]),
    )


def split_by_environment(group: np.ndarray, test_envs: int = 8, envs: int = 32) -> np.ndarray:
    """True on test samples: the last ``test_envs`` environments of every capture."""
    return (group % 1000) >= envs - test_envs
