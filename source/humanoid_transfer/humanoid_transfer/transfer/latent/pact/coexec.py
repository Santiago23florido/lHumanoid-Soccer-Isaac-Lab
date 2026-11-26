"""Pair co-executions: every body's ``z*`` at the same instants of the same replayed plan.

A replay capture (``capture_contract.py --replay-plans``) records, per step and env, the
instant of its chain the state was reached at (``replay_clock``, body times) and, per env,
its chain (``replay_chain``). Bodies step at different ``dt*``, so their rows fall at
different instants; here each body's first episode of each chain is interpolated onto one
common grid of instants, kept only where every body covers it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoExecutions:
    """``z (B, K, G, skill)`` on ``instants (G,)``; ``valid (K, G)`` where every body has it."""

    z: np.ndarray
    instants: np.ndarray
    valid: np.ndarray
    chains: np.ndarray
    """``(K,)``: the replay chain of each row."""


def first_episode(done: np.ndarray) -> np.ndarray:
    """``done (S, N) -> (S, N)`` bool: rows up to and including each env's first episode end
    (the state before the step that ended it)."""
    ended = np.cumsum(done, axis=0) - done  # episodes ended before this row
    return ended == 0


def body_on_grid(
    z: np.ndarray, clock: np.ndarray, keep: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One body: ``z (S, N, skill)``, ``clock (S, N)``, ``keep (S, N)`` -> its ``z`` on
    ``grid`` per env ``(N, G, skill)`` and where the env covers the instant ``(N, G)``.

    Only increasing runs of the clock are used (a reset restarts it).
    """
    steps, envs, skill = z.shape
    out = np.zeros((envs, len(grid), skill), dtype=np.float32)
    covered = np.zeros((envs, len(grid)), dtype=bool)
    for n in range(envs):
        rows = np.nonzero(keep[:, n])[0]
        if len(rows) < 2:
            continue
        t = clock[rows, n].astype(np.float64)
        increasing = np.concatenate([[True], np.diff(t) > 0])
        if not increasing.all():
            stop = int(np.argmin(increasing))
            rows, t = rows[:stop], t[:stop]
            if len(rows) < 2:
                continue
        inside = (grid >= t[0]) & (grid <= t[-1])
        covered[n] = inside
        for c in range(skill):
            out[n, inside, c] = np.interp(grid[inside], t, z[rows, n, c])
    return out, covered


def pair(captures: list[dict], step: float) -> CoExecutions:
    """Every body's replay capture (with ``skill_state``, ``replay_clock``, ``replay_chain``,
    ``done``) on one grid of ``step`` body times; chains are matched by their index."""
    chain_sets = [np.asarray(c["replay_chain"]) for c in captures]
    chains = np.array(sorted(set.intersection(*(set(map(int, s)) for s in chain_sets))))
    low = min(float(np.min(c["replay_clock"])) for c in captures)
    high = max(float(np.max(c["replay_clock"])) for c in captures)
    grid = np.arange(np.ceil(low / step) * step, high + 1e-9, step)
    zs, valid = [], np.ones((len(chains), len(grid)), dtype=bool)
    for c in captures:
        env_of = {int(k): n for n, k in enumerate(np.asarray(c["replay_chain"]))}
        envs = [env_of[int(k)] for k in chains]
        keep = first_episode(np.asarray(c["done"], dtype=bool))
        z, covered = body_on_grid(
            np.asarray(c["skill_state"])[:, envs],
            np.asarray(c["replay_clock"])[:, envs],
            keep[:, envs],
            grid,
        )
        zs.append(z)
        valid &= covered
    return CoExecutions(z=np.stack(zs), instants=grid, valid=valid, chains=chains)


def windows(valid: np.ndarray, length: int, stride: int) -> np.ndarray:
    """``(M, 2)`` ``(chain row, first instant)`` of every window of ``length`` valid instants,
    starting every ``stride``."""
    out = []
    for k in range(valid.shape[0]):
        run = np.convolve(valid[k].astype(int), np.ones(length, dtype=int), mode="valid")
        starts = np.nonzero(run == length)[0]
        starts = starts[(starts - (starts[0] if len(starts) else 0)) % stride == 0]
        out += [(k, int(s)) for s in starts]
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


__all__ = ["CoExecutions", "body_on_grid", "first_episode", "pair", "windows"]
