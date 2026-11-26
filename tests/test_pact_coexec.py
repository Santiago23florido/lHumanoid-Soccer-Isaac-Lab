"""Co-execution pairing: bodies at different steps meet on one grid of instants."""

from __future__ import annotations

import numpy as np
from humanoid_transfer.transfer.latent.pact.coexec import (
    body_on_grid,
    first_episode,
    pair,
    windows,
)


def _capture(dt: float, steps: int, chains: list[int], start: float = 1.0, fall=None) -> dict:
    """A body whose z* is a known function of the instant: z_c = sin(tau + c)."""
    envs = len(chains)
    clock = start + dt * np.arange(steps)[:, None].repeat(envs, 1)
    done = np.zeros((steps, envs), dtype=bool)
    if fall is not None:
        env, at = fall
        done[at, env] = True
        clock[at + 1 :, env] = start + dt * np.arange(steps - at - 1)  # restarted
    z = np.stack([np.sin(clock + c) for c in range(27)], -1)
    return {
        "skill_state": z,
        "replay_clock": clock,
        "replay_chain": np.asarray(chains),
        "done": done,
    }


def test_first_episode_keeps_the_state_before_the_ending_step() -> None:
    done = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=bool)
    keep = first_episode(done)
    assert keep[:, 0].tolist() == [True, True, False, False]
    assert keep[:, 1].tolist() == [True, True, True, True]


def test_bodies_at_different_steps_meet_on_the_grid() -> None:
    g1 = _capture(0.0774, 400, [0, 1, 2])
    nao = _capture(0.1261, 300, [2, 0, 1])  # other env order, same chains
    paired = pair([g1, nao], step=0.1261)
    assert paired.z.shape[:3] == (2, 3, len(paired.instants))
    tau = paired.instants
    for b in range(2):
        for k in range(3):
            ok = paired.valid[k]
            expected = np.sin(tau[ok] + 5)
            # Linear interpolation of a sine sampled every <= 0.13: error < dt^2 / 8.
            assert np.abs(paired.z[b, k, ok, 5] - expected).max() < 0.0021
    # Every body covers the instants kept; the G1 ends first (1 + 0.0774 * 399).
    assert paired.instants[paired.valid[0]].max() <= 1 + 0.0774 * 399 + 1e-9


def test_a_fall_ends_the_pairs_of_that_chain() -> None:
    g1 = _capture(0.0774, 400, [0, 1], fall=(1, 100))
    nao = _capture(0.1261, 300, [0, 1])
    paired = pair([g1, nao], step=0.1261)
    last = paired.instants[paired.valid[1]].max()
    assert last <= 1 + 0.0774 * 100 + 1e-9
    assert paired.instants[paired.valid[0]].max() > 20


def test_body_on_grid_ignores_a_restarted_clock() -> None:
    clock = np.array([[1.0], [1.1], [1.2], [1.0], [1.1]])
    z = clock[..., None].repeat(2, -1)
    keep = np.ones_like(clock, dtype=bool)
    out, covered = body_on_grid(z, clock, keep, np.array([1.05, 1.15, 1.3]))
    assert covered[0].tolist() == [True, True, False]
    assert np.allclose(out[0, :2, 0], [1.05, 1.15])


def test_windows_cover_only_valid_runs() -> None:
    valid = np.array([[1, 1, 1, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    w = windows(valid, length=3, stride=2)
    assert w.tolist() == [[0, 0], [0, 2], [1, 1], [1, 3], [1, 5]]
