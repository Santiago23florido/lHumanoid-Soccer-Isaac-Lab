"""The decoder's training windows: G-D0's complete windows, plus the episode starts the
driver acts at, refilled with the first reading as the driver refills its history."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from pact_gate_d import window_rows  # noqa: E402

H = 4


def _g_d0_windows(done: np.ndarray) -> set[tuple[int, int]]:
    """G-D0's rule: ``t >= h``, label in the capture, no reset in ``[t - h, t]``."""
    steps, envs = done.shape
    return {
        (t, e) for t in range(H, steps - 1) for e in range(envs) if not done[t - H : t + 1, e].any()
    }


def test_without_resets_the_windows_are_g_d0s():
    done = np.zeros((20, 3), bool)
    t, env, rows, start = window_rows(done, H)
    assert set(zip(t.tolist(), env.tolist(), strict=True)) == _g_d0_windows(done)
    assert not start.any()
    np.testing.assert_array_equal(rows, t[:, None] + np.arange(-H + 1, 1)[None])


def test_episode_starts_are_added_with_the_first_reading_repeated():
    done = np.zeros((20, 2), bool)
    done[7, 0] = True  # env 0: a new episode starts at step 8
    t, env, rows, start = window_rows(done, H)
    pairs = set(zip(t.tolist(), env.tolist(), strict=True))
    assert _g_d0_windows(done) <= pairs
    added = pairs - _g_d0_windows(done)
    assert added == {(8, 0), (9, 0), (10, 0), (11, 0)}  # ages 0..3 of the new episode
    for k, (tk, ek) in enumerate(zip(t.tolist(), env.tolist(), strict=True)):
        if (tk, ek) in added:
            assert start[k]
            expected = np.maximum(np.arange(tk - H + 1, tk + 1), 8)
            np.testing.assert_array_equal(rows[k], expected)
        else:
            assert not start[k]
    assert (7, 0) not in pairs  # its label would be the next episode's
    assert (12, 0) in pairs and not start[(t == 12) & (env == 0)].any()  # complete again


def test_an_episode_begun_before_the_capture_keeps_g_d0s_rule():
    done = np.zeros((10, 1), bool)
    t, _, _, start = window_rows(done, H)
    assert t.min() == H and not start.any()


def test_ordered_by_step_then_environment():
    done = np.zeros((12, 3), bool)
    done[5, 1] = True
    t, env, _, _ = window_rows(done, H)
    key = t * 10 + env
    assert np.all(np.diff(key) > 0)
