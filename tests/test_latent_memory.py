"""Checks on the shared skill's memory: its clock, its windows and what it may see.

The memory is where the gait's timing is supposed to live, and also where a
body could leak back into the shared space. The tests below pin both down on
the wiring rather than on training outcomes: windows never reach into another
episode, the grid means the same body time on every body, and the memory's
only input is the physical history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.memory import TimeGrid  # noqa: E402
from humanoid_transfer.transfer.latent.probes import linear_probe, mlp_probe  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402

G1_DT_STAR = 0.02 / np.sqrt(0.65457 / 9.81)
NAO_DT_STAR = 0.02 / np.sqrt(0.24801 / 9.81)


def _capture(path: Path, done: np.ndarray, seed: int = 0) -> Path:
    """A minimal rollout archive with the keys :func:`load_streams` reads."""
    steps, envs = done.shape
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        leg_length=0.65457,
        total_mass=32.2389,
        done=done,
        terminated=np.zeros_like(done),
        obs=rng.normal(size=(steps, envs, 123)).astype(np.float32),
        skill_state=np.zeros((steps, envs, 27), dtype=np.float32),
        command=rng.normal(size=(steps, envs, 3)).astype(np.float32),
        action_teacher=rng.normal(size=(steps, envs, 37)).astype(np.float32),
    )
    return path


# --- the clock ----------------------------------------------------------------


def test_the_grid_is_the_identity_on_the_body_it_was_built_for() -> None:
    grid = TimeGrid(step=G1_DT_STAR, samples=26)
    assert grid.history_steps(G1_DT_STAR) == 26
    np.testing.assert_allclose(grid.interpolation(G1_DT_STAR), np.eye(26))


def test_another_body_resamples_its_history_onto_the_same_instants() -> None:
    grid = TimeGrid(step=G1_DT_STAR, samples=26)
    matrix = grid.interpolation(NAO_DT_STAR)
    steps = grid.history_steps(NAO_DT_STAR)
    assert (steps - 1) * NAO_DT_STAR >= grid.span
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    # A signal linear in body time is reproduced exactly at the grid instants.
    history_time = (np.arange(steps) - (steps - 1)) * NAO_DT_STAR
    grid_time = (np.arange(26) - 25) * G1_DT_STAR
    signal = 3.0 * history_time - 0.5
    np.testing.assert_allclose(matrix @ signal, 3.0 * grid_time - 0.5, atol=1e-12)


def test_the_grid_spans_the_same_body_time_on_both_bodies() -> None:
    grid = TimeGrid(step=G1_DT_STAR, samples=26)
    for dt_star in (G1_DT_STAR, NAO_DT_STAR):
        matrix = grid.interpolation(dt_star)
        steps = grid.history_steps(dt_star)
        history_time = (np.arange(steps) - (steps - 1)) * dt_star
        np.testing.assert_allclose(matrix @ history_time, (np.arange(26) - 25) * G1_DT_STAR)


# --- the windows --------------------------------------------------------------


@pytest.fixture
def streams(tmp_path: Path):
    done = np.zeros((12, 2), dtype=bool)
    done[4, 0] = True  # env 0: episodes [0..4], [5..11]
    done[9, 1] = True  # env 1: episodes [0..9], [10..11]
    return load_streams([_capture(tmp_path / "a.npz", done)])


def test_columns_are_laid_end_to_end_with_their_episode_bounds(streams) -> None:
    assert len(streams) == 24
    np.testing.assert_array_equal(streams.start[:12].numpy(), [1] * 5 + [6] * 7)
    np.testing.assert_array_equal(streams.end[:12].numpy(), [4] * 5 + [11] * 7)
    np.testing.assert_array_equal(streams.start[12:].numpy(), [13] * 10 + [23] * 2)
    np.testing.assert_array_equal(streams.end[12:].numpy(), [21] * 10 + [23] * 2)


def test_no_window_crosses_a_reset_and_the_start_is_repeated(streams) -> None:
    windows = streams.windows(torch.tensor([7, 3]), length=4)
    np.testing.assert_array_equal(windows.numpy(), [[6, 6, 6, 7], [1, 1, 2, 3]])


def test_samples_exclude_reset_steps_and_keep_their_successors_in_the_episode(streams) -> None:
    samples = streams.samples(future=2)
    np.testing.assert_array_equal(samples.numpy(), [1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19])
    futures = streams.futures(samples, 2)
    assert bool((futures <= streams.end[samples][:, None]).all())
    assert bool((streams.start[futures] == streams.start[samples][:, None]).all())


def test_captures_are_concatenated_without_sharing_an_episode(tmp_path: Path) -> None:
    done = np.zeros((5, 1), dtype=bool)
    joined = load_streams(
        [_capture(tmp_path / "a.npz", done), _capture(tmp_path / "b.npz", done, seed=1)]
    )
    np.testing.assert_array_equal(joined.start.numpy(), [1] * 5 + [6] * 5)
    np.testing.assert_array_equal(joined.capture.numpy(), [0] * 5 + [1] * 5)
    windows = joined.windows(torch.tensor([6]), length=3)
    np.testing.assert_array_equal(windows.numpy(), [[6, 6, 6]])


# --- the probes ---------------------------------------------------------------


def test_the_probes_find_what_is_there_and_the_mlp_finds_more() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(6000, 4, generator=generator)
    targets = torch.stack([x @ torch.tensor([1.0, -2.0, 0.5, 0.0]), x[:, 0] ** 2], dim=1)
    steps = torch.arange(6000)
    train, test = steps[:5000], steps[5000:]

    def features(idx: torch.Tensor) -> torch.Tensor:
        return x[idx]

    linear = linear_probe(features, targets, train, test)
    nonlinear = mlp_probe(features, targets, train, test, epochs=40, hidden=64, batch=500)
    assert linear[0] > 0.999
    assert linear[1] < 0.05
    assert nonlinear[1] > 0.9
