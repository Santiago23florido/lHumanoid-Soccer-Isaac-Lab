"""Checks on reading the contract between the generator's grid and a body's control steps."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)
from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    ContractClock,
    ContractDims,
    ContractGenerator,
)
from humanoid_transfer.transfer.latent.memory import TimeGrid  # noqa: E402
from humanoid_transfer.transfer.latent.resampling import (  # noqa: E402
    is_selection,
    plan_interpolation,
    resample_rows,
)

REPO = Path(__file__).resolve().parents[1]
LOADS = SKILL_LAYOUT.slice_of("foot_load")
G1_STEP, NAO_STEP = 0.07742576129865368, 0.12584
CLOCK = ContractClock(step=G1_STEP, history=26, horizon=20, commit=13, view=8)
SMALL = ContractDims(memory=4, intent=2, hidden=32, channels=8, flow_hidden=64, warp_loads=True)


def _generator() -> ContractGenerator:
    """A generator whose normalisation is fitted on a stepping gait (loads 0.5 or 0)."""
    torch.manual_seed(0)
    generator = ContractGenerator(SMALL, CLOCK)
    skill = torch.randn(4000, SKILL_LAYOUT.size)
    t = torch.arange(4000)
    skill[:, LOADS.start] = 0.5 * ((t % 13) < 7).float()
    skill[:, LOADS.start + 1] = 0.5 * (((t + 6) % 13) < 7).float()
    generator.skill_norm.fit(skill)
    generator.fit_warp(skill)
    return generator


def _stepping(steps: int) -> torch.Tensor:
    """``(1, steps, 27)`` normalised: the left foot loaded 0.5 for 7 samples in 13, then 0."""
    generator = _generator()
    z = torch.zeros(steps, SKILL_LAYOUT.size)
    z[:, LOADS.start] = 0.5 * ((torch.arange(steps) % 13) < 7).float()
    return generator.skill_norm(z)[None]


def test_plan_interpolation_moved_without_change() -> None:
    g1 = plan_interpolation(G1_STEP, 20, G1_STEP)
    assert g1.shape == (20, 21) and is_selection(g1)
    torch.testing.assert_close(g1[:, 1:], torch.eye(20))
    nao = plan_interpolation(G1_STEP, 20, NAO_STEP)
    assert nao.shape == (12, 21) and not is_selection(nao)
    torch.testing.assert_close(nao.sum(dim=-1), torch.ones(12))


def test_a_body_on_the_grid_is_bit_identical_in_either_space() -> None:
    generator = _generator()
    rng = torch.Generator().manual_seed(1)
    sequence = torch.randn(5, 21, SKILL_LAYOUT.size, generator=rng)
    matrix = plan_interpolation(G1_STEP, 20, G1_STEP)
    assert torch.equal(
        resample_rows(generator, matrix, sequence, "linear"),
        resample_rows(generator, matrix, sequence, "generator"),
    )
    grid = TimeGrid(step=G1_STEP, samples=26)
    history = torch.tensor(grid.interpolation(G1_STEP), dtype=torch.float32)
    window = torch.randn(5, history.shape[1], SKILL_LAYOUT.size, generator=rng)
    assert torch.equal(
        resample_rows(generator, history, window, "linear"),
        resample_rows(generator, history, window, "generator"),
    )


def test_generator_space_keeps_the_stance_a_linear_blend_lengthens() -> None:
    # A stepping load read at the NAO's instants: blending in linear units counts nearly
    # every boundary sample as contact; in the generator's (log) space about half of them.
    # Eight NAO steps are 13.002 G1 steps, one stride of this gait: from a single start the
    # NAO's instants revisit a few phases only, so each run starts at a random offset.
    generator = _generator()
    source = _stepping(13 * 60)
    plain = generator.skill_norm.inverse(source)[..., LOADS.start]
    grid_duty = float((plain > DEFAULT_CONTACT_THRESHOLD).float().mean())
    rng = np.random.default_rng(0)
    contact = {"linear": [], "generator": []}
    for offset in rng.uniform(0.0, 13.0, size=64):
        times = torch.arange(offset, source.shape[1] - 2.0, NAO_STEP / G1_STEP, dtype=torch.float64)
        low = times.floor().long()
        weight = (times - low).float()
        matrix = torch.zeros(len(times), source.shape[1])
        matrix[torch.arange(len(times)), low] = 1.0 - weight
        matrix[torch.arange(len(times)), low + 1] = weight
        for space in contact:
            rows = generator.skill_norm.inverse(resample_rows(generator, matrix, source, space))
            contact[space].append(rows[..., LOADS.start] > DEFAULT_CONTACT_THRESHOLD)
    duty = {space: float(torch.cat(c, dim=1).float().mean()) for space, c in contact.items()}
    assert grid_duty == pytest.approx(7 / 13, abs=1e-3)
    assert duty["linear"] - grid_duty == pytest.approx(2 * 0.4 / 13, abs=0.01)
    assert abs(duty["generator"] - grid_duty) < 0.012


def test_generator_space_maps_back_to_non_negative_loads() -> None:
    generator = _generator()
    sequence = _stepping(21)
    matrix = plan_interpolation(G1_STEP, 20, NAO_STEP)
    rows = generator.skill_norm.inverse(resample_rows(generator, matrix, sequence, "generator"))
    assert (rows[..., LOADS] >= 0.0).all()
    other = [k for k in range(SKILL_LAYOUT.size) if not LOADS.start <= k < LOADS.stop]
    linear = resample_rows(generator, matrix, sequence, "linear")
    np.testing.assert_allclose(
        generator.skill_norm(rows)[..., other].numpy(), linear[..., other].numpy(), atol=1e-5
    )


def test_unknown_space_is_refused() -> None:
    with pytest.raises(ValueError):
        resample_rows(_generator(), torch.eye(3), torch.zeros(1, 3, SKILL_LAYOUT.size), "cubic")


@pytest.mark.parametrize(
    "path",
    [
        "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_track/g1_track_env_cfg.py",
        "source/humanoid_transfer/humanoid_transfer/nao/tasks/nao_track/nao_track_env_cfg.py",
    ],
)
def test_both_tracking_tasks_pass_the_resample_space(path: str) -> None:
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    keywords = {
        k.arg: k.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for k in node.keywords
        if k.arg == "resample_space"
    }
    assert isinstance(keywords.get("resample_space"), ast.Name)
    assert keywords["resample_space"].id == "RESAMPLE_SPACE"
