"""The robot parameters a teacher reads: normalisation and the per-robot drive draws."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.sim2real.params import PARAMS, draw_factors, normalise  # noqa: E402


def test_the_ranges_map_to_minus_one_and_one_and_clip_beyond() -> None:
    ranges = [(10.0, 60.0), (0.7, 1.3), (0.7, 1.3), (0.7, 1.1), (0.5, 1.5)]
    low = torch.tensor([[r[0] for r in ranges]])
    high = torch.tensor([[r[1] for r in ranges]])
    middle = (low + high) / 2
    assert torch.allclose(normalise(low, ranges), -torch.ones(1, len(PARAMS)))
    assert torch.allclose(normalise(high, ranges), torch.ones(1, len(PARAMS)))
    assert torch.allclose(normalise(middle, ranges), torch.zeros(1, len(PARAMS)), atol=1e-6)
    assert torch.allclose(normalise(low - 1.0, ranges), -torch.ones(1, len(PARAMS)))


def test_a_robots_factor_scatters_its_joints_within_the_spread() -> None:
    torch.manual_seed(0)
    per_joint, own = draw_factors(4096, 12, (0.7, 1.1), 0.1, "cpu")
    assert own.min() >= 0.7 and own.max() <= 1.1
    ratio = per_joint / own[:, None]
    assert ratio.min() >= 0.9 - 1e-6 and ratio.max() <= 1.1 + 1e-6
    independent, mean = draw_factors(4096, 12, (0.7, 1.1), None, "cpu")
    assert torch.allclose(mean, independent.mean(-1))
    # The point of the shared factor: a robot's joints vary together.
    assert per_joint.mean(-1).std() > 3 * independent.mean(-1).std()
