"""The per-swing pairing of lift and touchdown lag (``transfer/scripts/touchdown_lift.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from humanoid_transfer.common.skill_state import SKILL_LAYOUT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transfer" / "scripts"))

touchdown_lift = pytest.importorskip("touchdown_lift")

LOADS = SKILL_LAYOUT.slice_of("foot_load")
FEET = SKILL_LAYOUT.slice_of("foot_position")
HEIGHT = SKILL_LAYOUT.slice_of("height")


def _plan(steps: int) -> np.ndarray:
    """Period 10: the left foot in stance for 6 steps, the right foot always loaded."""
    z = np.zeros((steps, 1, SKILL_LAYOUT.size))
    z[..., HEIGHT] = 1.0
    z[..., FEET.start + 2] = -1.0
    z[..., FEET.start + 5] = -1.0
    t = np.arange(steps)
    z[:, 0, LOADS.start] = 0.5 * ((t % 10) < 6)
    z[:, 0, LOADS.start + 1] = 0.5
    return z


def test_a_low_swing_landing_early_is_paired_with_its_planned_touchdown():
    asked = _plan(40)
    body = asked.copy()
    # The second swing (planned steps 16-19) lands one step early and lifts only 0.01 L;
    # the first and third lift 0.05 L and land on time.
    body[19, 0, LOADS.start] = 0.5
    for start, stop, peak in ((6, 10, 0.05), (16, 19, 0.01), (26, 30, 0.05)):
        for k in range(start, stop):
            body[k, 0, FEET.start + 2] = -1.0 + peak * (1 if k in (start + 1, start + 2) else 0.5)
    valid = np.ones((40, 1), dtype=bool)
    rows = touchdown_lift.swing_touchdowns(body, asked, valid, dt_star=0.1)
    assert rows.shape == (3, 4)
    peak, early, lag, length = rows.T
    assert peak == pytest.approx([0.05, 0.01, 0.05])
    assert early == pytest.approx([0.05, 0.01, 0.05])
    assert lag == pytest.approx([0.0, -0.1, 0.0])
    assert length == pytest.approx([0.4, 0.3, 0.4])
    summary = touchdown_lift.summarize(rows)
    assert summary["peak"]["spearman_with_lag"] > 0


def test_a_swing_across_an_invalid_sample_is_left_out():
    asked = _plan(40)
    body = asked.copy()
    valid = np.ones((40, 1), dtype=bool)
    valid[8, 0] = False  # an episode boundary inside the first swing
    rows = touchdown_lift.swing_touchdowns(body, asked, valid, dt_star=0.1)
    assert len(rows) == 2
