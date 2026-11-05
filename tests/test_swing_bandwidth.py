"""The swing-bandwidth bounds: the excursion ceiling and the planar leg.

``transfer/scripts/swing_bandwidth.py`` bounds how far a swing joint can move within a
planned swing. The bound is closed-form; these tests hold it to a brute-force search over
bang-bang profiles, with and without a velocity limit, and the planar leg to a case done
by hand.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transfer" / "scripts"))

swing_bandwidth = pytest.importorskip("swing_bandwidth")


def _brute_force_peak(alpha: float, velocity: float, duration: float) -> float:
    """Best peak over profiles: full effort up until a switch, then full effort down."""
    steps = 4000
    dt = duration / steps
    best = 0.0
    for switch in np.linspace(0.0, duration, 401):
        x = v = peak = 0.0
        for k in range(steps):
            a = alpha if k * dt < switch else -alpha
            v = float(np.clip(v + a * dt, -velocity, velocity))
            x += v * dt
            peak = max(peak, x)
        if x <= 1e-9:  # back by the end
            best = max(best, peak)
    return best


def test_ceiling_without_a_velocity_limit_is_closed_form():
    alpha, duration = 589.0, 0.08
    expected = (1 - 1 / math.sqrt(2)) ** 2 * alpha * duration**2
    got = swing_bandwidth.excursion_ceiling(alpha, math.inf, duration)
    # The split between rising and returning is searched on a 4001-point grid.
    assert got == pytest.approx(expected, rel=1e-3)
    assert got <= expected


@pytest.mark.parametrize("velocity", [math.inf, 6.4, 3.0])
def test_ceiling_matches_a_brute_force_search(velocity):
    alpha, duration = 589.0, 0.08
    bound = swing_bandwidth.excursion_ceiling(alpha, velocity, duration)
    search = _brute_force_peak(alpha, velocity, duration)
    assert search <= bound * 1.01
    assert search >= bound * 0.98


def test_a_velocity_limit_only_lowers_the_ceiling():
    free = swing_bandwidth.excursion_ceiling(589.0, math.inf, 0.08)
    capped = swing_bandwidth.excursion_ceiling(589.0, 6.4, 0.08)
    assert capped < free


def test_planar_leg_by_hand():
    # A straight leg hanging down; the knee bends by theta, the hip stays.
    thigh, shank = np.array([0.0, -0.1]), np.array([0.0, -0.1029])
    theta = 0.3
    rise = swing_bandwidth.ankle_rise(thigh, shank, 0.0, theta)
    assert rise == pytest.approx(0.1029 * (1 - math.cos(theta)), abs=1e-12)
    # Turning the whole leg at the hip by theta lifts both segments.
    rise = swing_bandwidth.ankle_rise(thigh, shank, theta, 0.0)
    assert rise == pytest.approx(0.2029 * (1 - math.cos(theta)), abs=1e-12)
