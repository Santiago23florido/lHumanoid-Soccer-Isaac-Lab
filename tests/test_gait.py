"""Checks on the gait signature, on synthetic gaits whose numbers are known."""

from __future__ import annotations

import numpy as np
import pytest
from humanoid_transfer.common.gait import gait_signature, stride_periods, touchdowns
from humanoid_transfer.common.skill_state import SKILL_LAYOUT

PERIOD = 40
DUTY = 0.6
STEPS = 400


def _square_wave(steps: int, shift: int) -> np.ndarray:
    phase = (np.arange(steps) + shift) % PERIOD
    return phase < DUTY * PERIOD


def _walk(envs: int = 2, speed: float = 0.1) -> np.ndarray:
    """A gait in skill-state form: alternating feet, constant forward speed, unit height."""
    z = np.zeros((STEPS, envs, SKILL_LAYOUT.size))
    z[..., SKILL_LAYOUT.slice_of("height")] = 1.0
    z[..., SKILL_LAYOUT.slice_of("com_velocity").start] = speed
    gravity = SKILL_LAYOUT.slice_of("gravity")
    z[..., gravity.start + 2] = -1.0
    feet = SKILL_LAYOUT.slice_of("foot_position")
    z[..., feet.start : feet.start + 3] = [0.2, 0.15, -1.0]
    z[..., feet.start + 3 : feet.stop] = [0.2, -0.15, -1.0]
    loads = SKILL_LAYOUT.slice_of("foot_load")
    left, right = _square_wave(STEPS, 0), _square_wave(STEPS, PERIOD // 2)
    z[..., loads.start] = 0.5 * left[:, None]
    z[..., loads.start + 1] = 0.5 * right[:, None]
    return z


def test_a_regular_walk_has_the_period_duty_and_stride_it_was_built_with() -> None:
    dt_star = 0.08
    done = np.zeros((STEPS, 2), dtype=bool)
    sig = gait_signature(_walk(), done, dt_star)
    assert sig.stride_period == pytest.approx(PERIOD * dt_star)
    assert sig.duty_factor == pytest.approx((DUTY, DUTY))
    assert sig.speed == pytest.approx(0.1)
    assert sig.froude == pytest.approx(0.01)
    assert sig.stride_length == pytest.approx(0.1 * PERIOD * dt_star)


def test_the_dcm_offset_at_touchdown_is_measured_from_the_landing_foot() -> None:
    """Height 1 makes the DCM lead the CoM by exactly v*; the foot is 0.2 ahead."""
    sig = gait_signature(_walk(), np.zeros((STEPS, 2), dtype=bool), 0.08)
    left, right = sig.dcm_at_touchdown
    assert left == pytest.approx((0.1 - 0.2, -0.15))
    assert right == pytest.approx((0.1 - 0.2, 0.15))


def test_a_mirror_symmetric_walk_has_no_asymmetry_and_a_limp_does() -> None:
    sym = gait_signature(_walk(), np.zeros((STEPS, 2), dtype=bool), 0.08)
    limp = _walk()
    limp[..., SKILL_LAYOUT.slice_of("foot_load").start] *= 1.5
    lopsided = gait_signature(limp, np.zeros((STEPS, 2), dtype=bool), 0.08)
    assert sym.asymmetry == pytest.approx(0.0, abs=1e-12)
    assert lopsided.asymmetry > 0.01


def test_no_event_or_interval_crosses_a_reset() -> None:
    """A reset mid-stance: the contact that follows is not a landing, and no stride spans it."""
    contact = np.zeros((10, 1, 2), dtype=bool)
    contact[[2, 3, 6, 7, 8], 0, 0] = True
    done = np.zeros((10, 1), dtype=bool)
    landing = touchdowns(contact, done)
    assert np.flatnonzero(landing[:, 0, 0]).tolist() == [2, 6]
    assert stride_periods(landing, done).tolist() == [4.0]

    done[5, 0] = True  # reset after step 5: step 6 starts a new episode
    landing = touchdowns(contact, done)
    assert np.flatnonzero(landing[:, 0, 0]).tolist() == [2]
    assert stride_periods(landing, done).size == 0
