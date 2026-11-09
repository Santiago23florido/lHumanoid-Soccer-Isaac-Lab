"""Checks on contact-transition fidelity and on resampling across control rates."""

from __future__ import annotations

import math

import numpy as np
import pytest
from humanoid_transfer.common.contact import (
    LOAD_EPSILON,
    capture_fidelity,
    contact_fidelity,
    foot_height,
    resample,
)
from humanoid_transfer.common.skill_state import DEFAULT_CONTACT_THRESHOLD, SKILL_LAYOUT

LOADS = SKILL_LAYOUT.slice_of("foot_load")
FEET = SKILL_LAYOUT.slice_of("foot_position")
HEIGHT = SKILL_LAYOUT.slice_of("height")


def _gait(steps: int, period: int, stance: int, envs: int = 2, load: float = 0.5) -> np.ndarray:
    """Square-wave loads, the right foot half a period behind; unit height, feet on the ground."""
    z = np.zeros((steps, envs, SKILL_LAYOUT.size))
    z[..., HEIGHT] = 1.0
    z[..., FEET.start + 2] = -1.0
    z[..., FEET.start + 5] = -1.0
    t = np.arange(steps)
    z[..., LOADS.start] = load * ((t % period) < stance)[:, None]
    z[..., LOADS.start + 1] = load * (((t + period // 2) % period) < stance)[:, None]
    return z


def _duty(z: np.ndarray) -> np.ndarray:
    return (z[..., LOADS] > DEFAULT_CONTACT_THRESHOLD).mean(axis=(0, 1))


def test_linear_load_interpolation_lengthens_stance_by_its_analytic_amount() -> None:
    # Period 10 steps, stance 6: duty 0.6 on its own grid. Between a stance sample (0.5)
    # and a swing sample (0) a linearly interpolated load exceeds 5 % for 0.9 of the
    # interval, where nearest sampling counts half: 0.4 of a step more per boundary, two
    # boundaries a stride, so +0.08. In log units the crossing is at w = 0.545: +0.009.
    z = _gait(20000, 10, 6, envs=1)
    done = np.zeros(z.shape[:2], dtype=bool)
    assert _duty(z) == pytest.approx([0.6, 0.6])
    # The NAO's step in the G1's (0.1258 / 0.0774). A small fraction such as 13/8 would
    # sample only a lattice of phases of the period, not an average over them.
    rate = 1.62533
    crossing = (math.log(0.5 + LOAD_EPSILON) - math.log(0.05 + LOAD_EPSILON)) / (
        math.log(0.5 + LOAD_EPSILON) - math.log(LOAD_EPSILON)
    )
    expected = {
        "linear": 0.6 + 2 * 0.4 / 10,
        "log": 0.6 + 2 * (crossing - 0.5) / 10,
        "nearest": 0.6,
    }
    for mode, value in expected.items():
        resampled, _ = resample(z, done, 1.0, rate, loads=mode)
        assert _duty(resampled) == pytest.approx([value, value], abs=0.004), mode


def test_resampling_modes_differ_only_in_the_loads() -> None:
    rng = np.random.default_rng(0)
    z = _gait(200, 12, 7) + 0.01 * rng.standard_normal((200, 2, SKILL_LAYOUT.size))
    done = np.zeros(z.shape[:2], dtype=bool)
    done[90, 1] = True
    base, base_done = resample(z, done, 0.0774, 0.1258, loads="linear")
    other = [k for k in range(SKILL_LAYOUT.size) if not LOADS.start <= k < LOADS.stop]
    for mode in ("log", "nearest"):
        out, out_done = resample(z, done, 0.0774, 0.1258, loads=mode)
        np.testing.assert_array_equal(out[..., other], base[..., other])
        np.testing.assert_array_equal(out_done, base_done)
    assert base_done[:, 1].sum() == 1 and not base_done[:, 0].any()
    target = np.arange(0.0, 199 * 0.0774, 0.1258)
    np.testing.assert_allclose(
        base[:, 0, 0], np.interp(target, np.arange(200) * 0.0774, z[:, 0, 0])
    )


def test_a_body_that_does_what_it_is_asked_is_faithful() -> None:
    z = _gait(400, 20, 12)
    done = np.zeros(z.shape[:2], dtype=bool)
    report = contact_fidelity(z, done, 0.08, asked=z)
    assert report.agreement == (1.0, 1.0)
    assert report.graze_in_swing == (0.0, 0.0)
    assert report.liftoff_lag == (0.0, 0.0) and report.touchdown_lag == (0.0, 0.0)
    assert report.liftoff_matched == (1.0, 1.0)
    assert report.duty_body == pytest.approx((0.6, 0.6), abs=0.01)


def test_a_late_body_lags_by_its_delay() -> None:
    asked = _gait(400, 20, 12)
    body = np.roll(asked, 2, axis=0)
    done = np.zeros(asked.shape[:2], dtype=bool)
    report = contact_fidelity(body, done, 0.08, asked=asked)
    assert report.liftoff_lag == pytest.approx((0.16, 0.16))
    assert report.touchdown_lag == pytest.approx((0.16, 0.16))


def test_a_body_that_lifts_at_light_load_is_airborne_where_light_contact_is_asked() -> None:
    asked = _gait(400, 20, 12)
    t = np.arange(400)
    # The last two stance samples of each left stance carry a light 0.15: a load transfer.
    light = np.isin(t % 20, (10, 11))
    asked[light, :, LOADS.start] = 0.15
    body = asked.copy()
    body[light, :, LOADS.start] = 0.0  # this body lifts as soon as the load is light
    done = np.zeros(asked.shape[:2], dtype=bool)
    report = contact_fidelity(body, done, 0.08, asked=asked)
    assert report.light_share_asked[0] == pytest.approx(0.1, abs=0.01)
    assert report.light_share_body[0] == 0.0
    assert report.airborne_when_light == (1.0, 0.0)
    assert report.duty_body[0] == pytest.approx(report.duty_asked[0] - 0.1, abs=0.01)
    assert report.liftoff_lag[0] == pytest.approx(-2 * 0.08)


def test_clearance_is_the_swing_peak_above_the_stance_level() -> None:
    z = _gait(400, 20, 12)
    t = np.arange(400)
    swing = (t % 20) >= 12
    bump = np.where(swing, 0.05 * np.sin(np.pi * ((t % 20) - 11) / 9.0), 0.0)
    z[..., FEET.start + 2] += bump[:, None]
    done = np.zeros(z.shape[:2], dtype=bool)
    report = contact_fidelity(z, done, 0.08)
    assert report.clearance_body[0] == pytest.approx(0.05 * math.sin(math.pi * 4 / 9), abs=1e-3)
    np.testing.assert_allclose(foot_height(z)[~swing, 0, 0], 0.0, atol=1e-12)


def test_a_capture_scores_each_plan_against_the_state_after_it(tmp_path) -> None:
    z = _gait(300, 20, 12)
    due = np.roll(z, -1, axis=0)  # contract_due[t] is the target of the state at t + 1
    done = np.zeros(z.shape[:2], dtype=bool)
    done[150, 0] = True
    path = tmp_path / "capture.npz"
    np.savez(path, skill_state=z, contract_due=due, done=done)
    report = capture_fidelity(np.load(path), 0.08)
    assert report.agreement == (1.0, 1.0)
    assert report.liftoff_lag == (0.0, 0.0)
