"""Regression checks for honest baseline measurements, without Isaac Sim."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stand_metrics import Recorder, minimum_margin, validate_run  # noqa: E402


def test_worst_margin_is_not_always_the_largest_absolute_excursion():
    # The forward excursion is larger, but the smaller backward excursion has
    # already crossed the nearer heel. Selecting max(abs(x)) misses that fall.
    assert minimum_margin([0.08, -0.07], -0.0607, 0.1033) == pytest.approx(-0.0093)


@pytest.mark.parametrize(
    "parameters",
    [
        (0, 0, 0, 0),
        (1, 1, 0, 0),
        (1, -1, 0, 0),
        (1, 0.5, 0.3, 1),
        (1, 0.5, 0.3, 0.1),
        (float("nan"), 0, 0, 0),
        (1, 0, float("inf"), 0.5),
    ],
)
def test_invalid_or_unmeasurable_runs_are_rejected(parameters):
    with pytest.raises(ValueError):
        validate_run(*parameters)


def test_backward_push_is_a_valid_experiment():
    validate_run(3.0, 0.5, -0.3, 1.0)


def test_export_preserves_missing_cop_and_a_fall_before_the_first_sample(tmp_path):
    recorder = Recorder()
    recorder.completed = True
    recorder.fell_at = 0.2
    recorder.recovered_at = 0.4
    recorder.time = [0.5, 0.505]
    recorder.dcm_offset = [0.08, -0.07]
    recorder.cop_offset = [0.02, None]
    path = tmp_path / "measurements.json"
    recorder.export(path, {"settle_time_s": 0.5}, (-0.0607, 0.1033))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert not data["upright"]
    assert data["fell_at_s"] == 0.2
    assert data["samples"]["cop_x_quasistatic_m"] == [0.02, None]
    assert data["minimum_dcm_margin_m"] < 0


def test_an_interrupted_run_cannot_report_success(tmp_path):
    recorder = Recorder()
    recorder.time = [0.5]
    path = tmp_path / "interrupted.json"
    recorder.export(path, {}, (-0.0607, 0.1033))
    assert not json.loads(path.read_text(encoding="utf-8"))["upright"]
