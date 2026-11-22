"""R4's gap closure on synthetic result files laid out as the gates write them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import icml_r4_read as r4  # noqa: E402


def _write(root: Path, name: str, data: dict) -> None:
    (root / name).write_text(json.dumps(data), encoding="utf-8")


def test_closure_is_control_minus_held_over_control_minus_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(r4, "RESULTS", tmp_path)
    bodies = ["a", "b"]
    _write(tmp_path, "icml_final_gate_a3_s0.json", {"bodies": {
        b: {"physics": {"all": 1.0}, "pact_kd": {"all": 0.5}} for b in bodies}})  # fmt: skip
    _write(tmp_path, "icml_final_gate_x_s0.json", {"bodies": {
        b: {"copy": 0.8, "pact_x": 0.4} for b in bodies}})  # fmt: skip
    for b, enc, tr in (("a", 0.75, 0.6), ("b", 0.5, 0.8)):
        _write(tmp_path, f"icml_r4_encoder_{b}_s0.json",
               {"budgets": {"1": {"arms": {"zero": {"rmse": {"all": enc}}}}}})  # fmt: skip
        _write(tmp_path, f"icml_r4_translator_{b}_s0.json",
               {"budgets": {"1": {"arms": {"zero": {"rmse": tr}}}}})  # fmt: skip
    report = r4.read(bodies, [0])
    enc, tr = report["parts"]["encoder"], report["parts"]["translator"]
    assert enc["bodies"]["a"][0]["closure"] == 0.5 and enc["bodies"]["b"][0]["closure"] == 1.0
    assert abs(enc["mean_closure"] - 0.75) < 1e-12 and enc["passes"]
    assert abs(tr["mean_closure"] - 0.25) < 1e-12 and not tr["passes"]  # (0.5 + 0.0) / 2
    assert report["zero_shot_body_claim"] is False


def test_missing_files_leave_the_claim_unread(tmp_path, monkeypatch):
    monkeypatch.setattr(r4, "RESULTS", tmp_path)
    report = r4.read(["a"], [0])
    assert report["zero_shot_body_claim"] is None
