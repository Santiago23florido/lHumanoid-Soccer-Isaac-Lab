"""The teacher registry and the shell assignments every PACT script reads a body from."""

from __future__ import annotations

import math
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import body_conditions as bc  # noqa: E402


def test_every_teacher_body_is_a_registered_body_with_known_conditions():
    bodies, teachers = bc.registry(bc.BODIES), bc.registry(bc.TEACHERS)
    assert set(teachers) <= set(bodies)
    for body, entry in teachers.items():
        assert set(entry["conditions"]) <= set(bc.CONDITIONS), body
        assert "HUMANOID_CONTRACT" in entry["conditions"], body
        assert entry["variant"] and entry["teacher"], body
        assert (REPO / entry["mirror"]).exists(), body


def test_froude_scales_match_the_ladders_former_constants():
    # zero_shot_ladder.sh carried the NAO's scales as constants: sqrt(0.24801 / 0.65457...).
    g1, nao = bc.resolve("g1"), bc.resolve("nao")
    assert g1["speed_scale"] == pytest.approx(1.0) and g1["turn_scale"] == pytest.approx(1.0)
    expected = math.sqrt(0.24801 / 0.6545728842979989)
    assert nao["speed_scale"] == pytest.approx(expected, rel=1e-12)
    assert nao["turn_scale"] == pytest.approx(1.0 / expected, rel=1e-12)


def test_the_latest_teacher_run_and_checkpoint_are_chosen(tmp_path):
    runs = tmp_path / "rsl_rl"
    for stamp in ("2026-01-01_00-00-00", "2026-02-01_00-00-00"):
        run = runs / "h1_track" / f"{stamp}_contactprec_h1"
        run.mkdir(parents=True)
        for step in (99, 1000, 250):
            (run / f"model_{step}.pt").write_bytes(b"")
    (runs / "h1_track" / "2026-03-01_00-00-00_contactsched_h1").mkdir()
    r = bc.resolve("h1", runs=runs)
    assert r["run"].name == "2026-02-01_00-00-00_contactprec_h1"
    assert r["checkpoint"] == "model_1000"
    assert r["policy"].name == "model_1000_sym_pact.pt"
    assert r["plain_policy"].name == "model_1000.pt"


def test_shell_assignments_evaluate_in_bash(tmp_path):
    runs = tmp_path / "rsl_rl"
    run = runs / "nao_track" / "2026-01-01_00-00-00_contactprec_nao"
    (run / "exported").mkdir(parents=True)
    (run / "model_7.pt").write_bytes(b"")
    (run / "exported" / "model_7_sym_pact.pt").write_bytes(b"")
    text = bc.shell("nao", check=True, prefix="body_", runs=runs)
    script = text + "\nprintf '%s|%s|%s|%s' \"$HUMANOID_LOAD_SPACE\" \"$body_checkpoint\" " \
        "\"$body_play_task\" \"${HUMANOID_POSTURE_CONTRACT##*/}\""
    # The script goes in on stdin, as bytes: a multi-line -c argument does not survive Windows
    # quoting, and text mode would turn its newlines into CRLF.
    out = subprocess.run(["bash"], input=script.encode(), capture_output=True, check=True).stdout.decode()
    assert out == "linear|model_7|NaoTrack-Play-v0|generator.pt"
    # A body without a posture contract unsets it, so a previous body's does not leak.
    g1 = bc.shell("g1", runs=runs)
    assert "unset HUMANOID_POSTURE_CONTRACT" in g1.splitlines()
    assert all(shlex.split(line) for line in g1.splitlines())


def test_check_fails_without_a_teacher_policy(tmp_path):
    with pytest.raises(SystemExit):
        bc.shell("cassie", check=True, runs=tmp_path / "none")
