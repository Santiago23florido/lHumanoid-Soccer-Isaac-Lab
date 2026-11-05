"""W7's leg-drive switch: faster drives, the damping ratio kept, off by default."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
from humanoid_transfer.nao.assets import nao_kinematics as nk

CFG = (
    Path(__file__).resolve().parents[1]
    / "source/humanoid_transfer/humanoid_transfer/nao/tasks/nao_track/nao_track_env_cfg.py"
)


@pytest.mark.parametrize("gain", [0.5, 2.0, 4.0])
def test_scaling_keeps_the_damping_ratio_and_shortens_the_time_constant(gain):
    stiffness, damping = (
        {"LKneePitch": 18.9, "LHipPitch": 11.0},
        {"LKneePitch": 2.5, "LHipPitch": 1.6},
    )
    k, d = nk.scaled_drive_gains(stiffness, damping, gain)
    for name in stiffness:
        for inertia in (0.00513, 0.39):  # the leg alone in swing, the whole body in stance
            before = damping[name] / (2 * math.sqrt(stiffness[name] * inertia))
            after = d[name] / (2 * math.sqrt(k[name] * inertia))
            assert after == pytest.approx(before, rel=1e-12)
        assert d[name] / k[name] == pytest.approx(damping[name] / stiffness[name] / math.sqrt(gain))


def test_gain_one_returns_the_derived_gains_and_zero_is_refused():
    stiffness, damping = {"LKneePitch": 18.9}, {"LKneePitch": 2.5}
    assert nk.scaled_drive_gains(stiffness, damping, 1.0) == (stiffness, damping)
    with pytest.raises(ValueError):
        nk.scaled_drive_gains(stiffness, damping, 0.0)


def test_the_task_reads_the_switch_off_by_default_and_scales_only_the_legs():
    text = CFG.read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "HUMANOID_NAO_LEG_GAIN"
    ]
    assert len(calls) == 1 and calls[0].args[1].value == "1"
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "LEG_GAIN != 1.0" in ast.get_source_segment(text, node.test)
    ]
    assert len(guards) == 1
    body = ast.get_source_segment(text, guards[0])
    assert 'actuators["legs"]' in body and "scaled_drive_gains" in body
