"""The NAO's sole point: the URDF's frame by default, its resting contact on request.

``z*`` places each foot at its sole point and every plan puts a stance sole at height 0.
The NAO's URDF sole frame rests 2.494 mm above the ground (``transfer/results/
sole_offset.json``); ``HUMANOID_SOLE=contact`` lowers it there. The default must leave
every existing tracker's ``z*`` exactly as it was.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.assets import nao_paths  # noqa: E402

CFG = REPO / "source/humanoid_transfer/humanoid_transfer/nao/tasks/nao_track/nao_track_env_cfg.py"

requires_urdf = pytest.mark.skipif(
    not nao_paths.NAO_URDF_PATH.exists(), reason="NAO URDF not present (robot assets are local)"
)


@requires_urdf
def test_the_default_is_the_urdf_sole_frame_bit_for_bit() -> None:
    poses = nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)
    before = [
        list(map(float, (np.linalg.inv(poses[foot]) @ poses[sole])[:3, 3]))
        for foot, sole in zip(nk.FOOT_BODIES, nk.SOLE_FRAMES, strict=True)
    ]
    assert nk.sole_points_in_foot() == before


@requires_urdf
def test_contact_lowers_each_sole_point_along_the_foot_only() -> None:
    frame, contact = nk.sole_points_in_foot(), nk.sole_points_in_foot(nk.SOLE_REST)
    for a, b in zip(frame, contact, strict=True):
        assert b[:2] == a[:2]
        assert b[2] == pytest.approx(a[2] - nk.SOLE_REST, abs=1e-12)


def test_the_rest_offset_is_the_measured_one() -> None:
    result = REPO / "transfer" / "results" / "sole_offset.json"
    if not result.exists():
        pytest.skip("sole_offset.json not produced")
    report = json.loads(result.read_text(encoding="utf-8"))
    leg = next(c["leg_length"] for p, c in report["captures"].items() if p.startswith("nao/"))
    measured = report["flat_foot"]["nao"]["p10_median"] * leg
    assert nk.SOLE_REST == pytest.approx(measured, abs=1e-6)
    assert abs(report["flat_foot"]["g1"]["p10_median"]) < 1e-4


def test_the_task_reads_the_switch_with_the_urdf_as_default() -> None:
    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "HUMANOID_SOLE"
    ]
    assert len(calls) == 1 and calls[0].args[1].value == "urdf"
    source = ast.get_source_segment(
        CFG.read_text(encoding="utf-8"),
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sole_points"),
    )
    assert "SOLE_REST" in source and 'SOLE == "contact"' in source
