"""A decoder's base tasks (``latent/task_set.py``, ``transfer/scripts/base_task_set.py``).

The command that consumes them needs the simulator; what is tested here is what it is
handed: each body's form of the same task, the held-out task refused at every step, and
the ranges' units.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from humanoid_transfer.transfer.latent.task_set import HELD_OUT, task_specs, task_weights

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, tasks: list[dict], held_out: str = "rough") -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"held_out": held_out, "tasks": tasks}), encoding="utf-8")
    return path


def _task(name: str = "flat") -> dict:
    return {
        "name": name,
        "contract": f"logs/contract/{name}/generator.pt",
        "dart": f"g1/rollouts/{name}_dart.npz",
        "ranges": {
            "lin_vel_x": [0.0, 0.4],
            "lin_vel_y": [-0.2, 0.2],
            "ang_vel_z": [-0.26, 0.26],
            "heading": [-math.pi, math.pi],
        },
    }


def test_the_g1_starts_from_states_and_the_nao_from_windows(tmp_path):
    path = _write(tmp_path, [_task()])
    g1 = task_specs(path, tmp_path, own_states=True)[0]
    nao = task_specs(path, tmp_path, own_states=False)[0]
    assert g1["starts"] == nao["windows"] == str(tmp_path / "g1/rollouts/flat_dart.npz")
    assert "windows" not in g1 and "starts" not in nao
    assert g1["contract"] == nao["contract"] == str(tmp_path / "logs/contract/flat/generator.pt")
    assert g1["ranges"] == nao["ranges"]
    assert g1["ranges"]["lin_vel_x"] == (0.0, 0.4)


def test_the_held_out_task_is_refused(tmp_path):
    with pytest.raises(ValueError, match="held out"):
        task_specs(_write(tmp_path, [_task()], held_out="flat"), tmp_path, own_states=True)
    for field in ("name", "contract", "dart"):
        entry = _task()
        entry[field] = entry[field].replace("flat", HELD_OUT)
        with pytest.raises(ValueError, match="held-out"):
            task_specs(_write(tmp_path, [entry]), tmp_path, own_states=True)


def test_task_weights():
    assert task_weights("") is None
    assert task_weights(" ") is None
    assert task_weights("2,1,1") == [2.0, 1.0, 1.0]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "base_task_set", ROOT / "transfer/scripts/base_task_set.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_capture_must_lie_inside_its_declared_ranges(tmp_path):
    builder = _builder()
    ranges = {"lin_vel_x": [0.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-1.0, 1.0]}
    inside = np.array([[[0.0, 0.0, 0.0], [1.0, -0.5, 1.0]]])
    np.savez(tmp_path / "in.npz", command=inside)
    assert builder.capture_check(tmp_path / "in.npz", ranges)["inside"]
    np.savez(tmp_path / "out.npz", command=inside + np.array([0.0, 0.0, 0.1]))
    assert not builder.capture_check(tmp_path / "out.npz", ranges)["inside"]


def test_flat_ranges_are_isaac_labs():
    builder = _builder()
    if not builder.FLAT_CFG.exists():
        pytest.skip("Isaac Lab's source is not at C:/IsaacLab")
    ranges = builder.flat_ranges()
    assert ranges["lin_vel_x"] == [0.0, 1.0]
    assert ranges["lin_vel_y"] == [-0.5, 0.5]
    assert ranges["ang_vel_z"] == [-1.0, 1.0]
    assert ranges["heading"] == pytest.approx([-math.pi, math.pi])


def test_the_recorded_task_set_holds_no_rough_and_is_dimensionless():
    path = ROOT / "transfer/results/base_tasks.json"
    if not path.exists():
        pytest.skip("no task set recorded yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["held_out"] == HELD_OUT
    velocity, time = report["units_g1"]["velocity"], report["units_g1"]["time"]
    for task in report["tasks"]:
        assert HELD_OUT not in json.dumps(task)
        physical, ranges = task["ranges_physical_g1"], task["ranges"]
        for key in ("lin_vel_x", "lin_vel_y"):
            assert ranges[key] == pytest.approx([v / velocity for v in physical[key]])
        assert ranges["ang_vel_z"] == pytest.approx([v * time for v in physical["ang_vel_z"]])
        assert task["capture_commands"]["inside"]
