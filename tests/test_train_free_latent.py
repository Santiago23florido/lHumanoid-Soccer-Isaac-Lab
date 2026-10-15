"""The free latent's training script runs end to end on synthetic captures of both bodies."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "transfer" / "scripts" / "train_free_latent.py"


def _capture(path: Path, names: list[str], actions: int, seed: int, dt: float, length: float):
    rng = np.random.default_rng(seed)
    steps, envs, joints = 60, 6, len(names)
    terminated = np.zeros((steps, envs), dtype=bool)
    terminated[40, 1] = True
    np.savez(
        path,
        joint_names=np.array(names),
        action_executed=rng.normal(size=(steps, envs, actions)).astype(np.float32),
        encoder_input=rng.normal(size=(steps, envs, 9 + 2 * joints + actions)).astype(np.float32),
        skill_state=rng.normal(size=(steps, envs, 27)).astype(np.float32),
        balance=rng.normal(size=(steps, envs, 10)).astype(np.float32),
        biped_phase=rng.uniform(size=(steps, envs)).astype(np.float32),
        probe_index=np.arange(envs) % 3,
        done=terminated.copy(),
        terminated=terminated,
        leg_length=np.float64(length),
        total_mass=np.float64(30.0),
        control_dt=np.float32(dt),
    )


def test_the_script_trains_and_reports_the_cross_body_test(tmp_path: Path, monkeypatch) -> None:
    from humanoid_transfer.nao.assets import nao_kinematics as nk

    g1_names = [f"{side}_{part}_joint" for side in ("left", "right")
                for part in ("hip_pitch", "hip_roll", "knee", "ankle_pitch")]  # fmt: skip
    g1_names += ["torso_joint"]
    nao_names = list(nk.ACTUATED_JOINTS) + ["RHipYawPitch", "LFinger11"]
    _capture(tmp_path / "g1.npz", g1_names, len(g1_names), 0, 0.02, 0.65)
    _capture(tmp_path / "nao.npz", nao_names, len(nk.ACTUATED_JOINTS), 1, 0.02, 0.25)
    contract = tmp_path / "contract.pt"
    _contract(contract)
    results = tmp_path / "report.json"
    argv = [
        str(SCRIPT), "--g1", str(tmp_path / "g1.npz"), "--nao", str(tmp_path / "nao.npz"),
        "--contract", str(contract), "--steps", "20", "--batch", "32", "--device", "cpu",
        "--phase-bins", "4", "--held-out", "0.34",
        "--output", str(tmp_path / "model.pt"), "--results", str(results),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    spec = importlib.util.spec_from_file_location("train_free_latent", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "train_free_latent", module)
    spec.loader.exec_module(module)
    assert module.main() == 0
    report = json.loads(results.read_text(encoding="utf-8"))
    assert set(report["passes"]) == {"shared", "informative", "carried"}
    assert set(report["informative"]) == {"g1", "nao"}
    assert set(report["carried"]) == {"g1_zeta", "nao_zeta"}
    assert (tmp_path / "model.pt").is_file()


def _contract(path: Path) -> None:
    from humanoid_transfer.transfer.latent.contract import (
        Contract,
        ContractClock,
        ContractDims,
        ContractGenerator,
        save_contract,
    )

    clock = ContractClock(step=0.0774, history=26, horizon=20, commit=13, view=8)
    dims = ContractDims(memory=8, channels=8, flow_hidden=32, flow_steps=2)
    generator = ContractGenerator(dims, clock)
    save_contract(path, Contract(generator, None, torch.ones(27)), dims, clock, encoder_input=120)
