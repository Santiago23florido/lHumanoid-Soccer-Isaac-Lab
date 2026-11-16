"""The drive fit recovers a known drive from a synthetic harness record."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

pytest.importorskip("scipy")
from humanoid_transfer.nao.sim2real.actuator_fit import (  # noqa: E402
    Drive,
    Plant,
    fit_joint,
    simulate,
)


def test_fit_recovers_a_known_drive() -> None:
    plant = Plant(inertia=0.004, gravity=-0.3, gravity_slope=-0.5, reference=0.2)
    truth = Drive(
        stiffness=15.0, damping=0.6, delay=0.036, armature=0.005, ceiling=2.5, friction=0.05
    )
    cycle = 0.012
    time = np.arange(0.0, 3.0, cycle)
    steps = 0.2 + 0.15 * np.array([0.0, 1.0, 0.0, -1.0])[(time // 0.4).astype(int) % 4]
    chirp = 0.2 + 0.1 * np.sin(2 * np.pi * (0.5 * time + 0.8 * time * time))
    rng = np.random.default_rng(0)
    records = []
    for target in (steps, chirp):
        q = simulate(plant, truth, time, target, 0.2)
        q = q + rng.normal(scale=2e-4, size=q.shape)
        records.append({"time": time, "target": target, "position": q})
    start = Drive(
        stiffness=10.0, damping=1.0, delay=0.02, armature=0.002, ceiling=3.0, friction=0.02
    )
    drive, rms = fit_joint(plant, records, start, iterations=200)
    assert rms < 2e-3
    assert drive.stiffness == pytest.approx(truth.stiffness, rel=0.15)
    assert drive.delay == pytest.approx(truth.delay, abs=0.008)
