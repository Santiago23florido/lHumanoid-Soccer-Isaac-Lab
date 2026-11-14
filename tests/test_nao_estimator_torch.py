"""The simulator's torch estimator against the robot's NumPy one, on random sequences."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

torch = pytest.importorskip("torch")

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.deploy.estimator import Estimator  # noqa: E402
from humanoid_transfer.nao.deploy.kinematics import BodyModel  # noqa: E402
from humanoid_transfer.nao.sim2real.body_model import body_model  # noqa: E402
from humanoid_transfer.nao.sim2real.estimator_torch import TorchEstimator  # noqa: E402

SOLE = [[0.0, 0.0, -0.04511]] * 2
CORNERS = [
    [[x, y, -0.04511] for x in (-0.05, 0.1) for y in (-0.04, 0.03)],
    [[x, y, -0.04511] for x in (-0.05, 0.1) for y in (-0.03, 0.04)],
]


@pytest.fixture(scope="module")
def model() -> BodyModel:
    urdf = nk.load_model()
    moving = [j for j in urdf.joints if j.type != "fixed"]
    order = np.random.default_rng(5).permutation(len(moving))
    bodies = ["base_link"] + [moving[i].child for i in order]
    joints = [moving[i].name for i in order]
    return BodyModel(body_model(bodies, joints, SOLE, CORNERS, 0.24801))


@pytest.mark.parametrize("pivot", ["sole", "corner"])
def test_torch_estimator_matches_the_robot_s(model: BodyModel, pivot: str) -> None:
    rng = np.random.default_rng(0)
    count, steps, dt = 6, 25, 0.02
    ours = Estimator(model, count, dt, pivot)
    theirs = TorchEstimator(model, count, dt, "cpu", pivot, dtype=torch.float64)
    q = rng.uniform(-0.4, 0.4, size=(count, len(model.joint_names)))
    for k in range(steps):
        q = q + rng.normal(scale=0.02, size=q.shape)
        tilt = rng.normal(scale=0.1, size=(count, 2))
        gravity = np.stack(
            [
                np.sin(tilt[:, 1]),
                -np.sin(tilt[:, 0]) * np.cos(tilt[:, 1]),
                -np.cos(tilt[:, 0]) * np.cos(tilt[:, 1]),
            ],
            -1,
        )
        gyro = rng.normal(scale=0.5, size=(count, 3))
        loads = rng.uniform(0.0, 1.0, size=(count, 2))
        if k == 10:
            reset = rng.uniform(size=count) < 0.5
            ours.reset(reset)
            theirs.reset(torch.as_tensor(reset))
        z = ours.update(q, gravity, gyro, loads)
        zt = theirs.update(*(torch.as_tensor(a) for a in (q, gravity, gyro, loads))).numpy()
        np.testing.assert_allclose(zt, z, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(theirs.position.numpy(), ours.position, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(theirs.yaw.numpy(), ours.yaw, rtol=1e-9, atol=1e-9)
