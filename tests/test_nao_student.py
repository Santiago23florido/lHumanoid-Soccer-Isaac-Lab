"""The student on the robot: the teacher's actor acting on estimated or measured parameters.

Checks, on synthetic weights, what ``nao/scripts/check_runtime.py`` cannot isolate:
that the NumPy student writes its estimate (or given values) into the teacher's
``robot_params`` inputs and nothing else, that its TorchScript twin
(``nao/scripts/student_bundle.py``) computes the same, and that the controller
normalises measured values over the bundle's ranges, keeping the estimate for any
parameter not given.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(ROOT / "nao" / "scripts"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.deploy.controller import (  # noqa: E402
    Bundle,
    Controller,
    history_base,
    normalise_params,
)
from humanoid_transfer.nao.deploy.policy import Actor  # noqa: E402
from humanoid_transfer.nao.sim2real.body_model import body_model  # noqa: E402

VIEW, SKILL, JOINTS, PARAMS, LENGTH = 4, 27, len(nk.ACTUATED_JOINTS), 5, 4
TEACHER_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", JOINTS),
    ("joint_vel", JOINTS),
    ("actions", JOINTS),
    ("contract", VIEW * SKILL + SKILL),
    ("contract_path", VIEW * 3),
    ("robot_params", PARAMS),
]
HISTORY_TERMS = [
    ("base_ang_vel_adapt_history", 3 * LENGTH),
    ("actions_adapt_history", JOINTS * LENGTH),
]
RANGES = [[10.0, 60.0], [0.7, 1.3], [0.7, 1.3], [0.7, 1.1], [0.5, 1.5]]
NAMES = ["delay_ms", "stiffness_scale", "damping_scale", "effort_scale", "armature_scale"]


def _arrays(rng: np.random.Generator, mirror: bool) -> dict[str, np.ndarray]:
    width = sum(w for _, w in TEACHER_TERMS)
    history = sum(w for _, w in HISTORY_TERMS)
    arrays = {
        "mean": rng.normal(scale=0.1, size=width),
        "scale": rng.uniform(0.5, 2.0, size=width),
        "weight_0": rng.normal(scale=0.1, size=(16, width)),
        "bias_0": rng.normal(scale=0.1, size=16),
        "weight_1": rng.normal(scale=0.1, size=(JOINTS, 16)),
        "bias_1": rng.normal(scale=0.1, size=JOINTS),
        "adapt_mean": rng.normal(scale=0.1, size=history),
        "adapt_scale": rng.uniform(0.5, 2.0, size=history),
        "adapt_weight_0": rng.normal(scale=0.3, size=(8, history)),
        "adapt_bias_0": rng.normal(scale=0.1, size=8),
        "adapt_weight_1": rng.normal(scale=0.3, size=(PARAMS, 8)),
        "adapt_bias_1": rng.normal(scale=0.1, size=PARAMS),
        "adapt_mirror_linear": np.eye(history)[rng.permutation(history)],
        "adapt_mirror_offset": rng.normal(scale=0.1, size=history),
        "params_start": np.int64(width - PARAMS),
    }
    if mirror:
        arrays.update(
            mirror_linear=np.eye(width)[rng.permutation(width)],
            mirror_offset=rng.normal(scale=0.1, size=width),
            action_joints=np.eye(JOINTS)[rng.permutation(JOINTS)],
            action_offset=rng.normal(scale=0.1, size=JOINTS),
        )
    return arrays


def test_a_student_acts_on_its_estimate_or_on_given_values() -> None:
    rng = np.random.default_rng(0)
    arrays = _arrays(rng, mirror=True)
    student = Actor(arrays)
    teacher = Actor({k: v for k, v in arrays.items() if not k.startswith(("adapt_", "params_"))})
    width = teacher.width
    assert student.inputs == width + sum(w for _, w in HISTORY_TERMS)
    observation = rng.normal(size=(3, student.inputs))
    estimate = student.estimate_params(observation)
    assert estimate.shape == (3, PARAMS) and np.abs(estimate).max() <= 1.0
    told = observation[:, :width].copy()
    told[:, width - PARAMS :] = estimate
    np.testing.assert_allclose(student(observation), teacher(told), atol=1e-12)
    np.testing.assert_allclose(student.last_params, estimate)
    given = np.array([0.5, -0.5, 0.0, 1.0, -1.0])
    told[:, width - PARAMS :] = given
    np.testing.assert_allclose(student(observation, given), teacher(told), atol=1e-12)
    # Whatever the observation carries in robot_params is ignored.
    noisy = observation.copy()
    noisy[:, width - PARAMS : width] = rng.normal(size=(3, PARAMS))
    np.testing.assert_allclose(student(noisy), student(observation), atol=1e-12)


def test_the_torchscript_student_is_the_numpy_one(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from student_bundle import torchscript

    rng = np.random.default_rng(1)
    arrays = _arrays(rng, mirror=True)
    torchscript(arrays, tmp_path / "student.pt")
    student = Actor(arrays)
    observation = rng.normal(size=(5, student.inputs))
    traced = torch.jit.load(str(tmp_path / "student.pt"))
    out = traced(torch.as_tensor(observation, dtype=torch.float32)).detach().numpy()
    np.testing.assert_allclose(out, student(observation), atol=1e-4)


def test_the_torchscript_student_low_passes_its_estimate_as_the_runtime_does(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from student_bundle import torchscript

    rng = np.random.default_rng(5)
    arrays = _arrays(rng, mirror=True)
    tau, dt = 0.1, 0.02
    alpha = 1.0 - np.exp(-dt / tau)
    torchscript(arrays, tmp_path / "student.pt", tau, dt)
    traced = torch.jit.load(str(tmp_path / "student.pt"))
    assert traced.params_tau == pytest.approx(tau)
    student = Actor(arrays)
    smoothed = None
    for _ in range(4):
        observation = rng.normal(size=(3, student.inputs))
        estimate = student.estimate_params(observation)
        smoothed = estimate if smoothed is None else smoothed + alpha * (estimate - smoothed)
        out = traced(torch.as_tensor(observation, dtype=torch.float32)).detach().numpy()
        np.testing.assert_allclose(out, student(observation, smoothed), atol=1e-4)


def test_measured_values_are_normalised_over_the_bundles_ranges() -> None:
    config = {"robot_params": {"names": NAMES, "ranges": RANGES}}
    out = normalise_params({"delay_ms": 35.0, "effort_scale": 0.7, "armature_scale": 9.0}, config)
    np.testing.assert_allclose(out[[0, 3, 4]], [0.0, -1.0, 1.0])
    assert np.isnan(out[[1, 2]]).all()
    with pytest.raises(KeyError):
        normalise_params({"friction": 1.0}, config)
    assert history_base("joint_pos_adapt_history") == "joint_pos"
    assert history_base("joint_pos_history") == "joint_pos"
    assert history_base("contract") == "contract"


@pytest.fixture()
def bundle(tmp_path: Path) -> Bundle:
    rng = np.random.default_rng(2)
    urdf = nk.load_model()
    moving = [j for j in urdf.joints if j.type != "fixed"]
    bodies = ["base_link"] + [j.child for j in moving]
    sole = [[0.0, 0.0, -0.04511]] * 2
    corners = [[[x, y, -0.04511] for x in (-0.05, 0.1) for y in (-0.04, 0.04)]] * 2
    spec = body_model(bodies, [j.name for j in moving], sole, corners, 0.24801)
    (tmp_path / "body_model.json").write_text(json.dumps(spec))
    np.savez(tmp_path / "actor.npz", **_arrays(rng, mirror=False))
    config = {
        "control_dt": 0.02,
        "leg_length": 0.24801,
        "actuated": list(nk.ACTUATED_JOINTS),
        "default_pos": [0.1] * JOINTS,
        "action_scale": 0.25,
        "skill_mean": rng.normal(size=SKILL).tolist(),
        "skill_std": rng.uniform(0.5, 2.0, size=SKILL).tolist(),
        "units": None,
        "plan_limit": 50.0,
        "policy_terms": [list(t) for t in TEACHER_TERMS + HISTORY_TERMS],
        "history": {name: LENGTH for name, _ in HISTORY_TERMS},
        "robot_params": {"names": NAMES, "ranges": RANGES},
    }
    (tmp_path / "runtime.json").write_text(json.dumps(config))
    (tmp_path / "plans").mkdir()
    np.savez(
        tmp_path / "plans" / "test.npz",
        view=rng.normal(size=(5, VIEW, SKILL)), has_plan=np.ones(5, dtype=bool),
        path=rng.normal(scale=0.1, size=(5, VIEW, 3)), command=np.zeros((5, 3)),
    )  # fmt: skip
    return Bundle(tmp_path)


def test_the_controller_keeps_the_estimate_for_what_is_not_measured(bundle: Bundle) -> None:
    rng = np.random.default_rng(3)
    q = np.array([nk.NOMINAL_STAND_JOINT_POS.get(n, 0.0) for n in bundle.model.joint_names])
    estimating = Controller(bundle, "test")
    told = Controller(bundle, "test", robot_params={"delay_ms": 35.0, "effort_scale": 1.1})
    width = bundle.actor.width
    for _ in range(3):
        gyro = rng.normal(scale=0.1, size=3)
        sensors = (q, np.array([0.0, 0.0, -1.0]), gyro, np.array([0.5, 0.5]))
        observation = estimating.observe(*sensors)
        assert not observation[width - PARAMS : width].any()
        history = observation[width : width + 3 * LENGTH].reshape(LENGTH, 3)
        np.testing.assert_allclose(history[-1], gyro)
        estimating.act(observation)
        np.testing.assert_allclose(
            bundle.actor.last_params, bundle.actor.estimate_params(observation)
        )
        # Its own history (its actions differ), its own estimate for what is not given.
        observation = told.observe(*sensors)
        estimate = bundle.actor.estimate_params(observation)
        told.act(observation)
        np.testing.assert_allclose(bundle.actor.last_params[[0, 3]], [0.0, 1.0])
        np.testing.assert_allclose(bundle.actor.last_params[[1, 2, 4]], estimate[[1, 2, 4]])


def test_the_controller_low_passes_the_estimate_and_keeps_given_values(bundle: Bundle) -> None:
    rng = np.random.default_rng(4)
    q = np.array([nk.NOMINAL_STAND_JOINT_POS.get(n, 0.0) for n in bundle.model.joint_names])
    tau = 0.1
    alpha = 1.0 - np.exp(-0.02 / tau)
    smooth = Controller(bundle, "test", params_tau=tau)
    told = Controller(bundle, "test", robot_params={"delay_ms": 35.0}, params_tau=tau)
    expected = expected_told = None
    for _ in range(4):
        gyro = rng.normal(scale=0.5, size=3)
        sensors = (q, np.array([0.0, 0.0, -1.0]), gyro, np.array([0.5, 0.5]))
        observation = smooth.observe(*sensors)
        estimate = bundle.actor.estimate_params(observation)
        expected = estimate if expected is None else expected + alpha * (estimate - expected)
        smooth.act(observation)
        np.testing.assert_allclose(bundle.actor.last_params, expected, atol=1e-12)
        observation = told.observe(*sensors)
        estimate = bundle.actor.estimate_params(observation)
        if expected_told is None:
            expected_told = estimate
        else:
            expected_told = expected_told + alpha * (estimate - expected_told)
        told.act(observation)
        np.testing.assert_allclose(bundle.actor.last_params[0], 0.0)
        np.testing.assert_allclose(bundle.actor.last_params[1:], expected_told[1:], atol=1e-12)
    # A new run starts from its own first estimate.
    smooth.reset()
    observation = smooth.observe(q, np.array([0.0, 0.0, -1.0]), np.zeros(3), np.array([0.5, 0.5]))
    smooth.act(observation)
    np.testing.assert_allclose(
        bundle.actor.last_params, bundle.actor.estimate_params(observation), atol=1e-12
    )
