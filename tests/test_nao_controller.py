"""The robot's controller on a synthetic bundle: the observation's layout and its parts.

The simulator-built bundle is checked against the simulator by
``nao/scripts/check_runtime.py``; this checks, without it, what that replay relies on:
the layout follows ``policy_terms``, a history fills with its first reading and then
shifts (Isaac Lab's ``CircularBuffer``, oldest first), the contract view is the plan
relative to the estimate, the path is zero without a plan, and the targets are the
crouch plus the scaled action.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.deploy.controller import Bundle, Controller  # noqa: E402
from humanoid_transfer.nao.sim2real.body_model import body_model  # noqa: E402

VIEW, SKILL, JOINTS = 4, 27, len(nk.ACTUATED_JOINTS)
TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", JOINTS),
    ("joint_vel", JOINTS),
    ("actions", JOINTS),
    ("contract", VIEW * SKILL + SKILL),
    ("contract_path", VIEW * 3),
    ("actions_history", 3 * JOINTS),
]


@pytest.fixture()
def bundle(tmp_path: Path) -> Bundle:
    rng = np.random.default_rng(0)
    urdf = nk.load_model()
    moving = [j for j in urdf.joints if j.type != "fixed"]
    bodies = ["base_link"] + [j.child for j in moving]
    sole = [[0.0, 0.0, -0.04511]] * 2
    corners = [[[x, y, -0.04511] for x in (-0.05, 0.1) for y in (-0.04, 0.04)]] * 2
    spec = body_model(bodies, [j.name for j in moving], sole, corners, 0.24801)
    (tmp_path / "body_model.json").write_text(json.dumps(spec))
    width = sum(w for _, w in TERMS)
    np.savez(
        tmp_path / "actor.npz",
        mean=np.zeros(width), scale=np.ones(width),
        weight_0=rng.normal(scale=0.05, size=(8, width)), bias_0=np.zeros(8),
        weight_1=rng.normal(scale=0.05, size=(JOINTS, 8)), bias_1=np.zeros(JOINTS),
    )  # fmt: skip
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
        "policy_terms": [list(t) for t in TERMS],
        "history": {"actions_history": 3},
        "joint_lower": [-3.0] * JOINTS,
        "joint_upper": [3.0] * JOINTS,
    }
    (tmp_path / "runtime.json").write_text(json.dumps(config))
    (tmp_path / "plans").mkdir()
    steps = 5
    has_plan = np.array([False, True, True, True, True])
    np.savez(
        tmp_path / "plans" / "test.npz",
        view=rng.normal(size=(steps, VIEW, SKILL)), has_plan=has_plan,
        path=rng.normal(scale=0.1, size=(steps, VIEW, 3)), command=np.zeros((steps, 3)),
    )  # fmt: skip
    return Bundle(tmp_path)


def _sensors(bundle: Bundle, rng: np.random.Generator):
    q = np.array([nk.NOMINAL_STAND_JOINT_POS.get(n, 0.0) for n in bundle.model.joint_names])
    q = q + rng.normal(scale=0.01, size=q.shape)
    return q, np.array([0.0, 0.0, -1.0]), rng.normal(scale=0.1, size=3), np.array([0.5, 0.5])


def test_the_observation_follows_the_layout(bundle: Bundle) -> None:
    rng = np.random.default_rng(1)
    controller = Controller(bundle, "test")
    starts = np.cumsum([0] + [w for _, w in TERMS])
    part = {name: slice(starts[i], starts[i + 1]) for i, (name, _) in enumerate(TERMS)}
    actions = []
    for k in range(4):
        q, gravity, gyro, loads = _sensors(bundle, rng)
        observation = controller.observe(q, gravity, gyro, loads)
        assert observation.shape == (controller.actor.inputs,)
        np.testing.assert_allclose(observation[part["base_ang_vel"]], gyro)
        index = [bundle.model.joint_names.index(n) for n in controller.actuated]
        np.testing.assert_allclose(observation[part["joint_pos"]], q[index] - 0.1)
        np.testing.assert_allclose(observation[part["actions"]], controller.last_action)
        history = observation[part["actions_history"]].reshape(3, JOINTS)
        expected = ([np.zeros(JOINTS)] * 3 + actions)[-3:]
        np.testing.assert_allclose(history, np.stack(expected))
        plan = bundle.plan("test")
        contract = observation[part["contract"]]
        estimate = contract[VIEW * SKILL :]
        np.testing.assert_allclose(estimate, controller.estimate)
        relative = contract[: VIEW * SKILL].reshape(VIEW, SKILL)
        if plan["has_plan"][k]:
            np.testing.assert_allclose(relative, np.clip(plan["view"][k] - estimate, -50, 50))
        else:
            assert not relative.any()
            assert not observation[part["contract_path"]].any()
        targets = controller.act(observation)
        np.testing.assert_allclose(targets, 0.1 + 0.25 * controller.last_action)
        actions.append(controller.last_action.copy())


class _FakeRobot:
    """A robot standing still behind LoLA's interface; its chest button pressed when told."""

    def __init__(self, bundle: Bundle, presses: set[int], tilt_after: int | None = None) -> None:
        from humanoid_transfer.nao.deploy import lola

        self.lola = lola
        self.reads = 0
        self.presses = presses
        self.tilt_after = tilt_after
        posture = dict(zip(bundle.config["actuated"], bundle.config["default_pos"], strict=True))
        self.position = [posture.get(n, 0.0) for n in lola.JOINTS]
        self.written = []

    def read(self):
        self.reads += 1
        pitch = 0.9 if self.tilt_after is not None and self.reads > self.tilt_after else 0.0
        touch = [1.0 if self.reads in self.presses else 0.0] + [0.0] * 13
        return self.lola.Reading(
            {
                "Position": self.position,
                "Gyroscope": [0.0, 0.0, 0.0],
                "Angles": [0.0, pitch],
                "FSR": [0.6] * 8,
                "Temperature": [40.0] * 25,
                "Touch": touch,
            }
        )

    def write(self, positions, stiffness) -> None:
        self.written.append((dict(positions), stiffness))

    def close(self) -> None:
        pass


def test_a_session_waits_for_the_button_runs_and_stops(bundle: Bundle, tmp_path: Path) -> None:
    from humanoid_transfer.nao.deploy.session import Phases, Session

    phases = Phases(ramp_s=0.02, pose_s=0.02, hold_s=0.02, stop_s=0.02, wait_button=True)
    # Reads 1-4 are the first reading and the ramp, pose and hold phases' one step each; the
    # wait starts at read 5. Pressed at the 6th read, released at the 7th: the policy starts
    # and runs every plan step.
    robot = _FakeRobot(bundle, presses={6})
    session = Session(Controller(bundle, "test"), robot, phases=phases)
    assert session.run() == "end"
    assert len(session.log.rows["compute_s"]) == 5
    session.log.save(tmp_path / "run.npz", plan="test")
    # Pressed again during the policy: it stops.
    robot = _FakeRobot(bundle, presses={6, 10})
    session = Session(Controller(bundle, "test"), robot, phases=phases)
    assert session.run() == "button"
    # Tilted past the fall limit: it stops and every joint goes soft.
    robot = _FakeRobot(bundle, presses={6}, tilt_after=9)
    session = Session(Controller(bundle, "test"), robot, phases=phases)
    assert session.run() == "fall"
    assert robot.written[-1][1] == session.safety_config.fall_stiffness
