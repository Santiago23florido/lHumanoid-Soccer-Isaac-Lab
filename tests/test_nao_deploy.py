"""The robot's estimator against the definitions it transcribes.

* :class:`~humanoid_transfer.nao.deploy.kinematics.BodyModel` against the URDF's
  own forward kinematics (:func:`~humanoid_transfer.nao.assets.nao_kinematics.forward_kinematics`);
* the NumPy ``z*`` (:mod:`~humanoid_transfer.nao.deploy.skill`) against the torch
  one the tracker was trained on (:mod:`~humanoid_transfer.transfer.latent.skill_torch`);
* the estimator's finite differences and leg odometry against the exact
  velocities of a body moving as a rigid tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.deploy.estimator import (  # noqa: E402
    Estimator,
    rpy_matrix,
    tilt_from_gravity,
)
from humanoid_transfer.nao.deploy.kinematics import BodyModel  # noqa: E402
from humanoid_transfer.nao.deploy.skill import (  # noqa: E402
    proprioceptive_skill_state,
    skill_state,
)
from humanoid_transfer.nao.sim2real.body_model import body_model  # noqa: E402

torch = pytest.importorskip("torch")
from humanoid_transfer.transfer.latent import skill_torch  # noqa: E402

LEG_LENGTH = 0.24801
SOLE = [[0.0, 0.0, -0.04511]] * 2
CORNERS = [
    [[x, y, -0.04511] for x in (-0.05, 0.1) for y in (-0.04, 0.03)],
    [[x, y, -0.04511] for x in (-0.05, 0.1) for y in (-0.03, 0.04)],
]


def _simulator_like_names() -> tuple[list[str], list[str]]:
    """Bodies and joints as the importer would list them: the base, then each moving joint's
    child, in a shuffled order, so nothing depends on the URDF's."""
    model = nk.load_model()
    moving = [j for j in model.joints if j.type != "fixed"]
    order = np.random.default_rng(3).permutation(len(moving))
    bodies = ["base_link"] + [moving[i].child for i in order]
    joints = [moving[i].name for i in order]
    return bodies, joints


@pytest.fixture(scope="module")
def model() -> BodyModel:
    bodies, joints = _simulator_like_names()
    return BodyModel(body_model(bodies, joints, SOLE, CORNERS, LEG_LENGTH))


def _random_q(model: BodyModel, rng: np.random.Generator, count: int) -> np.ndarray:
    return rng.uniform(-0.6, 0.6, size=(count, len(model.joint_names)))


def test_forward_kinematics_matches_the_urdf(model: BodyModel) -> None:
    rng = np.random.default_rng(0)
    q = _random_q(model, rng, 3)
    rot, pos = model.forward(q)
    for n in range(len(q)):
        reference = nk.forward_kinematics(dict(zip(model.joint_names, q[n], strict=True)))
        for b, name in enumerate(model.names):
            np.testing.assert_allclose(rot[n, b], reference[name][:3, :3], atol=1e-12)
            np.testing.assert_allclose(pos[n, b], reference[name][:3, 3], atol=1e-12)


def test_merged_bodies_keep_the_robot_s_mass_and_centre(model: BodyModel) -> None:
    urdf = nk.load_model()
    assert model.total_mass == pytest.approx(urdf.total_mass, rel=1e-12)
    joint_pos = dict(nk.NOMINAL_STAND_JOINT_POS)
    q = np.array([[joint_pos.get(name, 0.0) for name in model.joint_names]])
    rot, pos = model.forward(q)
    centre = np.einsum("b,bi->i", model.mass, model.centres(rot, pos)[0]) / model.total_mass
    np.testing.assert_allclose(centre, nk.center_of_mass(joint_pos), atol=1e-12)


def _body_constants(model: BodyModel) -> skill_torch.BodyConstants:
    from humanoid_transfer.common.similarity import UnitSystem

    return skill_torch.BodyConstants(
        masses=torch.tensor(model.mass),
        inertia_local=torch.tensor(model.inertia),
        base_index=model.base,
        foot_index=model.feet,
        sole_from_com=torch.tensor(model.sole_from_com),
        units=UnitSystem(LEG_LENGTH, model.total_mass),
    ).to("cpu", torch.float64)


def _random_state(model: BodyModel, rng: np.random.Generator, count: int):
    q = _random_q(model, rng, count)
    base = rpy_matrix(*(rng.uniform(-0.3, 0.3, size=(3, count))))
    rot_b, pos_b = model.forward(q)
    rot = base[:, None] @ rot_b
    com = np.einsum("nij,nbj->nbi", base, model.centres(rot_b, pos_b)) + rng.normal(
        size=(count, 1, 3)
    )
    vel = rng.normal(size=com.shape)
    spin = rng.normal(size=com.shape)
    return rot, com, vel, spin


def _quat(rot: np.ndarray) -> torch.Tensor:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat()
    return torch.tensor(np.roll(xyzw, 1, axis=-1).reshape(*rot.shape[:-2], 4))


def test_skill_state_matches_the_torch_definition(model: BodyModel) -> None:
    rng = np.random.default_rng(1)
    rot, com, vel, spin = _random_state(model, rng, 4)
    force = rng.normal(size=(4, 2, 3))
    ours = skill_state(model, rot, com, vel, spin, force)
    theirs = skill_torch.skill_state(
        _body_constants(model),
        _quat(rot),
        torch.tensor(com),
        torch.tensor(vel),
        torch.tensor(spin),
        torch.tensor(force),
    ).numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_proprioceptive_state_matches_the_torch_definition(model: BodyModel) -> None:
    rng = np.random.default_rng(2)
    rot, com, vel, spin = _random_state(model, rng, 4)
    load = rng.uniform(0.0, 1.0, size=(4, 2))
    ours = proprioceptive_skill_state(model, rot, com, vel, spin, load)
    body = _body_constants(model)
    corners = torch.tensor(model.corners)
    theirs = skill_torch.proprioceptive_skill_state(
        body,
        _quat(rot),
        torch.tensor(com),
        torch.tensor(vel),
        torch.tensor(spin),
        torch.tensor(load),
        base_velocity=False,
        corners=corners,
    ).numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_tilt_is_read_back_from_gravity() -> None:
    roll, pitch = np.array([0.1, -0.3]), np.array([-0.2, 0.25])
    base = rpy_matrix(roll, pitch, np.array([1.0, -2.0]))
    gravity = np.einsum("nji,j->ni", base, np.array([0.0, 0.0, -1.0]))
    back = tilt_from_gravity(gravity)
    np.testing.assert_allclose(back[0], roll, atol=1e-12)
    np.testing.assert_allclose(back[1], pitch, atol=1e-12)


def test_estimator_follows_a_walking_tree(model: BodyModel) -> None:
    """A body whose left sole stays planted while it turns, tilts and moves its joints: the
    estimator's finite differences converge to the exact velocities measured from that sole,
    and its odometry to the true displacement of the centre of mass."""
    dt = 1e-4
    steps = 200
    q0 = np.array([nk.NOMINAL_STAND_JOINT_POS.get(n, 0.0) for n in model.joint_names])
    rate = np.random.default_rng(4).uniform(-1.0, 1.0, size=q0.shape)
    estimator = Estimator(model, 1, dt, pivot="sole")
    load = np.array([[0.9, 0.1]])
    left = model.feet[0]

    def state(t: float):
        q = q0 + rate * t
        roll, pitch, yaw = 0.05 + 0.3 * t, -0.04 + 0.2 * t, 0.7 + 0.5 * t
        base = rpy_matrix(np.array(roll), np.array(pitch), np.array(yaw))
        rot_b, pos_b = model.forward(q[None])
        rot = base @ rot_b[0]
        com_b = np.einsum("ij,bj->bi", base, model.centres(rot_b, pos_b)[0])
        sole_b = com_b[left] + rot[left] @ model.sole_from_com[0]
        com = com_b - sole_b  # the left sole is the world's origin, fixed
        gyro = base.T @ _base_rate(roll, pitch, yaw, t)
        gravity = base.T @ np.array([0.0, 0.0, -1.0])
        return q, rot, com, gravity, gyro

    def _base_rate(roll, pitch, yaw, t, h=1e-7):
        before = rpy_matrix(
            np.array(roll - 0.3 * h), np.array(pitch - 0.2 * h), np.array(yaw - 0.5 * h)
        )
        after = rpy_matrix(
            np.array(roll + 0.3 * h), np.array(pitch + 0.2 * h), np.array(yaw + 0.5 * h)
        )
        w = (
            (after - before)
            @ rpy_matrix(np.array(roll), np.array(pitch), np.array(yaw)).T
            / (2 * h)
        )
        return np.array([w[2, 1], w[0, 2], w[1, 0]])

    start = origin = None
    for k in range(steps):
        q, rot, com, gravity, gyro = state(k * dt)
        z = estimator.update(q[None], gravity[None], gyro[None], load)
        centre = np.einsum("b,bi->i", model.mass, com) / model.total_mass
        if start is None:
            start, origin = centre, estimator.position[0].copy()
    # Exact velocities at the last step, by central differences of the exact states.
    t = (steps - 1) * dt
    h = 1e-6
    _, rot_a, com_a, _, _ = state(t - h)
    _, rot_b, com_b, _, _ = state(t + h)
    velocity = (com_b - com_a) / (2 * h)
    spin = []
    for a, b, r in zip(rot_a, rot_b, rot, strict=True):
        w = (b - a) @ r.T / (2 * h)
        spin.append([w[2, 1], w[0, 2], w[1, 0]])
    exact = proprioceptive_skill_state(
        model, rot[None], com[None], velocity[None], np.array(spin)[None], load
    )
    np.testing.assert_allclose(z, exact, atol=2e-3)
    # Odometry: the centre of mass moved by what it moved over the planted sole; the
    # estimator's frame starts at yaw 0, the truth's at 0.7 rad.
    moved = centre - start
    turn = rpy_matrix(np.array(0.0), np.array(0.0), np.array(-0.7))
    np.testing.assert_allclose(estimator.position[0] - origin, (turn @ moved)[:2], atol=1e-5)
