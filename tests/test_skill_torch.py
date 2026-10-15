"""The GPU skill state is the certified NumPy one, transcribed.

An environment that rewards a body for the ``z*`` it reaches has to compute
``z*`` every step, in torch. The definition stays in
:mod:`humanoid_transfer.common.skill_state`; these tests hold the torch copy to
it, on random multibody states and on states recorded from the simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.frames import quat_to_matrix  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import MultibodySnapshot, encode  # noqa: E402
from humanoid_transfer.transfer.latent.skill_torch import (  # noqa: E402
    BodyConstants,
    skill_state,
)

REPO = Path(__file__).resolve().parents[1]
BODIES = 7
UNITS = UnitSystem(length=0.70, mass=30.0)


def _random_state(seed: int, envs: int = 5):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(BODIES, 3, 3))
    positions = rng.normal(size=(envs, BODIES, 3))
    positions[..., 2] = np.abs(positions[..., 2]) + 0.2
    return {
        "masses": rng.uniform(0.5, 6.0, size=BODIES),
        "inertia": a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(3),
        "quat": rng.normal(size=(envs, BODIES, 4)),
        "com_pos": positions,
        "com_lin_vel": rng.normal(size=(envs, BODIES, 3)),
        "ang_vel": rng.normal(size=(envs, BODIES, 3)),
        "foot_force": rng.normal(size=(envs, 2, 3)) * 100.0,
        "sole_from_com": rng.normal(size=(2, 3)) * 0.05,
    }


def _numpy(state: dict, units: UnitSystem, base: int, feet: tuple[int, int]) -> np.ndarray:
    envs = state["com_pos"].shape[0]
    snapshot = MultibodySnapshot(
        masses=np.broadcast_to(state["masses"], (envs, BODIES)),
        inertia_local=np.broadcast_to(state["inertia"], (envs, BODIES, 3, 3)),
        link_rot=quat_to_matrix(state["quat"]),
        com_pos=state["com_pos"],
        com_lin_vel=state["com_lin_vel"],
        ang_vel=state["ang_vel"],
        foot_force=state["foot_force"],
        base_index=base,
        foot_index=feet,
        sole_from_com=state["sole_from_com"],
    )
    return encode(snapshot, units)


def _torch(state: dict, units: UnitSystem, base: int, feet: tuple[int, int]) -> np.ndarray:
    def t(x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float64)

    body = BodyConstants(
        masses=t(state["masses"]),
        inertia_local=t(state["inertia"]),
        base_index=base,
        foot_index=feet,
        sole_from_com=t(state["sole_from_com"]),
        units=units,
    )
    z = skill_state(
        body,
        t(state["quat"]),
        t(state["com_pos"]),
        t(state["com_lin_vel"]),
        t(state["ang_vel"]),
        t(state["foot_force"]),
    )
    return z.numpy()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_torch_state_equals_the_numpy_encoder_on_random_bodies(seed: int) -> None:
    state = _random_state(seed)
    expected = _numpy(state, UNITS, base=0, feet=(5, 6))
    np.testing.assert_allclose(_torch(state, UNITS, base=0, feet=(5, 6)), expected, atol=1e-10)


def test_the_base_and_feet_are_read_from_where_the_body_says() -> None:
    state = _random_state(3)
    expected = _numpy(state, UNITS, base=2, feet=(4, 1))
    np.testing.assert_allclose(_torch(state, UNITS, base=2, feet=(4, 1)), expected, atol=1e-10)


def test_the_torch_state_reproduces_the_recorded_simulator_states() -> None:
    path = REPO / "g1" / "rollouts" / "teacher.npz"
    if not path.exists():
        pytest.skip("teacher rollouts are untracked and not on this machine")
    data = np.load(path)
    raw = data["raw_link_quat"].shape[1]
    steps = slice(0, 50)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    state = {
        "masses": data["masses"],
        "inertia": data["inertia_link"],
        "quat": data["raw_link_quat"][steps].reshape(-1, *data["raw_link_quat"].shape[2:]),
        "com_pos": data["raw_com_pos"][steps].reshape(-1, *data["raw_com_pos"].shape[2:]),
        "com_lin_vel": data["raw_com_lin_vel"][steps].reshape(
            -1, *data["raw_com_lin_vel"].shape[2:]
        ),
        "ang_vel": data["raw_ang_vel"][steps].reshape(-1, *data["raw_ang_vel"].shape[2:]),
        "foot_force": data["foot_force"][steps, :raw].reshape(-1, 2, 3),
        "sole_from_com": data["sole_from_com"],
    }
    feet = tuple(int(i) for i in data["foot_index"])
    got = _torch(state, units, int(data["base_index"]), feet)
    recorded = data["skill_state"][steps, :raw].reshape(-1, got.shape[-1])
    np.testing.assert_allclose(got, recorded, atol=2e-4, rtol=1e-4)


def _proprioceptive_case(seed: int):
    """A random body whose lower sole is on the ground and whose loaded foot stands still."""
    from humanoid_transfer.transfer.latent.skill_torch import sole_points

    state = _random_state(seed)
    t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float64) for k, v in state.items()}
    body = BodyConstants(
        masses=t["masses"],
        inertia_local=t["inertia"],
        base_index=0,
        foot_index=(5, 6),
        sole_from_com=t["sole_from_com"],
        units=UNITS,
    )
    soles = sole_points(body, t["quat"], t["com_pos"])
    t["com_pos"][..., 2] -= soles[..., 2].min(dim=1).values[:, None]
    load = t["foot_force"][..., 2].abs() / (UNITS.mass * 9.81)
    t["foot_force"] = torch.zeros_like(t["foot_force"])
    t["foot_force"][..., 2] = load * UNITS.mass * 9.81
    return body, t, load


@pytest.mark.parametrize("seed", [0, 1])
def test_the_body_reads_its_own_state_where_the_ground_is_under_its_lower_sole(seed: int) -> None:
    from humanoid_transfer.transfer.latent.skill_torch import proprioceptive_skill_state

    body, t, load = _proprioceptive_case(seed)
    args = (body, t["quat"], t["com_pos"], t["com_lin_vel"], t["ang_vel"])
    expected = skill_state(*args, t["foot_force"])
    measured = proprioceptive_skill_state(*args, load)
    np.testing.assert_allclose(measured.numpy(), expected.numpy(), atol=1e-10)
    # Where the body stands and which way it faces are not sensed, and do not matter.
    raised = t["com_pos"] + torch.tensor([3.0, -2.0, 0.7], dtype=torch.float64)
    moved = proprioceptive_skill_state(
        body, t["quat"], raised, t["com_lin_vel"], t["ang_vel"], load
    )
    np.testing.assert_allclose(moved.numpy(), expected.numpy(), atol=1e-10)


def test_leg_odometry_recovers_the_velocities_when_the_loaded_foot_stands_still() -> None:
    from humanoid_transfer.transfer.latent.skill_torch import (
        proprioceptive_skill_state,
        sole_points,
    )

    body, t, load = _proprioceptive_case(2)
    # Shift every velocity so that the more loaded foot's sole is at rest.
    feet = list(body.foot_index)
    soles = sole_points(body, t["quat"], t["com_pos"])
    offset = soles - t["com_pos"][:, feet]
    sole_vel = t["com_lin_vel"][:, feet] + torch.linalg.cross(t["ang_vel"][:, feet], offset, dim=-1)
    stance = load.argmax(dim=-1)
    still = t["com_lin_vel"] - sole_vel[torch.arange(len(stance)), stance][:, None, :]
    args = (body, t["quat"], t["com_pos"], still, t["ang_vel"])
    expected = skill_state(*args, t["foot_force"])
    np.testing.assert_allclose(
        proprioceptive_skill_state(*args, load, base_velocity=False).numpy(),
        expected.numpy(),
        atol=1e-10,
    )
    # Without a base velocity, a drift common to every link is not sensed.
    drift = still + torch.tensor([0.4, -0.1, 0.05], dtype=torch.float64)
    np.testing.assert_allclose(
        proprioceptive_skill_state(
            body, t["quat"], t["com_pos"], drift, t["ang_vel"], load, base_velocity=False
        ).numpy(),
        expected.numpy(),
        atol=1e-10,
    )
