"""The robot's runtime pieces that need no robot and no simulator.

* MessagePack: LoLA's message types round-trip, and the relay's prebuilt limp
  command is what the encoder writes;
* LoLA's framing: messages split across reads, or several in one, are found whole;
* the NumPy actor against a torch network with the same weights, plain and
  symmetrised the way ``transfer/scripts/symmetrize_actor.py`` builds it;
* the safety layer's bounds and stop conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.deploy import lola  # noqa: E402
from humanoid_transfer.nao.deploy.msgpack_lite import packb, unpackb  # noqa: E402
from humanoid_transfer.nao.deploy.policy import Actor  # noqa: E402
from humanoid_transfer.nao.deploy.relay import LIMP  # noqa: E402
from humanoid_transfer.nao.deploy.session import Safety, SafetyConfig  # noqa: E402


def _sensor_message(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    message = {key: [float(v) for v in rng.normal(size=n)] for key, n in lola.SENSOR_KEYS.items()}
    message["Status"] = [int(v) for v in rng.integers(0, 3, size=25)]
    return message


def test_messages_round_trip() -> None:
    message = {
        "Position": [0.1, -2.5, 3.0e-7],
        "Stiffness": [1.0, 0.0],
        "Flags": [True, False, None],
        "Ints": [0, 127, 128, -1, -32, -33, 70000, -70000, 2**40],
        "Text": "x" * 40,
        "Nested": {"a": [1, 2], "b": {}},
    }
    back = unpackb(packb(message))
    assert back["Ints"] == message["Ints"]
    assert back["Flags"] == message["Flags"]
    assert back["Text"] == message["Text"]
    assert back["Nested"] == message["Nested"]
    np.testing.assert_allclose(back["Position"], message["Position"], rtol=1e-6)


def test_relay_limp_command_is_the_runtime_s() -> None:
    # The relay builds it without the runtime (Python 2.7 on the robot): every key, all zero.
    assert unpackb(LIMP) == lola.command({}, 0.0)
    assert LIMP == packb(lola.command({}, 0.0))


def test_lola_messages_are_framed_across_reads() -> None:
    first, second = packb(_sensor_message(0)), packb(_sensor_message(1))
    stream = first + second
    cut = len(first) + 10
    messages, rest = lola.split_messages(stream[:cut])
    assert messages == [first]
    messages, rest = lola.split_messages(rest + stream[cut:])
    assert messages == [second] and rest == b""


def test_reading_maps_lola_s_joints_and_sensors() -> None:
    message = _sensor_message(2)
    message["Position"] = [float(i) for i in range(25)]
    message["Angles"] = [0.1, -0.2]
    message["FSR"] = [0.5] * 4 + [1.0] * 4
    reading = lola.Reading(message)
    assert list(reading.joint(["LHipYawPitch", "RHipYawPitch", "RHand"])) == [7.0, 7.0, 24.0]
    roll, pitch = 0.1, -0.2
    expected = [np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)]
    np.testing.assert_allclose(reading.gravity(), expected)
    np.testing.assert_allclose(reading.foot_loads(5.0), [0.4, 0.8])
    out = lola.command({"LKneePitch": 1.2, "RHipYawPitch": 9.0}, 0.5, base=np.zeros(25))
    assert out["Position"][lola.JOINTS.index("LKneePitch")] == pytest.approx(1.2)
    assert 9.0 not in out["Position"]
    assert out["Stiffness"] == [0.5] * 25
    assert set(out) == {"Position", "Stiffness", "Sonar", *lola.ACTUATOR_KEYS}


torch = pytest.importorskip("torch")


def _torch_actor(rng: np.random.Generator, sizes=(11, 16, 8, 5)):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [torch.nn.Linear(a, b), torch.nn.ELU()]
    mlp = torch.nn.Sequential(*layers[:-1]).double()
    mean = torch.tensor(rng.normal(size=sizes[0]))
    scale = torch.tensor(rng.uniform(0.5, 2.0, size=sizes[0]))
    arrays = {"mean": mean.numpy(), "scale": scale.numpy()}
    linear = [m for m in mlp if isinstance(m, torch.nn.Linear)]
    for i, layer in enumerate(linear):
        arrays[f"weight_{i}"] = layer.weight.detach().numpy()
        arrays[f"bias_{i}"] = layer.bias.detach().numpy()

    def forward(x):
        return mlp((x - mean) / scale)

    return forward, arrays


def test_actor_matches_torch() -> None:
    rng = np.random.default_rng(0)
    forward, arrays = _torch_actor(rng)
    x = rng.normal(size=(7, 11))
    expected = forward(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(Actor(arrays)(x), expected, rtol=1e-12, atol=1e-12)


def test_symmetrised_actor_matches_the_reynolds_average() -> None:
    rng = np.random.default_rng(1)
    forward, arrays = _torch_actor(rng)
    linear = rng.normal(size=(11, 11))
    offset = rng.normal(size=11)
    joints = rng.normal(size=(5, 5))
    action_offset = rng.normal(size=5)
    arrays.update(
        mirror_linear=linear,
        mirror_offset=offset,
        action_joints=joints,
        action_offset=action_offset,
    )
    x = rng.normal(size=(4, 11))
    t = torch.tensor(x)
    reflected = t @ torch.tensor(linear).T + torch.tensor(offset)
    expected = 0.5 * (
        forward(t) + forward(reflected) @ torch.tensor(joints).T + torch.tensor(action_offset)
    )
    np.testing.assert_allclose(Actor(arrays)(x), expected.detach().numpy(), rtol=1e-12, atol=1e-12)


def test_safety_bounds_targets_and_stops() -> None:
    safety = Safety(
        SafetyConfig(max_target_rate=10.0), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 0.02
    )
    safety.reset(np.zeros(2))
    np.testing.assert_allclose(safety.bound(np.array([5.0, -0.1])), [0.2, -0.1])
    upright = np.array([0.0, 0.0, -1.0])
    assert safety.stop_reason(upright, np.array([40.0]), False) is None
    tilted = np.array([np.sin(0.7), 0.0, -np.cos(0.7)])
    assert safety.stop_reason(tilted, np.array([40.0]), False) == "fall"
    assert safety.stop_reason(upright, np.array([80.0]), False) == "temperature"
    assert safety.stop_reason(upright, np.array([40.0]), True) == "button"
