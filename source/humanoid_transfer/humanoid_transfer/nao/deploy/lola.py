"""LoLA, the NAO V6's low-level interface (NAOqi 2.8): its socket, its message, its units.

LoLA serves one MessagePack map of sensor readings every 12 ms on the Unix
socket ``/tmp/robocup`` and reads one map of actuator commands back. Joint
arrays have 25 entries in :data:`JOINTS`' order -- the right hip yaw-pitch is
not among them: one motor drives both. Force sensors are in kilograms, angles
in radians, the gyroscope in rad/s.

This layout is LoLA's as documented for NAOqi 2.8 and used by RoboCup teams
(B-Human, ``nao_lola``). ``nao/deploy/tools/check_sensors.py`` prints the
robot's own keys and lengths on the first connection, to be checked against
:data:`SENSOR_KEYS` before anything moves.
"""

from __future__ import annotations

import socket
import struct
import time

import numpy as np

from .msgpack_lite import _unpack, packb, unpackb

SOCKET = "/tmp/robocup"
CYCLE = 0.012
"""LoLA's period, seconds."""

JOINTS = (
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
    "LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
    "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw",
    "LHand", "RHand",
)  # fmt: skip
"""The order of every 25-entry joint array LoLA reads and writes."""

FSR = ("LFrontLeft", "LFrontRight", "LRearLeft", "LRearRight",
       "RFrontLeft", "RFrontRight", "RRearLeft", "RRearRight")  # fmt: skip
"""The order of the 8 force-sensor readings, kilograms."""

TOUCH = (
    "ChestBoard/Button", "Head/Touch/Front", "Head/Touch/Middle", "Head/Touch/Rear",
    "LFoot/Bumper/Left", "LFoot/Bumper/Right", "LHand/Touch/Back", "LHand/Touch/Left",
    "LHand/Touch/Right", "RFoot/Bumper/Left", "RFoot/Bumper/Right", "RHand/Touch/Back",
    "RHand/Touch/Left", "RHand/Touch/Right",
)  # fmt: skip

SENSOR_KEYS = {
    "Position": 25, "Stiffness": 25, "Temperature": 25, "Current": 25, "Status": 25,
    "Accelerometer": 3, "Gyroscope": 3, "Angles": 2, "FSR": 8, "Touch": 14,
    "Battery": 4, "Sonar": 2, "RobotConfig": 4,
}  # fmt: skip
"""Each key LoLA is expected to send, and its length."""


class Reading:
    """One LoLA message, as arrays."""

    def __init__(self, message: dict) -> None:
        self.message = message
        self.position = np.asarray(message["Position"], dtype=np.float64)
        self.stiffness = np.asarray(message.get("Stiffness", [0.0] * 25), dtype=np.float64)
        self.temperature = np.asarray(message.get("Temperature", [0.0] * 25), dtype=np.float64)
        self.current = np.asarray(message.get("Current", [0.0] * 25), dtype=np.float64)
        self.status = np.asarray(message.get("Status", [0] * 25), dtype=np.float64)
        self.accelerometer = np.asarray(message.get("Accelerometer", [0.0] * 3), dtype=np.float64)
        self.gyroscope = np.asarray(message["Gyroscope"], dtype=np.float64)
        self.angles = np.asarray(message["Angles"], dtype=np.float64)
        self.fsr = np.asarray(message.get("FSR", [0.0] * 8), dtype=np.float64)
        self.touch = np.asarray(message.get("Touch", [0.0] * 14), dtype=np.float64)
        self.battery = np.asarray(message.get("Battery", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)

    def joint(self, names) -> np.ndarray:
        """The measured angles of ``names``; a right hip yaw-pitch reads the left's."""
        index = {n: i for i, n in enumerate(JOINTS)}
        index["RHipYawPitch"] = index["LHipYawPitch"]
        return self.position[[index[n] for n in names]]

    def gravity(self) -> np.ndarray:
        """Gravity's direction in the torso frame, from the IMU's filtered roll and pitch."""
        roll, pitch = self.angles
        return np.array(
            [np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)]
        )

    def foot_loads(self, mass: float) -> np.ndarray:
        """Each foot's load in units of the robot's weight (``mass`` in kg)."""
        return np.array([self.fsr[:4].sum(), self.fsr[4:].sum()]) / mass

    @property
    def chest_button(self) -> bool:
        return bool(self.touch[0] > 0.5)


def command(positions: dict[str, float], stiffness: dict[str, float] | float, base=None) -> dict:
    """A LoLA actuator message: every joint's target and stiffness (0 to 1).

    Joints not in ``positions`` keep ``base`` (a 25-entry array, e.g. the last reading);
    a right hip yaw-pitch target is ignored (the left's drives both).
    """
    target = list(np.zeros(25) if base is None else np.asarray(base, dtype=np.float64))
    for name, value in positions.items():
        if name in JOINTS:
            target[JOINTS.index(name)] = float(value)
    if isinstance(stiffness, dict):
        gains = [float(stiffness.get(name, 0.0)) for name in JOINTS]
    else:
        gains = [float(stiffness)] * 25
    return {"Position": [float(v) for v in target], "Stiffness": gains}


class LolaSocket:
    """A blocking client of LoLA's socket: :meth:`read` a message, :meth:`write` one back."""

    def __init__(self, path: str = SOCKET, timeout: float = 1.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self._buffer = b""

    def read_raw(self) -> bytes:
        """The newest complete message's bytes (older ones queued behind it are dropped)."""
        while True:
            messages, self._buffer = split_messages(self._buffer)
            if messages:
                return messages[-1]
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("LoLA closed the socket")
            self._buffer += chunk

    def read(self) -> Reading:
        return Reading(unpackb(self.read_raw()))

    def write(self, message: dict) -> None:
        self.sock.sendall(packb(message))

    def write_raw(self, data: bytes) -> None:
        self.sock.sendall(data)

    def close(self) -> None:
        self.sock.close()


def split_messages(data: bytes) -> tuple[list[bytes], bytes]:
    """The complete messages at the start of ``data``, and the incomplete rest."""
    view = memoryview(data)
    start, messages = 0, []
    while start < len(data):
        try:
            _, end = _unpack(view, start)
        except (IndexError, struct.error, UnicodeDecodeError):
            break
        messages.append(bytes(view[start:end]))
        start = end
    return messages, data[start:]


def wait_cycles(count: int) -> None:
    time.sleep(count * CYCLE)


__all__ = [
    "CYCLE",
    "FSR",
    "JOINTS",
    "LolaSocket",
    "Reading",
    "SENSOR_KEYS",
    "SOCKET",
    "TOUCH",
    "command",
    "split_messages",
]
