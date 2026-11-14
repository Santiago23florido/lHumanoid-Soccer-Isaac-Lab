"""Where a :class:`~.session.Session` reads sensors and sends targets.

Every backend has ``read() -> Reading`` (the most recent sensor message, as
:class:`~.lola.Reading`) and ``write(positions, stiffness)`` (targets by joint
name, stiffness 0 to 1, as one number or by joint name).

* :class:`LolaBackend` -- on the robot, LoLA's socket, served by a thread that
  answers every 12 ms cycle with the latest command, so the policy's 20 ms step
  and LoLA's cycle need not line up;
* :class:`RelayBackend` -- on a laptop, the same messages over UDP from
  ``relay.py`` running on the robot (for a robot without Python 3 and NumPy).
"""

from __future__ import annotations

import socket
import threading
import time

from .lola import SOCKET, LolaSocket, Reading, command, split_messages
from .msgpack_lite import packb, unpackb

LIMP = packb({"Position": [0.0] * 25, "Stiffness": [0.0] * 25})


class _Latest:
    """The newest sensor message and the command to answer with, shared with a thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fresh = threading.Event()
        self.raw: bytes | None = None
        self.count = 0
        self.command = LIMP
        self.last_position = None

    def put_reading(self, raw: bytes) -> None:
        with self.lock:
            self.raw = raw
            self.count += 1
        self.fresh.set()

    def reading(self, timeout: float) -> Reading:
        if not self.fresh.wait(timeout):
            raise TimeoutError("no sensor message from the robot")
        with self.lock:
            raw = self.raw
            self.fresh.clear()
        reading = Reading(unpackb(raw))
        self.last_position = reading.position
        return reading

    def put_command(self, positions: dict, stiffness) -> None:
        message = packb(command(positions, stiffness, base=self.last_position))
        with self.lock:
            self.command = message


class LolaBackend:
    def __init__(self, path: str = SOCKET, timeout: float = 0.5) -> None:
        self.socket = LolaSocket(path)
        self.latest = _Latest()
        self.timeout = timeout
        self.running = True
        self.cycles = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while self.running:
            raw = self.socket.read_raw()
            self.latest.put_reading(raw)
            with self.latest.lock:
                message = self.latest.command
            self.socket.write_raw(message)
            self.cycles += 1

    def read(self) -> Reading:
        return self.latest.reading(self.timeout)

    def write(self, positions: dict, stiffness) -> None:
        self.latest.put_command(positions, stiffness)

    def close(self) -> None:
        self.latest.put_command({}, 0.0)
        time.sleep(0.05)
        self.running = False
        self.socket.close()


class RelayBackend:
    """LoLA's messages through ``relay.py`` on the robot, over UDP."""

    def __init__(self, robot: str, port: int = 10500, timeout: float = 0.5) -> None:
        self.address = (robot, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port + 1))
        self.sock.settimeout(0.2)
        self.latest = _Latest()
        self.timeout = timeout
        self.running = True
        self.sock.sendto(b"hello", self.address)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        buffer = b""
        while self.running:
            try:
                raw, _ = self.sock.recvfrom(65536)
            except socket.timeout:  # noqa: UP041 -- an older Python 3 on the robot
                self.sock.sendto(b"hello", self.address)
                continue
            # The relay forwards LoLA's reads as they come; a message may span two.
            messages, buffer = split_messages(buffer + raw)
            if messages:
                self.latest.put_reading(messages[-1])

    def read(self) -> Reading:
        return self.latest.reading(self.timeout)

    def write(self, positions: dict, stiffness) -> None:
        self.latest.put_command(positions, stiffness)
        with self.latest.lock:
            message = self.latest.command
        self.sock.sendto(message, self.address)

    def close(self) -> None:
        self.sock.sendto(LIMP, self.address)
        self.running = False
        self.sock.close()


__all__ = ["LIMP", "LolaBackend", "RelayBackend"]
