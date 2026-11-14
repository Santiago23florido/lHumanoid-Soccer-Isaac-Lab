"""Bridge LoLA's socket to a laptop over UDP, for a robot without Python 3 and NumPy.

Runs on the robot (Python 2.7 or 3, standard library only)::

    python relay.py --laptop 192.168.1.10

Every LoLA message (12 ms) is forwarded to the laptop as it is; the laptop's
latest command is written back to LoLA every cycle. If the laptop has sent
nothing for ``--silence`` seconds every joint goes limp (stiffness 0): a run
that loses its controller hangs in its harness instead of fighting it.

On the laptop, ``RelayBackend`` (``backends.py``) is the other end.
"""

import argparse
import socket
import time

LIMP = (
    b"\x82\xa8Position\xdc\x00\x19" + b"\xca\x00\x00\x00\x00" * 25
    + b"\xa9Stiffness\xdc\x00\x19" + b"\xca\x00\x00\x00\x00" * 25
)  # fmt: skip
"""MessagePack of {"Position": [0.0] * 25, "Stiffness": [0.0] * 25}."""


def main():
    parser = argparse.ArgumentParser(description="LoLA <-> UDP relay")
    parser.add_argument("--laptop", required=True)
    parser.add_argument("--port", type=int, default=10500)
    parser.add_argument("--socket", default="/tmp/robocup")
    parser.add_argument("--silence", type=float, default=0.2)
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("", args.port))
    udp.setblocking(False)
    laptop = (args.laptop, args.port + 1)
    lola = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lola.connect(args.socket)
    command, heard = LIMP, 0.0
    cycles = 0
    while True:
        message = lola.recv(4096)
        if not message:
            break
        try:
            udp.sendto(message, laptop)
        except socket.error:  # noqa: UP024 -- Python 2.7 on the robot
            pass
        while True:
            try:
                data, sender = udp.recvfrom(65536)
            except socket.error:  # noqa: UP024
                break
            if data != b"hello":
                command, heard = data, time.time()
        if time.time() - heard > args.silence:
            command = LIMP
        lola.sendall(command)
        cycles += 1


if __name__ == "__main__":
    main()
