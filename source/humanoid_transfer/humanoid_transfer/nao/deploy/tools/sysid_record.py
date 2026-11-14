"""Excite each leg joint and record how it follows, for fitting the drives offline.

**The robot hangs in its harness, feet off the ground.** Every joint first goes
stiff where it is (a ramp); then each joint in turn follows, around where it
stood:

* steps of ``±amplitude`` held ``hold_s`` each -- latency, rise, overshoot,
  and the torque ceiling where a step saturates;
* a chirp of ``amplitude`` from 0.3 to 6 Hz over ``chirp_s`` -- the bandwidth.

Every LoLA cycle is recorded: the target sent, the measured position, the motor
current and temperature (``nao/scripts/fit_actuators.py`` reads it)::

    python3 -m deploy.tools.sysid_record --lola --output sysid.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .. import lola
from .common import add_backend_args, backend_from, save, stamp

LEG = ("LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll")


def signal(kind: str, t: np.ndarray, amplitude: float, hold: float, chirp: float) -> np.ndarray:
    if kind == "steps":
        phase = np.floor(t / hold).astype(int) % 4
        return amplitude * np.array([0.0, 1.0, 0.0, -1.0])[phase]
    f0, f1 = 0.3, 6.0
    k = (f1 - f0) / chirp
    return amplitude * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def excite(backend, joint: str, kind: str, seconds: float, base: dict, stiffness: float, **shape):
    """Drive ``joint`` with ``kind`` around ``base[joint]``, every other joint held at ``base``.
    Returns, per LoLA cycle: time, the joint's target, every joint's position, current and
    temperature."""
    rows = {"time": [], "target": [], "position": [], "current": [], "temperature": []}
    start = time.perf_counter()
    while True:
        reading = backend.read()
        t = time.perf_counter() - start
        if t > seconds:
            break
        target = base[joint] + float(signal(kind, np.array(t), **shape))
        backend.write({**base, joint: target}, stiffness)
        rows["time"].append(t)
        rows["target"].append(target)
        rows["position"].append(reading.position)
        rows["current"].append(reading.current)
        rows["temperature"].append(reading.temperature)
    return {k: np.asarray(v) for k, v in rows.items()}


def ramp_to_stiff(backend, seconds: float, stiffness: float) -> dict:
    reading = backend.read()
    base = dict(zip(lola.JOINTS, (float(v) for v in reading.position), strict=True))
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        backend.read()
        backend.write(base, stiffness * min(1.0, (time.perf_counter() - start) / seconds))
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_backend_args(parser)
    parser.add_argument("--joints", nargs="+", default=list(LEG))
    parser.add_argument("--amplitude", type=float, default=0.15)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--chirp-s", type=float, default=12.0)
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    backend = backend_from(args)
    base = ramp_to_stiff(backend, 1.5, args.stiffness)
    shape = {"amplitude": args.amplitude, "hold": args.hold_s, "chirp": args.chirp_s}
    record = {"joints": np.array(args.joints), "base": np.array([base[n] for n in lola.JOINTS])}
    for joint in args.joints:
        for kind, seconds in (("steps", 4 * args.hold_s * 2), ("chirp", args.chirp_s)):
            print(f"{joint}: {kind}")
            rows = excite(backend, joint, kind, seconds, base, args.stiffness, **shape)
            for key, value in rows.items():
                record[f"{joint}/{kind}/{key}"] = value
            # Back to rest between excitations.
            settle = time.perf_counter()
            while time.perf_counter() - settle < 0.5:
                backend.read()
                backend.write(base, args.stiffness)
    output = save(args.output or f"logs/sysid_{stamp()}.npz", **record, cycle=lola.CYCLE)
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
