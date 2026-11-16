"""Measure the latency from a joint target to the joint's response.

**The robot hangs in its harness, feet off the ground.** One joint (the knee by
default) follows a smooth random signal around where it stood; the latency is
the lag that best lines the measured motion up with the targets
(cross-correlation of their velocities, refined to a fraction of a cycle).
It is the delay the simulator's actions model (``delay_ms`` in
``humanoid_transfer.nao.sim2real``), seen from the robot's side of LoLA::

    python3 -m deploy.tools.measure_delay --lola --joint LKneePitch
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import lola
from .common import add_backend_args, backend_from, save, stamp
from .sysid_record import ramp_to_stiff


def latency(time: np.ndarray, target: np.ndarray, measured: np.ndarray, max_lag: float = 0.15):
    """The lag (s) maximising the correlation of the target's and the response's velocities,
    on a uniform grid of LoLA's cycle, refined by a parabola through the peak."""
    dt = float(np.median(np.diff(time)))
    grid = np.arange(time[0], time[-1], dt)
    a = np.gradient(np.interp(grid, time, target))
    b = np.gradient(np.interp(grid, time, measured))
    a, b = a - a.mean(), b - b.mean()
    lags = np.arange(int(max_lag / dt) + 1)
    corr = np.array([np.dot(a[: len(a) - k], b[k:]) / (len(a) - k) for k in lags])
    k = int(np.argmax(corr))
    shift = 0.0
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return (k + shift) * dt, corr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_backend_args(parser)
    parser.add_argument("--joint", default="LKneePitch")
    parser.add_argument("--amplitude", type=float, default=0.12)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    # A sum of sines from 0.5 to 4 Hz: smooth, rich, never periodic within the run.
    freqs = rng.uniform(0.5, 4.0, size=6)
    phases = rng.uniform(0, 2 * np.pi, size=6)
    backend = backend_from(args)
    base = ramp_to_stiff(backend, 1.5, args.stiffness)
    index = lola.JOINTS.index(args.joint)
    import time as clock

    rows = {"time": [], "target": [], "position": []}
    start = clock.perf_counter()
    while clock.perf_counter() - start < args.seconds:
        reading = backend.read()
        t = clock.perf_counter() - start
        offset = args.amplitude * np.sin(2 * np.pi * freqs * t + phases).sum() / np.sqrt(6)
        target = base[args.joint] + offset
        backend.write({**base, args.joint: target}, args.stiffness)
        rows["time"].append(t)
        rows["target"].append(target)
        rows["position"].append(reading.position[index])
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    lag, corr = latency(arrays["time"], arrays["target"], arrays["position"])
    result = {"joint": args.joint, "latency_ms": 1000 * lag, "cycles": len(arrays["time"])}
    print(json.dumps(result))
    output = save(args.output or f"logs/delay_{stamp()}.npz", **arrays, correlation=corr,
                  result=json.dumps(result))  # fmt: skip
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
