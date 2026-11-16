"""Check the robot's sensors against what the runtime assumes, before anything moves.

With the robot standing still (stiffness as NAOqi left it, or held by hand):

1. the message's keys and array lengths against ``lola.SENSOR_KEYS``;
2. the IMU's roll and pitch (``Angles``) against the accelerometer's gravity: the
   runtime takes gravity from the angles (``Reading.gravity``), so their axes and
   signs must agree with the accelerometer's measured direction;
3. the gyroscope's bias and noise at rest, all three axes (LoLA's documentation
   once said the z axis was not available);
4. the force sensors' total against the robot's weight, and each foot's share;
5. every joint's reading, and how far the legs are from the bundle's crouch.

Prints a report and saves the readings (``--output``)::

    python3 -m deploy.tools.check_sensors --lola --seconds 5 --bundle bundle
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .. import lola
from .common import add_backend_args, backend_from, collect, save, stamp


def report(readings: list, mass: float | None, crouch: dict | None) -> dict:
    first = readings[0].message
    keys = {}
    for key, length in lola.SENSOR_KEYS.items():
        got = first.get(key)
        keys[key] = (
            "missing" if got is None else ("ok" if len(got) == length else f"{len(got)} values")
        )
    extra = sorted(set(first) - set(lola.SENSOR_KEYS))
    accel = np.stack([r.accelerometer for r in readings])
    angles_gravity = np.stack([r.gravity() for r in readings])
    norm = np.linalg.norm(accel, axis=-1, keepdims=True)
    # At rest the accelerometer reads the support force: minus gravity.
    accel_gravity = -accel / np.where(norm > 0, norm, 1.0)
    gyro = np.stack([r.gyroscope for r in readings])
    fsr = np.stack([r.fsr for r in readings])
    position = np.stack([r.position for r in readings])
    out = {
        "readings": len(readings),
        "keys": keys,
        "extra_keys": extra,
        "accelerometer_norm": float(norm.mean()),
        "gravity_from_angles": angles_gravity.mean(axis=0).tolist(),
        "gravity_from_accelerometer": accel_gravity.mean(axis=0).tolist(),
        "angle_between_deg": float(
            np.degrees(np.arccos(np.clip((angles_gravity * accel_gravity).sum(-1), -1, 1))).mean()
        ),
        "gyro_bias": gyro.mean(axis=0).tolist(),
        "gyro_noise": gyro.std(axis=0).tolist(),
        "fsr_total_kg": float(fsr.sum(axis=1).mean()),
        "fsr_left_share": float(fsr[:, :4].sum(1).mean() / max(fsr.sum(axis=1).mean(), 1e-9)),
        "fsr_noise_kg": fsr.std(axis=0).tolist(),
        "joints": {n: float(v) for n, v in zip(lola.JOINTS, position.mean(axis=0), strict=True)},
        "joint_noise": float(position.std(axis=0).max()),
        "temperature_max": float(np.stack([r.temperature for r in readings]).max()),
        "battery": readings[-1].battery.tolist(),
    }
    if mass is not None:
        out["fsr_total_over_mass"] = out["fsr_total_kg"] / mass
    if crouch is not None:
        out["from_crouch"] = {
            n: out["joints"][n] - v for n, v in crouch.items() if n in out["joints"]
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_backend_args(parser)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--bundle", default=None, help="A deployment bundle: its mass and crouch.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    mass = crouch = None
    if args.bundle is not None:
        from ..controller import Bundle

        bundle = Bundle(args.bundle)
        mass = bundle.model.total_mass
        crouch = dict(zip(bundle.config["actuated"], bundle.config["default_pos"], strict=True))
    backend = backend_from(args)
    readings = collect(backend, args.seconds)
    result = report(readings, mass, crouch)
    print(json.dumps(result, indent=1))
    output = args.output or f"logs/check_sensors_{stamp()}.npz"
    save(
        output,
        position=[r.position for r in readings],
        gyro=[r.gyroscope for r in readings],
        angles=[r.angles for r in readings],
        accelerometer=[r.accelerometer for r in readings],
        fsr=[r.fsr for r in readings],
        temperature=[r.temperature for r in readings],
        report=json.dumps(result),
    )
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
