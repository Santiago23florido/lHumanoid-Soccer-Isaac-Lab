"""Measure, on the bodies' own walks, what the point-mass biped cannot derive.

The biped of ``transfer/latent/procedural.py`` writes every field of ``z*``.
Most follow from its dynamics; three do not, because a point mass has no
body: the base's tilt (the gravity it reads), the base's angular velocity
beyond the turn, and the centroidal angular momentum. The biped models them
as a lean, the turn rate, and

    k = μ Σ_i (p_i − c) × (ṗ_i − ċ) + ι ψ̇ ẑ

-- the legs' orbital momentum, scaled, plus a yaw inertia. This script fits
``μ`` and ``ι`` by least squares on each body's walk, measures the lean, and
reports what is left: the model's error per component, ``σ_model``, which the
tracking reward adds to the encoder's error as the tolerance on those fields
for procedural plans (a likelihood under the model the plan was written with).

The data are the bodies' own: the G1 teacher's walk, and the NAO's tracker
walking the walk's contract. Nothing here comes from a task the trackers will
be tested on. The fitted constants set the ranges the biped randomises over:
from half the smaller to one and a half times the larger of the two bodies'.

Usage::

    python transfer/scripts/calibrate_procedural.py \
        --body g1=g1/rollouts/teacher_dart.npz --body g1=g1/rollouts/closed_loop_teacher.npz \
        --body nao=nao/rollouts/track_walk_nao2_physical.npz \
        --body nao=nao/rollouts/track_walk_nao2_autonomous.npz \
        --output transfer/results/procedural_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402

SETTLE = 100
"""Steps dropped at the start of every capture: the robot settling, not walking."""

AROUND_RESET = 60
"""Steps dropped before a fall and after any reset."""


def walking_states(path: Path) -> np.ndarray:
    """``(M, 27)``: the capture's ``z*`` away from its start, its resets and its falls."""
    data = np.load(path)
    z = data["skill_state"][SETTLE:]
    done = data["done"][SETTLE:]
    keep = np.ones(done.shape, dtype=bool)
    for t, n in zip(*np.nonzero(done), strict=True):
        keep[max(t - AROUND_RESET, 0) : t + AROUND_RESET, n] = False
    return z[keep]


def fit(z: np.ndarray) -> dict:
    fields = SKILL_LAYOUT.unpack(z)
    feet = fields["foot_position"]
    relative = fields["foot_velocity"] - fields["com_velocity"][:, None, :]
    orbit = np.cross(feet, relative).sum(axis=1)
    momentum = fields["angular_momentum"]
    turn = fields["base_angular_velocity"][:, 2]
    # k_x = μ o_x, k_y = μ o_y, k_z = μ o_z + ι ψ̇: one least-squares problem in (μ, ι).
    design = np.zeros((3 * len(z), 2))
    design[0::3, 0], design[1::3, 0], design[2::3, 0] = orbit[:, 0], orbit[:, 1], orbit[:, 2]
    design[2::3, 1] = turn
    target = momentum.reshape(-1)
    (mu, iota), *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = (design @ np.array([mu, iota])).reshape(-1, 3)
    residual = momentum - predicted
    variance = momentum.var(axis=0)

    gravity = fields["gravity"]
    lean = np.arcsin(np.clip(gravity[:, 0], -1.0, 1.0))
    angular = fields["base_angular_velocity"]
    mean_lean = float(lean.mean())
    model_angular = np.stack(
        [-np.sin(mean_lean) * turn, np.zeros_like(turn), np.cos(mean_lean) * turn], axis=-1
    )
    model_gravity = np.broadcast_to(
        np.array([np.sin(mean_lean), 0.0, -np.cos(mean_lean)]), gravity.shape
    )

    sigma = np.zeros(SKILL_LAYOUT.size)
    sigma[SKILL_LAYOUT.slice_of("angular_momentum")] = np.sqrt((residual**2).mean(axis=0))
    sigma[SKILL_LAYOUT.slice_of("gravity")] = np.sqrt(((gravity - model_gravity) ** 2).mean(axis=0))
    sigma[SKILL_LAYOUT.slice_of("base_angular_velocity")] = np.sqrt(
        ((angular - model_angular) ** 2).mean(axis=0)
    )
    return {
        "samples": len(z),
        "leg_orbit": float(mu),
        "yaw_inertia": float(iota),
        "momentum_r2": [float(v) for v in 1.0 - (residual**2).mean(axis=0) / variance],
        "lean_mean": mean_lean,
        "lean_std": float(lean.std()),
        "sigma_model": [float(v) for v in sigma],
        "data_std": [float(v) for v in z.std(axis=0)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body", action="append", required=True, help="name=path/to/capture.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    captures: dict[str, list[Path]] = {}
    for item in args.body:
        name, _, path = item.partition("=")
        captures.setdefault(name, []).append(REPO_ROOT / path)
    bodies = {}
    for name, paths in captures.items():
        z = np.concatenate([walking_states(path) for path in paths])
        bodies[name] = {"captures": [str(p.relative_to(REPO_ROOT)) for p in paths], **fit(z)}

    def spread(key: str) -> list[float]:
        values = [body[key] for body in bodies.values()]
        return [0.5 * min(values), 1.5 * max(values)]

    names = SKILL_LAYOUT.component_names()
    sigma = np.max([body["sigma_model"] for body in bodies.values()], axis=0)
    report = {
        "bodies": bodies,
        "ranges": {"leg_orbit": spread("leg_orbit"), "yaw_inertia": spread("yaw_inertia")},
        "sigma_model": {n: float(v) for n, v in zip(names, sigma, strict=True) if v > 0},
        "lean_means": {name: body["lean_mean"] for name, body in bodies.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, body in bodies.items():
        print(
            f"{name}: {body['samples']} states, mu {body['leg_orbit']:.3f}, "
            f"iota {body['yaw_inertia']:.4f}, momentum R2 {np.round(body['momentum_r2'], 2)}, "
            f"lean {body['lean_mean']:.3f} +- {body['lean_std']:.3f}"
        )
    print("ranges", report["ranges"])
    print("sigma_model", {k: round(v, 4) for k, v in report["sigma_model"].items()})
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
