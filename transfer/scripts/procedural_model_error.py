"""How far a real gait is from the family's own version of it, field by field.

The tracking reward scores a reached state against a plan at the plan's
precision. For a task's plan that is the encoder's error: the plan is a
commitment the body is held to. A procedural plan is something else -- a
point-mass model's guess at a gait -- and a body that walks that gait for
real does not reproduce the model to the millimetre: its CoM rises and falls
with its legs, its loads ramp with its feet's shape, its swing follows its
joints. Scored at the encoder's precision, those differences dominate the
likelihood (on the published gait's twin, the CoM height alone was 43
tolerances² of a joint error of 15), the precision term is saturated, and
what remains of the reward hardly changes when the body gets the contact
timing right. ``calibrate_procedural.py`` measured this for the three fields
a point mass cannot derive; this measures it for all of them.

A real gait -- the walk teacher's, training data -- is compared with its twin:
the biped walking the gait's own parameters (``procedural_coverage.py``
measures them). Both are cut into strides at the left foot's landings and
averaged per phase; the model's error per component is

    σ_model² = mean over phase of (μ_real − μ_twin)² + mean over phase of var_real,

what the model gets wrong about the gait's shape plus what the real gait
varies by that no model of its parameters predicts. The tracking reward
widens a procedural plan's tolerance to ``sqrt(σ_enc² + σ_model²)`` where the
model's error is the larger.

Usage::

    python transfer/scripts/procedural_model_error.py \
        --capture g1/rollouts/closed_loop_teacher.npz \
        --calibration transfer/results/procedural_calibration.json \
        --output transfer/results/procedural_model_error.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.procedural import (  # noqa: E402
    GaitRanges,
    PointMassBiped,
    standing_skill,
)
from procedural_coverage import load, parameters  # noqa: E402

BINS = 24
THRESHOLD = 0.05
SKIP = 20.0
"""Pendulum times of the twin's run left out: its stand and its start."""


def phase_profiles(z: np.ndarray, done: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per phase bin of the stride (left landing to left landing): mean and variance of ``z*``."""
    load = SKILL_LAYOUT.unpack(z)["foot_load"][..., 0]
    sums = np.zeros((BINS, z.shape[-1]))
    squares = np.zeros((BINS, z.shape[-1]))
    counts = np.zeros(BINS)
    for env in range(z.shape[1]):
        down = load[:, env] > THRESHOLD
        landing = np.flatnonzero(down[1:] & ~down[:-1]) + 1
        for start, stop in zip(landing[:-1], landing[1:], strict=True):
            if done[start:stop, env].any() or stop - start < 4:
                continue
            phase = ((np.arange(start, stop) - start) / (stop - start) * BINS).astype(int)
            np.add.at(sums, phase, z[start:stop, env])
            np.add.at(squares, phase, z[start:stop, env] ** 2)
            np.add.at(counts, phase, 1)
    mean = sums / counts[:, None]
    return mean, squares / counts[:, None] - mean**2


def twin(params: dict, dt_star: float, n: int = 16, seconds: float = 80.0) -> np.ndarray:
    biped = PointMassBiped(
        n, GaitRanges(), body_step=dt_star, substep=dt_star / 4, horizon=1.0, seed=0
    )
    biped.set_offsets(torch.zeros(1, dtype=torch.float64))
    keys = ("forward", "lateral", "turn", "period", "width", "height", "clearance", "lean")
    probe = {k: torch.full((n,), float(params[k])) for k in keys}
    probe["duty"] = torch.tensor([params["duty"]] * n)
    probe["first"] = torch.arange(n) % 2
    probe["duration"] = torch.full((n,), 1e9)
    probe["offset"] = torch.full((n,), 0.5)
    ids = torch.arange(n)
    biped.set_probes(ids, probe)
    biped.anchor(ids, standing_skill(n))
    rows = []
    for _ in range(int(seconds / dt_star)):
        biped.advance()
        rows.append(biped.plan(ids)[2].numpy())
    return np.stack(rows)[int(SKIP / dt_star) :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path, default=REPO_ROOT / "logs/contract/walk_dagger3e/generator.pt"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    z, done, dt_star = load(args.capture)
    params = parameters(z, done, dt_star)
    model = twin(params, dt_star)
    real_mean, real_var = phase_profiles(z.astype(float), done)
    twin_mean, _ = phase_profiles(model.astype(float), np.zeros(model.shape[:2], dtype=bool))
    sigma = np.sqrt(np.nanmean((real_mean - twin_mean) ** 2, axis=0) + np.nanmean(real_var, axis=0))

    calibration = json.loads((REPO_ROOT / args.calibration).read_text(encoding="utf-8"))
    names = SKILL_LAYOUT.component_names()
    contract = load_contract(args.contract)
    encoder = (contract.sigma * contract.generator.skill_norm.std).numpy()
    merged = {}
    for i, name in enumerate(names):
        value = max(float(sigma[i]), calibration["sigma_model"].get(name, 0.0))
        merged[name] = value
    report = {
        "capture": str(args.capture),
        "parameters": params,
        "sigma_shape": {n: float(v) for n, v in zip(names, sigma, strict=True)},
        "sigma_encoder": {n: float(v) for n, v in zip(names, encoder, strict=True)},
        "sigma_model": merged,
        "widened": [n for i, n in enumerate(names) if merged[n] > encoder[i]],
        "ratio": {n: merged[n] / float(encoder[i]) for i, n in enumerate(names)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for i, name in enumerate(names):
        ratio = merged[name] / encoder[i]
        print(f"{name:28s} model {merged[name]:.4f}  encoder {encoder[i]:.4f}  ratio {ratio:5.1f}")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
