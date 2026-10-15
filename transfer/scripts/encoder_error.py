"""How well a body's encoder reads the states a capture visited, field by field.

The encoder ``E_b`` estimates ``z*`` from what the body senses. Its error on the
body's own states is the precision the tracker's reward and the generator's
history noise assume (``sigma``). On states it never saw -- a crouched gait, a
high swing -- it can err more, and a tracker fed that estimate reads a plan
against a state the body is not in (condition (vi) of skill_space.md §8.1).

For each capture (``capture_contract.py``: ``encoder_input`` next to the true
``skill_state``), this reports the encoder's error in units of the contract's
``sigma`` -- per field, overall, and by CoM height -- with bootstrap standard
errors over environments. It only measures; nothing is trained.

Usage::

    python transfer/scripts/encoder_error.py logs/contract/rough_b3_zs_g1_r3_6194_sym/generator.pt \
        g1/rollouts/zs_r3_6194_sym_walk_truth.npz \
        g1/rollouts/zs_r3_6194_sym_rough_b3_auto_fast_truth.npz
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

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402

HEIGHT_BINS = (0.0, 0.85, 0.90, 0.95, 10.0)
"""CoM height bins, leg lengths: below the family, then its range in thirds and above."""


def bootstrap(per_env: np.ndarray, weights: np.ndarray, draws: int, rng) -> tuple[float, float]:
    """Weighted mean over environments and its bootstrap standard error."""
    total = weights.sum()
    if total == 0:
        return float("nan"), float("nan")
    mean = float((per_env * weights).sum() / total)
    picks = rng.integers(0, len(per_env), size=(draws, len(per_env)))
    w = weights[picks]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (per_env[picks] * w).sum(axis=1) / w.sum(axis=1)
    return mean, float(np.nanstd(means))


def measure(contract, path: Path, device: str, draws: int, rng, recorded: bool) -> dict:
    data = np.load(path)
    done = data["done"]
    fresh = np.zeros_like(done)
    fresh[1:] = done[:-1]
    keep = ~fresh
    inputs = torch.as_tensor(data["encoder_input"], dtype=torch.float32, device=device)
    skill = torch.as_tensor(data["skill_state"], dtype=torch.float32, device=device)
    with torch.no_grad():
        if recorded:
            # What the deployed path read, whatever its estimator (capture_contract.py).
            read = torch.as_tensor(data["estimate"], dtype=torch.float32, device=device)
            estimate = contract.generator.skill_norm(read)
        else:
            estimate = contract.encoder(inputs.flatten(0, 1)).reshape(skill.shape)
        target = contract.generator.skill_norm(skill)
    ratio = ((estimate - target) / contract.sigma.to(device)).cpu().numpy()
    squared = ratio**2
    height = data["skill_state"][..., SKILL_LAYOUT.slice_of("height")][..., 0]

    weights = keep.astype(float).sum(axis=0)
    report: dict = {"capture": str(path), "steps": int(keep.sum())}
    fields = {}
    for field in SKILL_LAYOUT.fields:
        span = SKILL_LAYOUT.slice_of(field.name)
        per_step = squared[..., span].mean(axis=-1)
        per_env = np.where(keep, per_step, 0.0).sum(axis=0) / np.maximum(weights, 1)
        mean, se = bootstrap(per_env, weights, draws, rng)
        fields[field.name] = {"rms": float(np.sqrt(mean)), "mean_square": mean, "se": se}
    report["fields"] = fields
    per_step = squared.mean(axis=-1)
    per_env = np.where(keep, per_step, 0.0).sum(axis=0) / np.maximum(weights, 1)
    mean, se = bootstrap(per_env, weights, draws, rng)
    report["overall"] = {"rms": float(np.sqrt(mean)), "mean_square": mean, "se": se}
    bins = []
    for low, high in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:], strict=True):
        inside = keep & (height >= low) & (height < high)
        count = inside.astype(float).sum(axis=0)
        per_env = np.where(inside, per_step, 0.0).sum(axis=0) / np.maximum(count, 1)
        mean, se = bootstrap(per_env, count, draws, rng)
        bins.append(
            {"height": [low, high], "steps": int(inside.sum()), "mean_square": mean, "se": se}
        )
    report["by_height"] = bins
    report["height_median"] = float(np.median(height[keep]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path, help="A contract carrying the body's encoder.")
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--recorded",
        action="store_true",
        help="Score the estimate each capture recorded instead of re-running the encoder.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    contract = load_contract(args.contract, args.device)
    if contract.encoder is None:
        raise SystemExit(f"{args.contract} carries no encoder")
    contract.encoder.eval()
    rng = np.random.default_rng(args.seed)
    reports = [
        measure(contract, path, args.device, args.draws, rng, args.recorded)
        for path in args.captures
    ]
    names = [field.name for field in SKILL_LAYOUT.fields]
    print(f"encoder of {args.contract}, error in units of its sigma (rms; overall ± se)")
    print(
        f"{'capture':58s} {'height':>6s} {'overall':>14s} "
        + " ".join(f"{n[:9]:>9s}" for n in names)
    )
    for report in reports:
        overall = report["overall"]
        row = " ".join(f"{report['fields'][n]['rms']:9.2f}" for n in names)
        print(
            f"{Path(report['capture']).stem[:58]:58s} {report['height_median']:6.3f} "
            f"{overall['rms']:6.2f} ± {0.5 * overall['se'] / max(overall['rms'], 1e-9):5.2f} {row}"
        )
    if args.output is not None:
        result = {"contract": str(args.contract), "unit": "contract sigma", "captures": reports}
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
