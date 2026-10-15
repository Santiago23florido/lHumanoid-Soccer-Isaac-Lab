"""Give a task's contract the body's current encoder.

The encoder belongs to the body and the generator to the task: when the body's
encoder is refitted (``train_encoder.py``), every task the body runs should read
the body through the new one. A contract stores its encoder's output in the
generator's normalised units, so moving an encoder between contracts whose
normalisations differ is an affine change of units; it is folded into the
encoder's last linear layer, so the estimate in physical units is unchanged.

The task's noise level ``sigma`` -- what its generator was trained under, and the
tolerance its plans are scored at -- is kept.

Usage::

    python transfer/scripts/body_encoder.py logs/contract/flat_h20/generator.pt \
        logs/contract/walk_dagger2e/generator.pt --output logs/contract/flat_h20e/generator.pt
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    ContractClock,
    ContractDims,
    load_contract,
    save_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("task", type=Path, help="The task's contract (its generator is kept).")
    parser.add_argument("body", type=Path, help="A contract carrying the body's current encoder.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.task, map_location="cpu", weights_only=False)
    body_saved = torch.load(args.body, map_location="cpu", weights_only=False)
    task = load_contract(args.task)
    body = load_contract(args.body)
    if body.encoder is None:
        raise ValueError(f"{args.body} carries no encoder")
    encoder = copy.deepcopy(body.encoder)

    # z_task = (z * s_b + m_b - m_t) / s_t: a per-dimension scale and shift.
    source, target = body.generator.skill_norm, task.generator.skill_norm
    scale = source.std / target.std
    shift = (source.mean - target.mean) / target.std
    last = [m for m in encoder.net.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.weight.mul_(scale[:, None])
        last.bias.mul_(scale).add_(shift)

    probe = torch.randn(256, encoder.obs_norm.mean.shape[-1])
    with torch.no_grad():
        physical_body = source.inverse(body.encoder(probe))
        physical_task = target.inverse(encoder(probe))
    # Relative: random inputs drive the estimate far out, where float32 rounds coarsely.
    error = float(((physical_body - physical_task).abs() / (1 + physical_body.abs())).max())
    if error > 1e-4:
        raise RuntimeError(f"the change of units moved the estimate by {error} (relative)")

    task.encoder = encoder
    task.meta = {**task.meta, "encoder_from": str(args.body)}
    save_contract(
        args.output,
        task,
        ContractDims(**saved["dims"]),
        ContractClock(**saved["clock"]),
        # The encoder reads the body's senses: its input is the body's, not the task's.
        body_saved["encoder_input"],
    )
    print(
        f"{args.output}: {args.task} with the encoder of {args.body} "
        f"(units moved, {error:.1e} relative)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
