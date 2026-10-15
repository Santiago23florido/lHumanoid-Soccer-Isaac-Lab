"""Retrain a body's encoder on the states its own controller visits, and put it in the contract.

The encoder estimates ``z*`` from the body's observation, which includes its
last action. Trained on the teacher's rollouts alone, it read the tracker's
actions out of distribution: on the deployed path the G1 tracker's own-plan
error rose from 0.35 to 18 tolerances and it walked backwards. An estimator
belongs to the body and has to be fitted on the body's own state distribution
-- the anchor alone, as before, no task gradient -- so it is refitted here on
the teacher's captures together with captures of the tracker (driven with the
contract fed by the true state, so the robot walks while the data is taken).

The contract keeps its generator and the noise it was trained under; the new
encoder's held-out error is recorded next to it.

Usage::

    python transfer/scripts/train_encoder.py logs/contract/walk_warp/generator.pt \
        --data g1/rollouts/teacher_dart.npz --data g1/rollouts/track_dart.npz \
        --output logs/contract/walk_warp/generator_encoder.pt
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

from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    ContractClock,
    ContractDims,
    load_contract,
    save_contract,
)
from humanoid_transfer.transfer.latent.model import G1_OBSERVATION  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_memory import (  # noqa: E402
    MemoryConfig,
    residual_sigma,
    train_encoder,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--val-envs", type=int, default=8, help="Held out of every capture.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    saved = torch.load(args.contract, map_location="cpu", weights_only=False)
    contract = load_contract(args.contract, args.device)
    streams = load_streams(args.data).to(args.device)
    envs = torch.as_tensor(
        [int(np.load(p)["done"].shape[1]) for p in args.data], device=args.device
    )
    held_out = streams.env >= envs[streams.capture] - args.val_envs
    trustworthy = streams.samples()
    train = trustworthy[~held_out[trustworthy] & ~streams.falling[trustworthy]]
    validation = trustworthy[held_out[trustworthy] & ~streams.falling[trustworthy]]

    before = residual_sigma(
        contract.encoder, contract.generator, G1_OBSERVATION, streams, validation
    )
    encoder, history = train_encoder(
        streams,
        train,
        G1_OBSERVATION,
        contract.generator,
        MemoryConfig(encoder_epochs=args.epochs, seed=args.seed),
    )
    after = residual_sigma(encoder, contract.generator, G1_OBSERVATION, streams, validation)
    per_capture = {}
    for index, path in enumerate(args.data):
        steps = validation[streams.capture[validation] == index]
        per_capture[str(path)] = {
            "before_median": float(
                residual_sigma(
                    contract.encoder, contract.generator, G1_OBSERVATION, streams, steps
                ).median()
            ),
            "after_median": float(
                residual_sigma(encoder, contract.generator, G1_OBSERVATION, streams, steps).median()
            ),
        }

    contract.encoder = encoder
    contract.meta = {
        **contract.meta,
        "encoder_captures": [str(p) for p in args.data],
        "encoder_sigma_after": after.tolist(),
    }
    save_contract(
        args.output,
        contract,
        ContractDims(**saved["dims"]),
        ContractClock(**saved["clock"]),
        saved["encoder_input"],
    )
    report = {
        "contract": str(args.contract),
        "output": str(args.output),
        "captures": [str(p) for p in args.data],
        "train_steps": len(train),
        "validation_steps": len(validation),
        "final_anchor_loss": history[-1]["anchor"],
        "sigma_median_before": float(before.median()),
        "sigma_median_after": float(after.median()),
        "per_capture": per_capture,
    }
    if args.results is not None:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
