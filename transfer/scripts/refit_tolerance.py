"""Refit a saved contract's tolerances, and optionally change how it samples, without retraining.

A contract's tolerance is a measurement of the trained generator (its spread
of plans on held-out states), not a parameter of it. When the measurement
grows -- the trajectory integrals were added after the walk's generator was
trained -- or the sampler changes, the tolerances are refitted here on the
same held-out walking and the contract saved in place.

Usage::

    python transfer/scripts/refit_tolerance.py logs/contract/walk_big/generator.pt \
        --solver midpoint --flow-steps 8
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    contract_steps,
    fit_tolerance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "g1/rollouts/teacher_dart.npz")
    parser.add_argument("--val-envs", type=int, default=16)
    parser.add_argument("--solver", choices=("euler", "midpoint"), default=None)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    saved = torch.load(args.contract, map_location="cpu", weights_only=False)
    contract = load_contract(args.contract, args.device)
    generator = contract.generator
    dims = ContractDims(**saved["dims"])
    if args.solver is not None:
        generator.flow_solver = args.solver
        dims = replace(dims, flow_solver=args.solver)
    if args.flow_steps is not None:
        generator.flow_steps = args.flow_steps
        dims = replace(dims, flow_steps=args.flow_steps)

    streams = load_streams([args.data]).to(args.device)
    envs = int(np.load(args.data)["done"].shape[1])
    planned = contract_steps(streams, generator.horizon)
    held_out = planned[(streams.env[planned] >= envs - args.val_envs)]
    tolerance = fit_tolerance(generator, streams, held_out, contract.sigma)
    save_contract(
        args.contract,
        contract,
        dims,
        ContractClock(**saved["clock"]),
        saved["encoder_input"] or G1_OBSERVATION.encoder_size,
    )
    print(f"tolerance by k {tolerance.mean(dim=-1).cpu().numpy().round(3)}")
    print(f"integral tolerance by k {generator.integral_tolerance.cpu().numpy().round(4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
