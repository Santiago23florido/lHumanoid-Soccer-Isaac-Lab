"""Fit a body's encoder on its own captures, and give a task's contract to it.

The encoder ``E_b`` estimates the body's ``z*`` from what the body senses
(``ReferenceCommand.encoder_input``: base velocities, gravity, joint positions and
velocities, last action -- never the command). It belongs to the body, is trained
on the anchor alone (mean squared error to the true ``z*`` in the contract's
normalised units, as the G1's was), and is fitted on the states the body's own
controller visits: captures of its tracker by ``capture_contract.py``, which
record ``encoder_input`` next to the true ``skill_state``.

The contract keeps its generator and the noise it was trained under; the new
encoder's held-out error is reported next to it.

Usage::

    python transfer/scripts/train_body_encoder.py logs/contract/walk_dagger2e/generator.pt \
        --data nao/rollouts/track_walk_physical.npz --data nao/rollouts/track_walk_auto.npz \
        --output logs/contract/walk_nao/generator.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    ContractClock,
    ContractDims,
    load_contract,
    save_contract,
)
from humanoid_transfer.transfer.latent.memory import StateEncoder  # noqa: E402


def load(paths: list[Path], val_envs: int) -> tuple[torch.Tensor, ...]:
    """``(inputs, skill, held_out)`` over every step that follows a step of the same episode."""
    inputs, skills, held = [], [], []
    for path in paths:
        data = np.load(path)
        done = data["done"]
        # A step right after a reset is read before the simulator applied the reset.
        fresh = np.zeros_like(done)
        fresh[1:] = done[:-1]
        keep = ~fresh
        envs = done.shape[1]
        held_out = np.zeros_like(done)
        held_out[:, envs - val_envs :] = True
        inputs.append(data["encoder_input"][keep])
        skills.append(data["skill_state"][keep])
        held.append(held_out[keep])
    return (
        torch.as_tensor(np.concatenate(inputs), dtype=torch.float32),
        torch.as_tensor(np.concatenate(skills), dtype=torch.float32),
        torch.as_tensor(np.concatenate(held)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path, help="The task's contract (generator, units).")
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--val-envs", type=int, default=4, help="Held out of every capture.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    saved = torch.load(args.contract, map_location="cpu", weights_only=False)
    contract = load_contract(args.contract, args.device)
    dims = ContractDims(**saved["dims"])
    inputs, skill, held_out = load(args.data, args.val_envs)
    inputs, skill = inputs.to(args.device), skill.to(args.device)
    held_out = held_out.to(args.device)
    target = contract.generator.skill_norm(skill)
    train = (~held_out).nonzero().squeeze(-1)
    validation = held_out.nonzero().squeeze(-1)

    encoder = StateEncoder(inputs.shape[-1], dims.skill, dims.hidden).to(args.device)
    encoder.obs_norm.fit(inputs[train])
    optimiser = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    per_epoch = max(1, math.ceil(len(train) / args.batch_size))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * per_epoch)
    for _ in range(args.epochs):
        order = train[torch.randperm(len(train), device=args.device)]
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            loss = F.mse_loss(encoder(inputs[batch]), target[batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
    encoder.eval()
    with torch.no_grad():
        error = encoder(inputs[validation]) - target[validation]
    sigma = error.pow(2).mean(dim=0).sqrt()

    contract.encoder = encoder
    contract.meta = {
        **contract.meta,
        "encoder_captures": [str(p) for p in args.data],
        "encoder_sigma_after": sigma.tolist(),
    }
    save_contract(
        args.output,
        contract,
        dims,
        ContractClock(**saved["clock"]),
        inputs.shape[-1],
    )
    report = {
        "contract": str(args.contract),
        "output": str(args.output),
        "captures": [str(p) for p in args.data],
        "encoder_input": int(inputs.shape[-1]),
        "train_steps": len(train),
        "validation_steps": len(validation),
        "final_loss": float(loss),
        "sigma_median": float(sigma.median()),
        "sigma": sigma.tolist(),
        "contract_sigma_median": float(contract.sigma.median()),
    }
    if args.results is not None:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.output}: encoder on {len(train)} steps, held-out error median "
        f"{report['sigma_median']:.3f} (the contract's noise {report['contract_sigma_median']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
