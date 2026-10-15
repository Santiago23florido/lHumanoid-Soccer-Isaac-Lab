"""How far a contract generator's own chain walks from its teacher, with its standard error.

The measurement ``train_contract.py`` reports for a new generator, for one that
already exists: chained from 32 of the teacher's histories at the teacher's
command for 900 steps, its stride period and stride length against the
teacher's episodes, and the bootstrap standard error of the larger relative
deviation.

Usage::

    python transfer/scripts/chain_fidelity.py logs/contract/rough_dagger2/generator.pt \
        --gait g1/rollouts/rough_play_forward.npz
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

from humanoid_transfer.common.similarity import TIME, UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import chain_fidelity  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path)
    parser.add_argument("--gait", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    contract = load_contract(args.contract, args.device)
    generator = contract.generator
    archive = np.load(args.gait)
    units = UnitSystem(float(archive["leg_length"]), float(archive["total_mass"]))
    dt_star = float(units.to_dimensionless(float(archive["control_dt"]), TIME))
    gait = load_streams([args.gait]).to(args.device)
    gait_z = archive["skill_state"].astype(np.float32)
    start = torch.as_tensor(gait_z[100 : 100 + generator.history], device=args.device)
    start = generator.skill_norm(start.transpose(0, 1))
    command = gait.command[: start.shape[0]].mean(dim=0, keepdim=True).expand(len(start), 3)
    with torch.no_grad():
        measured = chain_fidelity(
            generator,
            start,
            command,
            contract.sigma,
            900,
            generator.commit,
            dt_star,
            gait_z[100 + generator.history :].astype(float),
            args.seed,
        )
    measured["contract"] = str(args.contract)
    print(
        f"{args.contract}: score {measured['score']:.3f} +- {measured['score_se']:.3f} "
        f"(period {measured['deviation']['stride_period']:.3f}, "
        f"stride {measured['deviation']['stride_length']:.3f}, chains {measured['chains']})"
    )
    if args.output is not None:
        args.output.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
