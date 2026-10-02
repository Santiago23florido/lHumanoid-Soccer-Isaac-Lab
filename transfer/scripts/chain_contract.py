"""Run a saved contract generator on its own plans and compare its gait with the teacher's.

A plan is checked on its own when the generator is trained; a closed loop
chains plans, each drawn from the history the previous ones produced. With
exact tracking that chain *is* the body's gait, so it is measured before any
body is trained on it: stride, the threshold-free weight-shift rhythm, and the
stance and swing segments, against the teacher's recorded gait.

Usage::

    python transfer/scripts/chain_contract.py logs/contract/walk_warp/generator.pt \
        --output logs/contract/chain_walk_warp.json
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
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    chained_gait,
    gait_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path)
    parser.add_argument("--gait", type=Path, default=REPO_ROOT / "g1/rollouts/teacher.npz")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--commit", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    contract = load_contract(args.contract, args.device)
    generator = contract.generator
    data = np.load(args.gait)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    dt_star = float(units.to_dimensionless(float(data["control_dt"]), TIME))
    z = data["skill_state"].astype(np.float32)
    start = torch.as_tensor(z[100 : 100 + generator.history], device=args.device)
    start = generator.skill_norm(start.transpose(0, 1))
    gait = load_streams([args.gait]).to(args.device)
    command = gait.command.mean(dim=0, keepdim=True).expand(len(start), 3)
    commit = args.commit or generator.commit

    runs = [
        chained_gait(generator, start, command, contract.sigma, args.steps, commit, dt_star, seed)
        for seed in range(args.seeds)
    ]
    reference = gait_reference(z[100 + generator.history :].astype(float), dt_star)

    def mean(key: str) -> float:
        return float(np.mean([r["signature"][key] for r in runs]))

    summary = {
        "stride_period": mean("stride_period"),
        "stride_length": mean("stride_length"),
        "rhythm_period": float(np.mean([r["rhythm_period"] for r in runs])),
        "swing_median": float(np.mean([r["segments"]["swing"]["median"] for r in runs])),
        "stance_median": float(np.mean([r["segments"]["stance"]["median"] for r in runs])),
    }
    teacher = {
        "stride_period": reference["signature"]["stride_period"],
        "stride_length": reference["signature"]["stride_length"],
        "rhythm_period": reference["rhythm_period"],
        "swing_median": reference["segments"]["swing"]["median"],
        "stance_median": reference["segments"]["stance"]["median"],
    }
    deviation = {k: abs(summary[k] - teacher[k]) / teacher[k] for k in summary}
    report = {
        "contract": str(args.contract),
        "flow_solver": generator.flow_solver,
        "flow_steps": generator.flow_steps,
        "commit": commit,
        "steps": args.steps,
        "seeds": args.seeds,
        "summary": summary,
        "teacher": teacher,
        "deviation": deviation,
        "runs": runs,
        "teacher_detail": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key in summary:
        print(
            f"{key:14s} chain {summary[key]:.3f}  teacher {teacher[key]:.3f}  "
            f"dev {deviation[key]:.3f}"
        )
    print(f"stable {[r['environments_stable'] for r in runs]} of {runs[0]['environments']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
