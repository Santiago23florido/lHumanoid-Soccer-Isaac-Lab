"""The action noise of a PPO run over its checkpoints: whether it falls, flattens or climbs.

PPO's entropy bonus pushes every action's log std up by a constant, and near convergence it
wins (``g1/tasks/g1_track/agents/rsl_rl_ppo_cfg.py``, ``ENTROPY``). This prints, per run
(``logs/rsl_rl/<body>_track/<stamp>_<tag>``), the mean and largest std of the actor's Gaussian
(``distribution.std_param``) at the first, middle and last checkpoints, and the mean's change
over the run.

    python transfer/scripts/pact_action_std.py h1:contactprec cassie:contactprec
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def checkpoints(run: Path) -> list[Path]:
    found = [p for p in run.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)]
    return sorted(found, key=lambda p: int(p.stem.split("_")[1]))


def action_std(path: Path) -> torch.Tensor:
    state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    return state["distribution.std_param"].float()


def read(body: str, tag: str) -> dict | None:
    runs = sorted((REPO / "logs" / "rsl_rl" / f"{body}_track").glob(f"*_{tag}_{body}"))
    if not runs:
        return None
    files = checkpoints(runs[-1])
    picks = [files[0], files[len(files) // 2], files[-1]]
    stds = [action_std(p) for p in picks]
    return {
        "run": runs[-1].name,
        "checkpoints": [int(p.stem.split("_")[1]) for p in picks],
        "mean": [float(s.mean()) for s in stds],
        "max": [float(s.max()) for s in stds],
        "change": float(stds[-1].mean() - stds[0].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", help="body:tag, e.g. h1:contactprec")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = {}
    for spec in args.runs:
        body, tag = spec.split(":")
        row = read(body, tag)
        report[spec] = row
        if row is None:
            print(f"{spec}: no run")
            continue
        steps = " -> ".join(
            f"{c}: {m:.2f} (max {x:.2f})" for c, m, x in zip(row["checkpoints"], row["mean"], row["max"])
        )
        print(f"{spec} ({row['run']}): action std {steps}; change {row['change']:+.2f}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
