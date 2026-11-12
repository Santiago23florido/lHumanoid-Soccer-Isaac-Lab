"""A tracker training's tracking errors, read from its TensorBoard log.

For each run named, the mean over its first and its last ``--window`` logged iterations of the
command term's errors (``Metrics/base_velocity/...``): on the task plans (``error_task``), on
the procedural family (``error_procedural``), and overall (``contract_error``); with the
contract reward, the episode length and, when the run drew several tasks, each task's share.
Errors are summed over an episode, so two runs compare only at similar episode lengths, which
the output shows.

    python transfer/scripts/training_errors.py nao mt_nao_s31 warm_nao_r1 mt_nao_s32 warm_nao_r1_s32

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KEYS = (
    "Metrics/base_velocity/error_task",
    "Metrics/base_velocity/error_procedural",
    "Metrics/base_velocity/contract_error",
    "Episode_Reward/contract",
    "Train/mean_episode_length",
)


def scalars(run: Path) -> dict[str, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    return {t: np.array([s.value for s in events.Scalars(t)]) for t in events.Tags()["scalars"]}


def summary(body: str, name: str, window: int) -> dict:
    runs = sorted((ROOT / "logs" / "rsl_rl" / f"{body}_track").glob(f"*_{name}"))
    if not runs:
        raise SystemExit(f"no run named {name} for {body}")
    series = scalars(runs[-1])
    keys = [*KEYS, *sorted(k for k in series if k.startswith("Metrics/base_velocity/task_"))]
    return {
        "run": runs[-1].name,
        **{
            k.split("/")[-1]: {
                "first": float(series[k][:window].mean()),
                "last": float(series[k][-window:].mean()),
            }
            for k in keys
            if k in series
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("body", choices=("g1", "nao"))
    parser.add_argument("names", nargs="+", help="run names, as train.py's --run_name")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {name: summary(args.body, name, args.window) for name in args.names}
    for name, row in report.items():
        cells = "  ".join(
            f"{k} {v['first']:.3f} -> {v['last']:.3f}" for k, v in row.items() if k != "run"
        )
        print(f"{name:18s} {cells}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
