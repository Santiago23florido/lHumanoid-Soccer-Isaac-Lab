"""Does the physical history carry the gait's phase? The gate before the memory.

The shared memory is only allowed to read the history of the skill state
``z*`` -- never the body's observations -- so that nothing about the body can
reach the shared skill through it. That restriction is free only if the
physical history carries the timing the policy needs. This script measures it
before anything is trained, on held-out environments of the DART capture,
with three inputs:

(i)   ``z*_t`` and the command: the present, no memory;
(ii)  the history of ``z*`` on the memory's grid, and the command;
(iii) the history of the full observation: everything the robot knows.

and two targets that a sense of phase would predict:

1. where the physical state goes next: ``z*_{t+k} - z*_t``;
2. how many control steps until each foot next lands (contact threshold 5 %
   of body weight), capped at 2 ``T``, where a landing follows in the episode.

Rule, fixed in the plan before running: the restriction is accepted if (ii)
scores at least (iii) - 0.05 on both targets with the MLP probe. Otherwise the
missing quantity has to be named and added to ``z*`` -- not let in through a
free channel.

Results go to ``transfer/results/phase_probe.json``.

Usage::

    python transfer/scripts/probe_phase.py
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

from humanoid_transfer.common.gait import touchdowns  # noqa: E402
from humanoid_transfer.common.similarity import TIME, UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    contact_state,
)
from humanoid_transfer.transfer.latent.memory import TimeGrid  # noqa: E402
from humanoid_transfer.transfer.latent.probes import linear_probe, mlp_probe  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402

GRID_SAMPLES = 26
"""About 2 ``T`` on the G1's control step: one stride of the teacher and some."""

HORIZON = 8
"""``k``: about 0.6 ``T`` ahead on the G1."""

TOLERANCE = 0.05


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def steps_to_landing(landing: np.ndarray, done: np.ndarray) -> np.ndarray:
    """``(T, N, 2)`` control steps until the foot next lands in the same episode; inf if never."""
    result = np.full(landing.shape, np.inf)
    following = np.full(landing.shape[1:], np.inf)
    for t in range(landing.shape[0] - 1, -1, -1):
        following[done[t]] = np.inf  # landings after t belong to the next episode
        result[t] = following - t
        following = np.where(landing[t], t, following)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "g1/rollouts/teacher_dart.npz")
    parser.add_argument("--val-envs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--results", type=Path, default=REPO_ROOT / "transfer/results/phase_probe.json"
    )
    args = parser.parse_args()

    archive = np.load(args.data)
    units = UnitSystem(float(archive["leg_length"]), float(archive["total_mass"]))
    dt_star = float(units.to_dimensionless(float(archive["control_dt"]), TIME))
    grid = TimeGrid(step=dt_star, samples=GRID_SAMPLES)
    assert np.allclose(grid.interpolation(dt_star), np.eye(GRID_SAMPLES))

    # Time to the next landing, laid out like the stream: column by column.
    done = archive["done"]
    steps, envs = done.shape
    contact = contact_state(archive["skill_state"], threshold=DEFAULT_CONTACT_THRESHOLD)
    landing = steps_to_landing(touchdowns(contact, done), done)
    cap = round(2.0 / dt_star)
    landing_flat = torch.as_tensor(np.transpose(landing, (1, 0, 2)).reshape(-1, 2))

    streams = load_streams([args.data]).to(args.device)
    samples = streams.samples(future=HORIZON)
    held_out = streams.env[samples] >= envs - args.val_envs
    train, test = samples[~held_out], samples[held_out]

    def present(idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([streams.skill[idx], streams.command[idx]], dim=-1)

    def physical_history(idx: torch.Tensor) -> torch.Tensor:
        window = streams.skill[streams.windows(idx, GRID_SAMPLES)].flatten(1)
        return torch.cat([window, streams.command[idx]], dim=-1)

    def observation_history(idx: torch.Tensor) -> torch.Tensor:
        return streams.obs[streams.windows(idx, GRID_SAMPLES)].flatten(1)

    inputs = {
        "present": present,
        "physical_history": physical_history,
        "observation_history": observation_history,
    }

    future = streams.skill[streams.futures(samples, HORIZON)[:, -1]]
    drift = torch.zeros_like(streams.skill)
    drift[samples] = future - streams.skill[samples]

    to_land = landing_flat.to(args.device).float()
    lands = torch.isfinite(to_land).all(dim=1)
    to_land = to_land.clamp(max=cap).nan_to_num(posinf=float(cap))
    landing_train = train[lands[train]]
    landing_test = test[lands[test]]

    targets = {
        "future_skill_state": (drift, train, test),
        "steps_to_next_landing": (to_land, landing_train, landing_test),
    }

    report: dict = {
        "data": _display(args.data),
        "teacher_sha256": str(archive["teacher_sha256"]),
        "grid": {"step_T": dt_star, "samples": GRID_SAMPLES, "span_T": grid.span},
        "horizon_steps": HORIZON,
        "landing_cap_steps": cap,
        "contact_threshold": DEFAULT_CONTACT_THRESHOLD,
        "validation_envs": args.val_envs,
        "rule": f"physical_history >= observation_history - {TOLERANCE} (MLP) on both targets",
        "targets": {},
    }
    for target_name, (values, fit, score) in targets.items():
        rows: dict = {"train_steps": len(fit), "test_steps": len(score)}
        for input_name, features in inputs.items():
            mlp = mlp_probe(features, values, fit, score, epochs=args.epochs, seed=args.seed)
            linear = linear_probe(features, values, fit, score)
            rows[input_name] = {
                "mlp_r2": float(np.nanmedian(mlp)),
                "linear_r2": float(np.nanmedian(linear)),
                "mlp_r2_per_output": mlp.tolist(),
            }
            print(
                f"{target_name:24s} {input_name:20s} "
                f"mlp {rows[input_name]['mlp_r2']:.3f}  linear {rows[input_name]['linear_r2']:.3f}"
            )
        physical = rows["physical_history"]["mlp_r2"]
        rows["memory_gain"] = physical - rows["present"]["mlp_r2"]
        rows["gap_to_observation_history"] = rows["observation_history"]["mlp_r2"] - physical
        rows["passes"] = physical >= rows["observation_history"]["mlp_r2"] - TOLERANCE
        report["targets"][target_name] = rows

    report["passes"] = all(row["passes"] for row in report["targets"].values())
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"gate {'PASSES' if report['passes'] else 'FAILS'}; written {_display(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
