"""Replay files for co-executions: each base task's generator chained on its own plans.

For each base task -- the walk contract, and flat and jog from ``base_tasks_xf.json`` (their
cross-fitted twins are the same tasks) -- ``--per-task`` chains start from clean windows of the
task teacher's capture, at that window's command, and run ``--steps`` grid samples on their own
plans (:func:`~humanoid_transfer.transfer.latent.train_contract.chain_states`: the generator
reads its own plans with its training noise). Unstable chains are dropped. The held-out tasks
(sidestep, rough) have no generator here: replays and the co-executions made from them never
contain their plans.

The file (``latent/replay.py``) holds dimensionless ``z*``, so any body's contract tracker
reads it in its own units (``capture_contract.py --replay-plans``).

Usage::

    python transfer/scripts/pact_make_replays.py --output logs/pact/replays/base_tasks.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

import torch  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402

TASKS = {
    "walk": ("logs/contract/walk_dagger3e/generator.pt", "g1/rollouts/teacher.npz"),
}
HELD_OUT = ("sidestep", "rough")


def chained(generator, start, command, sigma, steps: int, commit: int, seed: int) -> np.ndarray:
    """``(steps, N, skill)``: ``train_contract.chain_states``'s
    loop, every chain kept (the caller drops the unstable ones with their commands)."""
    torch.manual_seed(seed)
    clean, out = start, []
    while len(out) * commit < steps:
        window = clean + sigma * torch.randn_like(clean)
        noise = torch.randn(len(window), generator.noise_size, device=window.device)
        chunk = generator.sample(window, command, noise)[:, :commit]
        out.append(chunk)
        clean = torch.cat([clean[:, commit:], chunk], dim=1)
    z = generator.skill_norm.inverse(torch.cat(out, dim=1)[:, :steps]).transpose(0, 1)
    return z.cpu().numpy().astype(float)


def base_tasks(path: Path) -> dict[str, tuple[str, str]]:
    tasks = dict(TASKS)
    for spec in json.loads(path.read_text("utf-8"))["tasks"]:
        if spec["name"].endswith("_xf"):
            continue
        tasks[spec["name"]] = (spec["contract"], spec["dart"])
    for name in tasks:
        if any(h in name for h in HELD_OUT):
            raise SystemExit(f"{name}: a held-out task cannot be replayed")
    return tasks


def windows(capture: dict, history: int, count: int, rng: np.random.Generator):
    """``count`` clean windows (no episode end inside) of ``history`` rows and their commands,
    dimensionless."""
    z = capture["skill_state"]
    done = capture["done"] if "done" in capture.files else np.zeros(z.shape[:2], dtype=bool)
    clean = ~np.lib.stride_tricks.sliding_window_view(done, history + 1, axis=0).any(-1)
    t, n = np.nonzero(clean)
    pick = rng.choice(len(t), size=count, replace=False)
    t, n = t[pick] + history, n[pick]
    rows = np.stack([z[t - history + 1 + k, n] for k in range(history)], axis=1)
    units = UnitSystem(float(capture["leg_length"]), float(capture["total_mass"]))
    scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
    return rows, capture["command"][t, n] * scale, np.stack([t, n], -1), units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tasks-file", type=Path, default=REPO / "transfer/results/base_tasks_xf.json"
    )
    parser.add_argument("--tasks", default="walk,flat,jog")
    parser.add_argument("--per-task", type=int, default=48)
    parser.add_argument("--steps", type=int, default=2000, help="Grid samples per chain.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    available = base_tasks(args.tasks_file)
    rng = np.random.default_rng(args.seed)
    states, commands, names, starts, report = [], [], [], [], {}
    grid = None
    for name in [t for t in args.tasks.split(",") if t]:
        contract_path, capture_path = available[name]
        contract = load_contract(REPO / contract_path, args.device)
        generator = contract.generator
        step = float(generator.grid_step)
        if grid is not None and abs(step - grid) > 1e-6:
            raise SystemExit(f"{name}: grid {step} differs from {grid}")
        grid = step
        capture = np.load(REPO / capture_path)
        rows, command, where, units = windows(capture, generator.history, args.per_task, rng)
        dt = float(capture["control_dt"]) / units.time if "control_dt" in capture.files else step
        if abs(dt - step) > 1e-4:
            raise SystemExit(f"{capture_path}: its step {dt} is not the generator's grid {step}")
        start = generator.skill_norm(torch.as_tensor(rows, dtype=torch.float32, device=args.device))
        command_t = torch.as_tensor(command, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            z = chained(
                generator,
                start,
                command_t,
                contract.sigma.to(args.device),
                args.steps,
                generator.commit,
                args.seed,
            )
        # A chain that leaves the generator's range (as chain_states judges it) is dropped,
        # with its command and start.
        stable = np.isfinite(z).all(axis=(0, 2)) & (np.abs(z).max(axis=(0, 2)) < 10.0)
        z, command, where = z[:, stable], command[stable], where[stable]
        fields = SKILL_LAYOUT.unpack(z.reshape(-1, SKILL_LAYOUT.size))
        speed = fields["com_velocity"].reshape(z.shape[0], z.shape[1], 3)[..., :2]
        height = fields["height"].reshape(z.shape[0], z.shape[1])
        report[name] = {
            "contract": contract_path,
            "capture": capture_path,
            "chains": int(z.shape[1]),
            "unstable_dropped": int((~stable).sum()),
            "standing_commands": int((np.linalg.norm(command, axis=-1) == 0).sum()),
            "command_speed_mean": float(np.linalg.norm(command[:, :2], axis=-1).mean()),
            "chain_speed_mean": float(np.linalg.norm(speed, axis=-1).mean()),
            "height_min": float(height.min()),
            "height_mean": float(height.mean()),
        }
        print(name, report[name])
        states.append(z.astype(np.float32))
        commands.append(command.astype(np.float32))
        names += [name] * z.shape[1]
        starts.append(where)
    chains = np.concatenate(states, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        chain_states=chains,
        grid_step=np.float64(grid),
        command=np.concatenate(commands),
        task=np.array(names),
        start=np.concatenate(starts),
        seed=np.int64(args.seed),
        report=np.array(json.dumps(report)),
    )
    print(f"written {args.output}: {chains.shape[1]} chains x {chains.shape[0]} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
