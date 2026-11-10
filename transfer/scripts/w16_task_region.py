"""W16 follow-up, offline: what a generator asks for from a teacher's own history.

W16's rough L4 found the multi-task generator asking for a long-stance, slow gait at rough's
forward command (duty 0.82 on the G1, 0.92 on the NAO; the rough teacher's 0.57). Three
sources can produce that, and this script separates them without the simulator:

* the command region -- at 0.25 m/s the base tasks hold the walk expert's gait, not rough's;
* the history -- a rough-terrain history is outside what the base tasks contain;
* the closed loop -- the body's slower execution feeds back into the next plan's history.

For each generator, plans are drawn (one per window, the deployed integrator, the contract's
history noise) from the teachers' own play histories, so the loop is open:

* ``rough_<command>`` and ``flat_<command>`` -- rough's and flat's play captures at L4's three
  commands, the same commands on two terrains' teachers;
* ``walk_0.25`` -- the walk expert's DART capture at forward commands of 0.22-0.25 m/s
  (lateral and yaw below 0.05).

Per source it reports the plans' duty factor and forward speed ratio (the plan's CoM velocity
over the command), beside the same figures for the future the teacher walked.

    python transfer/scripts/w16_task_region.py --output transfer/results/w16_task_region.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from sampler_sharpness import load_stats  # noqa: E402

GENERATORS = {
    "mt16_c": "logs/contract/mt16_c/generator.pt",
    "mt16_m": "logs/contract/mt16_m/generator.pt",
    "mt16_d": "logs/contract/mt16_d/generator.pt",
    "rough_b4": "logs/contract/rough_b4/generator.pt",
    "flat_dagger2": "logs/contract/flat_dagger2/generator.pt",
}
COMMANDS = ("forward", "fast", "turning")
LOADS = SKILL_LAYOUT.slice_of("foot_load")
COM_VELOCITY = SKILL_LAYOUT.slice_of("com_velocity")
WALK = "g1/rollouts/teacher_dart.npz"
WALK_RANGE = (0.22, 0.25)  # m/s, forward
WALK_OTHER = 0.05  # m/s lateral and rad/s yaw, at most


def windows(path: Path, history: int, horizon: int, stride: int, keep=None) -> dict:
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
    z = data["skill_state"].astype(np.float32)
    done = data["done"].astype(bool)
    command = data["command"]
    out = {"window": [], "future": [], "command": []}
    for env in range(z.shape[1]):
        for t in range(100 + history, z.shape[0] - horizon, stride):
            if done[t - history : t + horizon, env].any():
                continue
            if keep is not None and not keep(command[t, env]):
                continue
            out["window"].append(z[t - history + 1 : t + 1, env])
            out["future"].append(z[t + 1 : t + 1 + horizon, env])
            out["command"].append(command[t, env] * scale)
    return {k: np.stack(v) for k, v in out.items()}


def gait(z: np.ndarray, command: np.ndarray) -> dict:
    """Duty factor (both feet) and forward speed ratio of plans or futures ``(B, K, 27)``."""
    speed = z[..., COM_VELOCITY][..., 0].mean(axis=-1)
    ratio = float(np.mean(speed) / np.mean(command[:, 0]))
    return {"duty": float(np.mean(load_stats(z[..., LOADS])["duty"])), "speed_ratio": ratio}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stride", type=int, default=14, help="steps between windows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = {
        f"{t}_{c}": f"g1/rollouts/{t}_play_{c}.npz" for t in ("rough", "flat") for c in COMMANDS
    }

    def walking(c: np.ndarray) -> bool:
        lo, hi = WALK_RANGE
        return lo <= c[0] <= hi and abs(c[1]) <= WALK_OTHER and abs(c[2]) <= WALK_OTHER

    report: dict = {"stride": args.stride, "generators": GENERATORS, "sources": {}}
    for name, path in GENERATORS.items():
        torch.manual_seed(args.seed)
        contract = load_contract(REPO_ROOT / path, device)
        generator = contract.generator
        norm = generator.skill_norm
        entry = {}
        cases = {**{k: (REPO_ROOT / v, None) for k, v in sources.items()}}
        cases["walk_0.25"] = (REPO_ROOT / WALK, walking)
        for source, (file, keep) in cases.items():
            w = windows(file, generator.history, generator.horizon, args.stride, keep)
            window = norm(torch.as_tensor(w["window"], device=device))
            window = window + contract.sigma * torch.randn_like(window)
            command = torch.as_tensor(w["command"], dtype=torch.float32, device=device)
            noise = torch.randn(
                len(window), generator.noise_size, device=device,
                generator=torch.Generator(device).manual_seed(args.seed + 1),
            )  # fmt: skip
            with torch.no_grad():
                plan = norm.inverse(generator.sample(window, command, noise).float())
            plan = plan.double().cpu().numpy()
            entry[source] = {
                "windows": len(plan),
                "plan": gait(plan, w["command"]),
                "teacher": gait(w["future"].astype(float), w["command"]),
            }
        report["sources"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    names = list(GENERATORS)
    print(f"{'source':16s} {'teacher':>14s} " + " ".join(f"{n:>14s}" for n in names))
    for source in report["sources"][names[0]]:
        t = report["sources"][names[0]][source]["teacher"]
        row = f"{source:16s} {t['duty']:.3f} / {t['speed_ratio']:.2f} "
        for n in names:
            p = report["sources"][n][source]["plan"]
            row += f"  {p['duty']:.3f} / {p['speed_ratio']:.2f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
