"""What a body at another control rate is asked for: the generator chained as the task runs it.

Under the autonomous reference (the one every tracker trains in and L4 deploys with) a
plan is drawn from the history of the plans already committed -- each due state
appended with the encoder's noise -- read onto the generator's grid, and its samples
read back at the body's own instants (``ReferenceCommand._update_command``). The body
enters only at the start, and at a resync. So the sequence of states a body is asked
for can be computed without the body, at any control rate, under either resampling
space (:mod:`~humanoid_transfer.transfer.latent.resampling`).

The start states and commands are read from an L4 capture of the body, so the chain
starts where the deployed one did. The capture's own recorded plans (``contract_due``)
check the model: in the space the capture ran in, the chain's asked gait must match
them.

    python transfer/scripts/chain_at_rate.py \
        logs/contract/flat_dagger2_zs_nao_r1_3498_sym_v3/generator.pt \
        --capture nao/rollouts/zs_r1_3498_sym_v3_flat_dagger2_auto_kin_forward_encoder.npz \
        --output transfer/results/chain_at_rate_nao_flat_forward.json
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

from humanoid_transfer.common.contact import capture_fidelity, contact_fidelity  # noqa: E402
from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.memory import TimeGrid  # noqa: E402
from humanoid_transfer.transfer.latent.resampling import (  # noqa: E402
    RESAMPLE_SPACES,
    plan_interpolation,
    resample_rows,
)


def chain_at_rate(
    generator,
    start: torch.Tensor,
    command: torch.Tensor,
    sigma: torch.Tensor,
    dt_star: float,
    steps: int,
    space: str,
    seed: int = 0,
) -> torch.Tensor:
    """``(steps, N, 27)`` normalised due states, as ``ReferenceCommand`` (autonomous) asks.

    ``start`` ``(N, 27)`` fills the history, as a reset's refill does; ``command`` is
    dimensionless ``(N, 3)``; ``sigma`` the noise appended with each due state.
    """
    generator_step = float(generator.grid_step)
    grid = TimeGrid(step=generator_step, samples=generator.history)
    interpolation = torch.tensor(grid.interpolation(dt_star), dtype=torch.float32)
    plan_matrix = plan_interpolation(generator_step, generator.horizon, dt_star)
    commit = max(1, round(generator.commit * generator_step / dt_star))
    history = start[:, None].expand(-1, interpolation.shape[1], -1).clone()
    generate = torch.Generator().manual_seed(seed)

    def draw(history: torch.Tensor) -> torch.Tensor:
        window = resample_rows(generator, interpolation, history, space)
        noise = torch.randn(len(window), generator.noise_size, generator=generate)
        with torch.no_grad():
            plan = generator.sample(window, command, noise).float()
        plan = torch.nan_to_num(plan, nan=0.0).clamp(-50.0, 50.0)
        return resample_rows(generator, plan_matrix, torch.cat([window[:, -1:], plan], 1), space)

    plan, age, dues = draw(history), 0, []
    for _ in range(steps):
        due = plan[:, age]
        dues.append(due)
        noisy = due + sigma * torch.randn(due.shape, generator=generate)
        history = torch.cat([history[:, 1:], noisy[:, None]], dim=1)
        age += 1
        if age >= commit:
            plan, age = draw(history), 0
    return torch.stack(dues)


def summary(z: np.ndarray, dt_star: float) -> dict:
    done = np.zeros(z.shape[:2], dtype=bool)
    signature = gait_signature(z, done, dt_star).as_dict()
    fidelity = contact_fidelity(z, done, dt_star).as_dict()
    return {
        "duty_factor": signature["duty_factor"],
        "stride_period": signature["stride_period"],
        "stride_length": signature["stride_length"],
        "speed": signature["speed"],
        "light_share": fidelity["light_share_body"],
        "clearance": fidelity["clearance_body"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("contract", type=Path)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = load_contract(args.contract, "cpu")
    generator = contract.generator
    generator.flow_solver, generator.flow_steps = "midpoint", 8  # as the tracking tasks run it
    data = np.load(args.capture)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    dt_star = float(data["control_dt"]) / units.time
    steps = data["skill_state"].shape[0]
    # The refill after the first reset: the second recorded state of each environment.
    start = generator.skill_norm(torch.as_tensor(data["skill_state"][1], dtype=torch.float32))
    velocity = torch.as_tensor(data["command"][1], dtype=torch.float32)
    scale = torch.tensor([1.0 / units.velocity, 1.0 / units.velocity, units.time])
    command = velocity * scale

    recorded = capture_fidelity(data, dt_star).as_dict()
    due = data["contract_due"][:-1].astype(float)
    report = {
        "contract": args.contract.as_posix(),
        "capture": args.capture.as_posix(),
        "dt_star": dt_star,
        "grid_step": float(generator.grid_step),
        "recorded_asked": {
            **summary(due, dt_star),
            "duty_factor": recorded["duty_asked"],
        },
        "chains": {},
    }
    for space in RESAMPLE_SPACES:
        runs = []
        for seed in range(args.seeds):
            chain = chain_at_rate(
                generator, start, command, contract.sigma, dt_star, steps, space, seed
            )
            z = generator.skill_norm.inverse(chain).numpy().astype(float)
            runs.append(summary(z, dt_star))
        report["chains"][space] = {
            key: np.mean([r[key] for r in runs], axis=0).tolist() for key in runs[0]
        } | {"seeds": args.seeds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def line(name: str, s: dict) -> str:
        duty = "/".join(f"{v:.3f}" for v in s["duty_factor"])
        light = "/".join(f"{v:.3f}" for v in s["light_share"])
        return (
            f"  {name:<16} duty {duty}  period {s['stride_period']:.3f}  "
            f"stride {s['stride_length']:.3f}  speed {s['speed']:.4f}  light {light}"
        )

    print(f"{args.capture.name}  dt* {dt_star:.4f} (grid {float(generator.grid_step):.4f})")
    print(line("recorded asked", report["recorded_asked"]))
    for space, s in report["chains"].items():
        print(line(f"chain {space}", s))
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
