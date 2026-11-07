"""Offline proxy for a held-out task: each generator chained on its own plans, a perfect tracker.

For each generator and each of rough's L4 commands, 32 chains start from the rough teacher's own
play histories (the same starts ``train_contract.py``'s chain gait uses: steps 100 to 100 +
history), at that command, and run 900 steps on their own plans (``chained_gait``). Their gait
is compared with the teacher's, as gate C compares a body's: the relative deviation of stride
length, stride period and each foot's duty factor, and the largest of them; and the speed ratio,
chain over teacher. The rough captures are read here only, never trained on.

What it separates: a deviation the chain already shows is the generator's, before any body; one
that only the simulator shows is the loop through a body.

    python transfer/scripts/held_out_chains.py --output transfer/results/held_out_chains_w16.json
    python transfer/scripts/held_out_chains.py --generators mt17_c mt17_m --output ...
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

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    chained_gait,
    gait_reference,
)

DEFAULT = ("mt16_c", "mt16_m", "mt16_d", "rough_b4", "flat_dagger2")
COMMANDS = ("forward", "fast", "turning")
STEPS = 900
STARTS = 32


def signature(sig: dict) -> dict:
    return {
        "stride_length": sig["stride_length"],
        "stride_period": sig["stride_period"],
        "duty_factor_0": sig["duty_factor"][0],
        "duty_factor_1": sig["duty_factor"][1],
        "speed": sig["speed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generators", nargs="*", default=list(DEFAULT))
    parser.add_argument("--task", default="rough", help="the held-out task's play captures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {"task": args.task, "steps": STEPS, "starts": STARTS, "generators": {}}
    for name in args.generators:
        path = f"logs/contract/{name}/generator.pt"
        contract = load_contract(REPO_ROOT / path, device)
        generator = contract.generator
        dt_star = float(contract.meta["dt_star"])
        sigma = contract.sigma.to(device)
        entry: dict = {"path": path}
        for command_name in COMMANDS:
            data = np.load(REPO_ROOT / f"g1/rollouts/{args.task}_play_{command_name}.npz")
            units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
            scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
            z = data["skill_state"].astype(np.float32)
            history = generator.history
            start = torch.as_tensor(z[100 : 100 + history, :STARTS], device=device)
            start = generator.skill_norm(start.transpose(0, 1))
            command = torch.as_tensor(
                data["command"][100, :STARTS] * scale, dtype=torch.float32, device=device
            )
            chained = chained_gait(
                generator, start, command, sigma, STEPS, generator.commit, dt_star, seed=args.seed
            )
            reference = gait_reference(z[100 + history :, :STARTS].astype(float), dt_star)
            chain, teacher = signature(chained["signature"]), signature(reference["signature"])
            deviation = {k: abs(chain[k] - teacher[k]) / teacher[k] for k in chain if k != "speed"}
            entry[command_name] = {
                "chain": chain,
                "teacher": teacher,
                "deviation": deviation,
                "largest": max(deviation.values()),
                "speed_ratio": chain["speed"] / teacher["speed"],
                "stable": chained["environments_stable"],
            }
        entry["largest_mean"] = float(np.mean([entry[c]["largest"] for c in COMMANDS]))
        report["generators"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for name, entry in report["generators"].items():
        for c in COMMANDS:
            e = entry[c]
            duty = (e["chain"]["duty_factor_0"] + e["chain"]["duty_factor_1"]) / 2
            t_duty = (e["teacher"]["duty_factor_0"] + e["teacher"]["duty_factor_1"]) / 2
            print(
                f"{name:13s} {c:8s} largest {e['largest']:.3f}  duty {duty:.3f} (teacher "
                f"{t_duty:.3f})  period {e['chain']['stride_period']:.2f} (teacher "
                f"{e['teacher']['stride_period']:.2f})  speed {e['speed_ratio']:.3f}  "
                f"stable {e['stable']}"
            )
        print(f"{name:13s} mean largest {entry['largest_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
