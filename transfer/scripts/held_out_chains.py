"""Offline proxy for a held-out task: each generator chained on its own plans, a perfect tracker.

For each generator and each of the task's L4 commands, 32 chains start from the task teacher's
own captures (by default the same starts ``train_contract.py``'s chain gait uses: steps 100 to
100 + history), at that command, and run 900 steps on their own plans (``chain_states``). Their
gait is compared with the teacher's, as gate C compares a body's: the relative deviation of
stride length, stride period and each foot's duty factor, and the largest of them; and the
speed ratio, chain over teacher. The held-out captures are read here only, never trained on.

What it separates: a deviation the chain already shows is the generator's, before any body; one
that only the simulator shows is the loop through a body.

Where the chains start (``--start``):
- ``teacher``: the teacher's own history. It already shows the teacher's gait, so the generator
  is asked to continue a gait, not to choose one (W17, ``skill_space.md`` §10.18: this start
  passed a generator that the deployed path then failed).
- ``static``: the teacher's first captured state, repeated over the window.
- ``body``: the first estimate of a deployed L4 capture (``--start-capture``, a path with
  ``{command}`` and optionally ``{arm}``), repeated over the window; ``--start-step`` picks
  another recorded step. A capture recorded with no settle steps holds the body's state after a
  reset. The deployed autonomous path fills its history with the estimate of the second update
  after a reset (``command.py``: ``just_reset``, then ``refill``), recorded at step 2, before
  chaining on its own plans. The ladder's captures start after 100 settle steps under their
  own generator.

    python transfer/scripts/held_out_chains.py --output transfer/results/held_out_chains_w16.json
    python transfer/scripts/held_out_chains.py --generators mt17_c mt17_m --output ...
    python transfer/scripts/held_out_chains.py --start body \\
        --start-capture "g1/rollouts/zs_mt_g1_s41_sym_{arm}_own_auto_kin_{command}_encoder.npz" \\
        --generators mt16_c mt17_c mt17_m rough_b4 \\
        --output transfer/results/held_out_chains_w17_own_start.json
    python transfer/scripts/held_out_chains.py --task base_sidestep \\
        --commands left left_fast right --generators mt17_c --output ...
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

from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import DEFAULT_CONTACT_THRESHOLD  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    chain_states,
    gait_reference,
)

DEFAULT = ("mt16_c", "mt16_m", "mt16_d", "rough_b4", "flat_dagger2")
COMMANDS = ("forward", "fast", "turning")
STEPS = 900
STARTS = 32
HEIGHT = 0
LOAD = slice(22, 24)


def signature(sig: dict) -> dict:
    return {
        "stride_length": sig["stride_length"],
        "stride_period": sig["stride_period"],
        "duty_factor_0": sig["duty_factor"][0],
        "duty_factor_1": sig["duty_factor"][1],
        "speed": sig["speed"],
    }


def posture(z: np.ndarray) -> dict:
    """``z`` ``(T, N, skill)``: the mean height and the share of loaded foot-steps."""
    return {
        "height": float(z[..., HEIGHT].mean()),
        "loaded": float((z[..., LOAD] > DEFAULT_CONTACT_THRESHOLD).mean()),
    }


def start_window(args, name: str, command_name: str, z: np.ndarray, history: int) -> np.ndarray:
    if args.start == "body":
        path = args.start_capture.format(command=command_name, arm=name)
        estimate = np.load(REPO_ROOT / path)["estimate"]
        first = estimate[args.start_step : args.start_step + 1, :STARTS].astype(np.float32)
        return np.repeat(first, history, axis=0)
    if args.start == "static":
        return np.repeat(z[:1, :STARTS], history, axis=0)
    return z[100 : 100 + history, :STARTS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generators", nargs="*", default=list(DEFAULT))
    parser.add_argument("--task", default="rough", help="the held-out task's teacher captures")
    parser.add_argument("--commands", nargs="*", default=list(COMMANDS))
    parser.add_argument("--rollout", default="play", help="g1/rollouts/<task>_<rollout>_<command>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", choices=("teacher", "static", "body"), default="teacher")
    parser.add_argument(
        "--start-capture",
        default="g1/rollouts/zs_mt_g1_s41_sym_rough_b4_own_auto_kin_{command}_encoder.npz",
        help="with --start body: the L4 capture whose estimate starts the chains",
    )
    parser.add_argument("--start-step", type=int, default=0, help="with --start body")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {
        "task": args.task,
        "commands": args.commands,
        "rollout": args.rollout,
        "start": args.start,
        "start_capture": args.start_capture if args.start == "body" else None,
        "start_step": args.start_step if args.start == "body" else None,
        "steps": STEPS,
        "starts": STARTS,
        "generators": {},
    }
    for name in args.generators:
        path = f"logs/contract/{name}/generator.pt"
        contract = load_contract(REPO_ROOT / path, device)
        generator = contract.generator
        dt_star = float(contract.meta["dt_star"])
        sigma = contract.sigma.to(device)
        entry: dict = {"path": path}
        for command_name in args.commands:
            data = np.load(REPO_ROOT / f"g1/rollouts/{args.task}_{args.rollout}_{command_name}.npz")
            units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
            scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
            z = data["skill_state"].astype(np.float32)
            history = generator.history
            window = start_window(args, name, command_name, z, history)
            start = generator.skill_norm(torch.as_tensor(window, device=device).transpose(0, 1))
            command = torch.as_tensor(
                data["command"][100, :STARTS] * scale, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                states = chain_states(
                    generator, start, command, sigma, STEPS, generator.commit, args.seed
                )
            done = np.zeros(states.shape[:2], dtype=bool)
            chained = gait_signature(states, done, dt_star).as_dict()
            reference = gait_reference(z[100 + history :, :STARTS].astype(float), dt_star)
            chain, teacher = signature(chained), signature(reference["signature"])
            deviation = {k: abs(chain[k] - teacher[k]) / teacher[k] for k in chain if k != "speed"}
            entry[command_name] = {
                "chain": chain,
                "teacher": teacher,
                "deviation": deviation,
                "largest": max(deviation.values()),
                "speed_ratio": chain["speed"] / teacher["speed"],
                "stable": int(states.shape[1]),
                "posture": posture(states),
                "teacher_posture": posture(z[100 + history :, :STARTS]),
                "start_height": float(window[..., HEIGHT].mean()),
            }
        entry["largest_mean"] = float(np.mean([entry[c]["largest"] for c in args.commands]))
        report["generators"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for name, entry in report["generators"].items():
        for c in args.commands:
            e = entry[c]
            duty = (e["chain"]["duty_factor_0"] + e["chain"]["duty_factor_1"]) / 2
            t_duty = (e["teacher"]["duty_factor_0"] + e["teacher"]["duty_factor_1"]) / 2
            print(
                f"{name:13s} {c:9s} largest {e['largest']:.3f}  duty {duty:.3f} (teacher "
                f"{t_duty:.3f})  period {e['chain']['stride_period']:.2f} (teacher "
                f"{e['teacher']['stride_period']:.2f})  height {e['posture']['height']:.3f} "
                f"(start {e['start_height']:.3f}, teacher {e['teacher_posture']['height']:.3f})  "
                f"speed {e['speed_ratio']:.3f}  stable {e['stable']}"
            )
        print(f"{name:13s} mean largest {entry['largest_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
