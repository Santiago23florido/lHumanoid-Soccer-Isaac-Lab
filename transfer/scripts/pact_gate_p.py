"""G-P: do a generator's own plans keep a loaded foot still, and keep the teacher's gait?

For each generator and each task command, 32 chains start from the task teacher's captures
(steps 100 to 100 + history, as ``held_out_chains.py`` starts them) and run ``--steps`` steps on
their own plans (``chain_states``: every plan drawn from the history the chain wrote). Tasks:
seen (flat forward/fast/turning, jog slow/mid/fast) and held out (sidestep left/left_fast/right,
which no generator compared here was trained on). Per command:

* **slip**: the median horizontal speed, in ``V``, of firmly loaded feet (load above
  ``LIGHT_LOAD`` body weights), pooled over the chains. The teachers' captures read 0.002 V.
* **calibration**: the largest relative deviation of the chains' gait signature from the
  teacher's (stride length, stride period, each foot's duty factor), as ``held_out_chains.py``.
* **Newton** (reported, not gated): the median of ``|a_z - (sum of loads - 1)|`` in ``g``, from
  the plan's own vertical CoM velocity and loads, and the mean total load in body weights.

Each group (seen, held out) is the mean over its commands. Standard errors are bootstrap over
the chains (``--bootstrap`` draws, the chains of each command resampled); an arm's difference
from the control has the SE ``sqrt(se_arm^2 + se_control^2)``.

Gate (registered 2026-11-22, before any data from the stick arm): the arm passes if, in both
groups, its slip is at most 0.02 V and its calibration is no worse than the control's by more
than 2 SE. The control is the same recipe retrained without the loss, on the same device and
seed.

    python transfer/scripts/pact_gate_p.py --control mt18_no_sidestep_s0 \\
        --arms mt18_no_sidestep_s1 --reference mt18_no_sidestep \\
        --output transfer/results/icml_gate_p.json

An arm may also be ``name@plan_contact``: the same generator with each plan projected onto
the contact-consistent plans before it is committed (``stick``, as deployment projects a plan)
or drawn with that projection guiding the sampler (``guided``). Their slip is zero by
construction for every foot loaded above the contact threshold, so for them the gate reads
calibration; ``--contrast A B`` reports A's calibration minus B's.

    python transfer/scripts/pact_gate_p.py --control mt18_no_sidestep_s0 \\
        --arms mt18_no_sidestep_s0@stick mt18_no_sidestep_s0@guided \\
        --contrast mt18_no_sidestep_s0@guided mt18_no_sidestep_s0@stick \\
        --output transfer/results/icml_gate_p_guided.json
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

from humanoid_transfer.common.contact import LIGHT_LOAD  # noqa: E402
from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    chain_states,
    gait_reference,
)

TASKS = {
    "seen": (
        ("flat", "play", ("forward", "fast", "turning")),
        ("base_jog", "play", ("slow", "mid", "fast")),
    ),
    "held_out": (("base_sidestep", "play", ("left", "left_fast", "right")),),
}
STARTS = 32
SLIP_BOUND = 0.02
SIGNATURE = ("stride_length", "stride_period", "duty_factor_0", "duty_factor_1")


def firm_speeds(z: np.ndarray) -> list[np.ndarray]:
    """Per chain, the horizontal speeds of its firmly loaded feet; ``z`` ``(T, N, skill)``."""
    fields = SKILL_LAYOUT.unpack(z)
    speed = np.linalg.norm(fields["foot_velocity"][..., :2], axis=-1)
    firm = fields["foot_load"] > LIGHT_LOAD
    return [speed[:, n][firm[:, n]] for n in range(z.shape[1])]


def newton(z: np.ndarray, dt_star: float) -> tuple[float, float]:
    """``(median |a_z - (sum load - 1)|, mean total load)`` of ``z`` ``(T, N, skill)``."""
    fields = SKILL_LAYOUT.unpack(z)
    load = fields["foot_load"].sum(-1)
    vertical = fields["com_velocity"][..., 2]
    acceleration = (vertical[1:] - vertical[:-1]) / dt_star
    residual = acceleration - 0.5 * (load[1:] + load[:-1]) + 1.0
    return float(np.median(np.abs(residual))), float(load.mean())


def signature(z: np.ndarray, dt_star: float) -> dict:
    sig = gait_signature(z, np.zeros(z.shape[:2], dtype=bool), dt_star).as_dict()
    return {
        "stride_length": sig["stride_length"],
        "stride_period": sig["stride_period"],
        "duty_factor_0": sig["duty_factor"][0],
        "duty_factor_1": sig["duty_factor"][1],
    }


def largest(chain: dict, teacher: dict) -> float:
    return max(abs(chain[k] - teacher[k]) / teacher[k] for k in SIGNATURE)


def command_read(z: np.ndarray, teacher: dict, dt_star: float, draws: int, rng) -> dict:
    """Slip and calibration of one command's chains, with their bootstrap draws."""
    speeds = firm_speeds(z)
    pooled = np.concatenate(speeds)
    chains = z.shape[1]
    slip_draws, calibration_draws = np.empty(draws), np.empty(draws)
    for d in range(draws):
        pick = rng.integers(0, chains, chains)
        slip_draws[d] = np.median(np.concatenate([speeds[i] for i in pick]))
        calibration_draws[d] = largest(signature(z[:, pick], dt_star), teacher)
    residual, load = newton(z, dt_star)
    chain_signature = signature(z, dt_star)
    return {
        "chains": int(chains),
        "slip": float(np.median(pooled)),
        "slip_p90": float(np.quantile(pooled, 0.9)),
        "calibration": largest(chain_signature, teacher),
        "signature": chain_signature,
        "newton_residual": residual,
        "mean_load": load,
        "_slip_draws": slip_draws,
        "_calibration_draws": calibration_draws,
    }


def read_generator(arm: str, steps: int, draws: int, seed: int, device: str) -> dict:
    """``arm`` is a generator's name, or ``name@plan_contact`` (``chain_states``'s modes)."""
    name, _, plan_contact = arm.partition("@")
    plan_contact = plan_contact or "as_drawn"
    contract = load_contract(REPO_ROOT / f"logs/contract/{name}/generator.pt", device)
    generator, sigma = contract.generator, contract.sigma.to(device)
    dt_star = float(contract.meta["dt_star"])
    rng = np.random.default_rng(seed)
    entry: dict = {
        "path": f"logs/contract/{name}/generator.pt",
        "plan_contact": plan_contact,
        "groups": {},
    }
    for group, tasks in TASKS.items():
        commands = {}
        for task, rollout, names in tasks:
            for command_name in names:
                data = np.load(REPO_ROOT / f"g1/rollouts/{task}_{rollout}_{command_name}.npz")
                units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
                scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
                z = data["skill_state"].astype(np.float32)
                window = z[100 : 100 + generator.history, :STARTS]
                start = generator.skill_norm(torch.as_tensor(window, device=device).transpose(0, 1))
                command = torch.as_tensor(
                    data["command"][100, :STARTS] * scale, dtype=torch.float32, device=device
                )
                with torch.no_grad():
                    states = chain_states(
                        generator,
                        start,
                        command,
                        sigma,
                        steps,
                        generator.commit,
                        seed,
                        plan_contact=plan_contact,
                    )
                own = z[100 + generator.history :, :STARTS].astype(float)
                teacher = signature(own, dt_star)
                read = command_read(states, teacher, dt_star, draws, rng)
                read["teacher_slip"] = float(np.median(np.concatenate(firm_speeds(own))))
                read["teacher_newton_residual"], read["teacher_mean_load"] = newton(own, dt_star)
                commands[f"{task}:{command_name}"] = read
        slip = np.mean([c["_slip_draws"] for c in commands.values()], axis=0)
        calibration = np.mean([c["_calibration_draws"] for c in commands.values()], axis=0)
        entry["groups"][group] = {
            "slip": float(np.mean([c["slip"] for c in commands.values()])),
            "slip_se": float(slip.std(ddof=1)),
            "calibration": float(np.mean([c["calibration"] for c in commands.values()])),
            "calibration_se": float(calibration.std(ddof=1)),
            "newton_residual": float(np.mean([c["newton_residual"] for c in commands.values()])),
            "mean_load": float(np.mean([c["mean_load"] for c in commands.values()])),
            "teacher_slip": float(np.mean([c["teacher_slip"] for c in commands.values()])),
            "commands": {
                k: {n: v for n, v in c.items() if not n.startswith("_")}
                for k, c in commands.items()
            },
        }
    return entry


def gate(arm: dict, control: dict) -> dict:
    out, passes = {}, True
    for group in TASKS:
        a, c = arm["groups"][group], control["groups"][group]
        difference = a["calibration"] - c["calibration"]
        se = float(np.hypot(a["calibration_se"], c["calibration_se"]))
        slip_ok = a["slip"] <= SLIP_BOUND
        calibration_ok = difference <= 2.0 * se
        out[group] = {
            "slip": a["slip"],
            "slip_ok": bool(slip_ok),
            "calibration_minus_control": float(difference),
            "se": se,
            "calibration_ok": bool(calibration_ok),
        }
        passes = passes and slip_ok and calibration_ok
    out["passes"] = bool(passes)
    return out


def contrast(a: dict, b: dict) -> dict:
    """``a``'s calibration minus ``b``'s per group, with the SE of the difference."""
    out = {}
    for group in TASKS:
        x, y = a["groups"][group], b["groups"][group]
        out[group] = {
            "difference": float(x["calibration"] - y["calibration"]),
            "se": float(np.hypot(x["calibration_se"], y["calibration_se"])),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--control", required=True, help="the recipe without the loss")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--reference", nargs="*", default=[], help="reported, not gated")
    parser.add_argument(
        "--contrast",
        nargs=2,
        action="append",
        default=[],
        metavar=("A", "B"),
        help="report A's calibration minus B's with its SE (both among the arms)",
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {
        "steps": args.steps,
        "starts": STARTS,
        "bootstrap": args.bootstrap,
        "slip_bound": SLIP_BOUND,
        "light_load": LIGHT_LOAD,
        "control": args.control,
        "generators": {},
        "gate": {},
    }
    for name in [args.control, *args.arms, *args.reference]:
        report["generators"][name] = read_generator(
            name, args.steps, args.bootstrap, args.seed, args.device
        )
    for name in args.arms:
        report["gate"][name] = gate(report["generators"][name], report["generators"][args.control])
    report["contrasts"] = {
        f"{a} - {b}": contrast(report["generators"][a], report["generators"][b])
        for a, b in args.contrast
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for name, entry in report["generators"].items():
        for group, g in entry["groups"].items():
            print(
                f"{name:24s} {group:8s} slip {g['slip']:.4f} ± {g['slip_se']:.4f} V "
                f"(teacher {g['teacher_slip']:.4f})  calibration {g['calibration']:.3f} ± "
                f"{g['calibration_se']:.3f}  newton {g['newton_residual']:.3f} g  "
                f"load {g['mean_load']:.3f}"
            )
    for name, g in report["gate"].items():
        verdict = "PASSES" if g["passes"] else "FAILS"
        detail = "; ".join(
            f"{group}: slip {g[group]['slip']:.4f} (ok {g[group]['slip_ok']}), calibration "
            f"{g[group]['calibration_minus_control']:+.3f} ± {g[group]['se']:.3f}"
            for group in TASKS
        )
        print(f"G-P {name}: {verdict} ({detail})")
    for name, c in report["contrasts"].items():
        detail = "; ".join(
            f"{group}: calibration {c[group]['difference']:+.3f} ± {c[group]['se']:.3f}"
            for group in TASKS
        )
        print(f"contrast {name}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
