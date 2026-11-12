"""Does the flow sampler blur contact transitions? Plans at several integrators against the teacher.

The task generators' plans ask for light contact (above the 5 % threshold, at most a
quarter of the weight) three times as often as the flat teacher shows it. A flow sampled with too
few steps lands between modes -- a load half-way between
stance and swing -- so the first suspect is the integrator, which costs no training to
change. For each integrator this measures, on the teacher's own play histories:

* one plan from each window, against the future the teacher actually walked: the share of
  light contact, the duty factor, and contact changes per plan (as calibrated for the contract);
* the generator's own chain at the G1's rate (``chain_at_rate.py``, autonomous reference):
  the same, plus stride period and length;
* the time a plan takes, on this machine's CPU (relative cost only).

    python transfer/scripts/sampler_sharpness.py \
        --contract logs/contract/flat_dagger2_zs_g1_warm_r3_logload_sym/generator.pt \
        --teacher g1/rollouts/flat_play_forward.npz \
        --capture g1/rollouts/zs_warm_r3_logload_sym_flat_dagger2_auto_kin_forward_encoder.npz \
        --output transfer/results/sampler_sharpness_flat_forward.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chain_at_rate import chain_at_rate, summary  # noqa: E402
from humanoid_transfer.common.contact import LIGHT_LOAD  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402

LOADS = SKILL_LAYOUT.slice_of("foot_load")
INTEGRATORS = (("midpoint", 8), ("midpoint", 16), ("midpoint", 32), ("euler", 16), ("euler", 64))


def load_stats(loads: np.ndarray) -> dict:
    """``loads`` ``(..., K, 2)``: shares over every sample, contact changes per plan."""
    contact = loads > DEFAULT_CONTACT_THRESHOLD
    light = contact & (loads <= LIGHT_LOAD)
    changes = (contact[..., 1:, :] != contact[..., :-1, :]).sum(axis=-2)
    return {
        "duty": contact.reshape(-1, 2).mean(axis=0).tolist(),
        "light_share": light.reshape(-1, 2).mean(axis=0).tolist(),
        "changes_per_plan": changes.reshape(-1, 2).mean(axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True, help="a teacher play capture")
    parser.add_argument("--capture", type=Path, required=True, help="an L4 capture: chain starts")
    parser.add_argument("--stride", type=int, default=7, help="steps between windows")
    parser.add_argument(
        "--sigma-from",
        type=Path,
        default=None,
        help="a contract whose sigma is the history noise (default: the contract's own)",
    )
    parser.add_argument(
        "--integrators", nargs="*", default=None, help="e.g. midpoint8 euler64 (default: all)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    contract = load_contract(args.contract, "cpu")
    generator = contract.generator
    if args.sigma_from is not None:
        contract.sigma = load_contract(args.sigma_from, "cpu").sigma
    integrators = INTEGRATORS
    if args.integrators:
        integrators = [
            (i.rstrip("0123456789"), int(i[len(i.rstrip("0123456789")) :]))
            for i in args.integrators
        ]
    history, horizon = generator.history, generator.horizon
    data = np.load(args.teacher)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    scale = np.array([1.0 / units.velocity, 1.0 / units.velocity, units.time])
    z = data["skill_state"].astype(np.float32)
    done = data["done"].astype(bool)
    windows, futures, commands = [], [], []
    for env in range(z.shape[1]):
        for t in range(100 + history, z.shape[0] - horizon, args.stride):
            if done[t - history : t + horizon, env].any():
                continue
            windows.append(z[t - history + 1 : t + 1, env])
            futures.append(z[t + 1 : t + 1 + horizon, env])
            commands.append(data["command"][t, env] * scale)
    window = generator.skill_norm(torch.as_tensor(np.stack(windows)))
    window = window + contract.sigma * torch.randn_like(window)
    command = torch.as_tensor(np.stack(commands), dtype=torch.float32)
    future = np.stack(futures)
    report = {
        "contract": args.contract.as_posix(),
        "teacher": args.teacher.as_posix(),
        "windows": len(window),
        "teacher_future": load_stats(future[..., LOADS]),
        "integrators": {},
    }

    capture = np.load(args.capture)
    capture_units = UnitSystem(float(capture["leg_length"]), float(capture["total_mass"]))
    dt_star = float(capture["control_dt"]) / capture_units.time
    start = generator.skill_norm(torch.as_tensor(capture["skill_state"][1], dtype=torch.float32))
    capture_scale = torch.tensor(
        [1.0 / capture_units.velocity, 1.0 / capture_units.velocity, capture_units.time]
    )
    chain_command = torch.as_tensor(capture["command"][1], dtype=torch.float32) * capture_scale
    for solver, steps in integrators:
        generator.flow_solver, generator.flow_steps = solver, steps
        noise = torch.randn(
            len(window), generator.noise_size, generator=torch.Generator().manual_seed(1)
        )
        began = time.perf_counter()
        with torch.no_grad():
            plans = generator.sample(window, command, noise).float()
        seconds = time.perf_counter() - began
        plain = generator.skill_norm.inverse(plans).numpy()
        chain = chain_at_rate(
            generator, start, chain_command, contract.sigma, dt_star,
            capture["skill_state"].shape[0], "linear", seed=args.seed,
        )  # fmt: skip
        chained = generator.skill_norm.inverse(chain).numpy().astype(float)
        report["integrators"][f"{solver}{steps}"] = {
            "one_step": load_stats(plain[..., LOADS]),
            "chain": summary(chained, dt_star),
            "ms_per_plan_cpu": 1000.0 * seconds / len(window),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def fmt(v) -> str:
        return "/".join(f"{x:.3f}" for x in v)

    t = report["teacher_future"]
    print(f"{args.teacher.name}: {len(window)} windows")
    print(f"  teacher future   light {fmt(t['light_share'])}  duty {fmt(t['duty'])}  "
          f"changes {fmt(t['changes_per_plan'])}")  # fmt: skip
    for name, r in report["integrators"].items():
        o, c = r["one_step"], r["chain"]
        print(
            f"  {name:<11} plan light {fmt(o['light_share'])} duty {fmt(o['duty'])} "
            f"changes {fmt(o['changes_per_plan'])} | chain light {fmt(c['light_share'])} "
            f"duty {fmt(c['duty_factor'])} period {c['stride_period']:.3f} | "
            f"{r['ms_per_plan_cpu']:.3f} ms"
        )
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
