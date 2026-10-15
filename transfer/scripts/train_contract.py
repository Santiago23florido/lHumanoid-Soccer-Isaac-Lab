"""Train the contract generator of one task on its teacher's captures, and gate it.

The task side of the contract (:mod:`humanoid_transfer.transfer.latent.contract`):
a generator that reads the history of ``z*`` and the dimensionless command
and samples the next ``z*``, learned from what a teacher's body *did*, never
from what it commanded. Before it, stage A trains the body's encoder on the
anchor alone (as for the memory model) and measures its error, which becomes
the noise the generator is trained under.

The gate, fixed before training: on held-out walking, the instant each
foot's contact next changes must be calibrated in the generator's samples to
within 0.05 -- the same bound the contact planner was held to.

Results go to ``transfer/results/contract_<name>.json``; the generator and
encoder to ``logs/contract/<name>/generator.pt`` (untracked), which the
tracking task loads.

Usage::

    python transfer/scripts/train_contract.py --name walk \
        --data g1/rollouts/teacher_dart.npz --gait g1/rollouts/teacher.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import TIME, UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    Contract,
    ContractClock,
    ContractDims,
    load_contract,
    save_contract,
)
from humanoid_transfer.transfer.latent.events import contact_events  # noqa: E402
from humanoid_transfer.transfer.latent.model import G1_OBSERVATION  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_contract import (  # noqa: E402
    ChainConfig,
    ContractConfig,
    chain_fidelity,
    chained_gait,
    contract_steps,
    evaluate_generator,
    event_categories,
    fit_tolerance,
    gait_reference,
    new_generator,
    population_scores,
    train_generator,
    train_on_chains,
)
from humanoid_transfer.transfer.latent.train_memory import (  # noqa: E402
    MemoryConfig,
    residual_sigma,
    train_encoder,
)

CALIBRATION_LIMIT = 0.05


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="walk", help="The task this contract describes.")
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Teacher-driven capture; repeat to aggregate. The first one is split for "
        "validation. Default: g1/rollouts/teacher_dart.npz.",
    )
    parser.add_argument("--gait", type=Path, default=REPO_ROOT / "g1/rollouts/teacher.npz")
    parser.add_argument(
        "--encoder-from",
        type=Path,
        default=None,
        help="A body's existing contract (generator.pt): reuse its encoder and its units of z* "
        "instead of training an encoder -- a new task on a body that already has one.",
    )
    parser.add_argument(
        "--sigma-from-body",
        action="store_true",
        help="With --encoder-from: train under the noise of the body's contract (its encoder's "
        "error on its own task) instead of that encoder's error on this task's states, which "
        "says how far the encoder is from covering the task, not how precisely a body that "
        "covers it would read the state. Both are reported.",
    )
    parser.add_argument("--val-envs", type=int, default=16)
    parser.add_argument("--encoder-epochs", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--free-bits", type=float, default=0.1)
    parser.add_argument("--kind", choices=("flow", "cvae"), default="flow")
    parser.add_argument("--flow-steps", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=13, help="Samples of future per plan.")
    parser.add_argument("--commit", type=int, default=6, help="Samples a plan is held.")
    parser.add_argument("--view", type=int, default=8, help="Samples a tracker sees ahead.")
    parser.add_argument("--solver", choices=("euler", "midpoint"), default="euler")
    parser.add_argument(
        "--warp-loads",
        action="store_true",
        help="Model foot loads as log(load + 0.01): forces are never negative.",
    )
    parser.add_argument("--memory", type=int, default=64, help="Width of the history summary.")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--flow-hidden", type=int, default=1024)
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplies the encoder error used as training noise; 0 trains on clean history.",
    )
    parser.add_argument(
        "--chain-rounds",
        type=int,
        default=0,
        help="Rounds of training on the generator's own chains (Data as Demonstrator); 0: none.",
    )
    parser.add_argument("--chain-plans", type=int, default=6, help="Plans per chain.")
    parser.add_argument("--chain-starts", type=int, default=8192, help="Chains per round.")
    parser.add_argument("--chain-epochs", type=int, default=300, help="Epochs after each round.")
    parser.add_argument(
        "--chain-constraint",
        choices=("calibration", "data"),
        default="calibration",
        help="What the chains' weight is held by: the plans' timing calibration against its "
        "gate, or the held-out data loss within twice its own error (too tight to move).",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=Path, default=REPO_ROOT / "logs/contract")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    paths = args.data or [REPO_ROOT / "g1/rollouts/teacher_dart.npz"]
    results = args.results or REPO_ROOT / f"transfer/results/contract_{args.name}.json"
    archive = np.load(paths[0])
    drivers = [str(np.load(p)["driver"]) for p in paths]
    if any(driver != "teacher" for driver in drivers):
        raise SystemExit(f"a contract is learned from teacher-driven captures only: {drivers}")
    units = UnitSystem(float(archive["leg_length"]), float(archive["total_mass"]))
    dt_star = float(units.to_dimensionless(float(archive["control_dt"]), TIME))

    streams = load_streams(paths).to(args.device)
    gait = load_streams([args.gait]).to(args.device)
    envs = int(archive["done"].shape[1])
    held_out = (streams.capture == 0) & (streams.env >= envs - args.val_envs)

    config = ContractConfig(
        encoder_epochs=args.encoder_epochs,
        epochs=args.epochs,
        free_bits=args.free_bits,
        dims=ContractDims(
            kind=args.kind,
            flow_steps=args.flow_steps,
            flow_solver=args.solver,
            warp_loads=args.warp_loads,
            memory=args.memory,
            channels=args.channels,
            flow_hidden=args.flow_hidden,
        ),
        seed=args.seed,
    )
    clock = ContractClock(step=dt_star, horizon=args.horizon, commit=args.commit, view=args.view)
    planned = contract_steps(streams, clock.horizon)
    train_steps = planned[~held_out[planned]]
    val_steps = planned[held_out[planned]]
    trustworthy = streams.samples()
    encoder_steps = trustworthy[~held_out[trustworthy]]
    walking = trustworthy[held_out[trustworthy] & ~streams.falling[trustworthy]]
    gait_steps = contract_steps(gait, clock.horizon)

    started = time.time()
    if args.encoder_from is None:
        generator = new_generator(streams, train_steps, config, clock)
        encoder, encoder_history = train_encoder(
            streams,
            encoder_steps,
            G1_OBSERVATION,
            generator,
            MemoryConfig(encoder_epochs=args.encoder_epochs, seed=args.seed),
        )
        final_anchor = encoder_history[-1]["anchor"]
    else:
        body = load_contract(args.encoder_from, args.device)
        encoder = body.encoder
        generator = new_generator(
            streams, train_steps, config, clock, skill_norm=body.generator.skill_norm
        )
        final_anchor = None
    # Measured on this task's held-out states either way: an encoder is only as
    # good as it is on the states a task visits.
    sigma = residual_sigma(encoder, generator, G1_OBSERVATION, streams, walking)
    sigma_on_task = sigma
    if args.encoder_from is not None and args.sigma_from_body:
        sigma = body.sigma.to(args.device)
    encoder_seconds = time.time() - started
    print(f"[encoder] {encoder_seconds:.0f} s, sigma median {float(sigma.median()):.3f}")

    # What a perfect tracker would walk: the generator chained on its own plans
    # from 32 of the teacher's histories, at the teacher's command. Added after
    # the first closed loop showed that calibrated plans can chain into a gait
    # that steps a quarter faster than the teacher's.
    gait_z = np.load(args.gait)["skill_state"].astype(np.float32)
    start = torch.as_tensor(gait_z[100 : 100 + clock.history], device=args.device)
    start = generator.skill_norm(start.transpose(0, 1))
    command = gait.command[: start.shape[0]].mean(dim=0, keepdim=True).expand(len(start), 3)
    teacher_z = gait_z[100 + clock.history :].astype(float)

    def fidelity() -> dict:
        return chain_fidelity(
            generator, start, command, sigma, 900, clock.commit, dt_star, teacher_z, args.seed
        )

    events = contact_events(streams)
    gait_events = contact_events(gait)
    population = population_scores(
        event_categories(events, train_steps, clock.horizon), clock.horizon
    )
    # The calibration gate's own measurement, on a fixed quarter of the held-out
    # steps with fewer samples: cheap enough to check during training.
    check_steps = val_steps[
        torch.randperm(len(val_steps), generator=torch.Generator().manual_seed(args.seed))[
            : max(1, len(val_steps) // 4)
        ].to(val_steps.device)
    ]

    def calibration() -> float:
        block = evaluate_generator(
            generator, streams, check_steps, events, population, None, samples=16, seed=args.seed
        )
        return float(max(block["calibration_error"]))

    started = time.time()
    history = train_generator(streams, train_steps, generator, args.noise_scale * sigma, config)
    chain = ChainConfig(
        rounds=args.chain_rounds,
        plans=args.chain_plans,
        starts=args.chain_starts,
        epochs=args.chain_epochs,
    )
    chain_log = []
    if chain.rounds:
        chain_log = train_on_chains(
            streams,
            train_steps,
            generator,
            args.noise_scale * sigma,
            config,
            chain,
            fidelity,
            val_steps,
            calibration if args.chain_constraint == "calibration" else None,
            CALIBRATION_LIMIT,
        )
        print(f"[chains] {chain_log[-1]}")
    seconds = time.time() - started
    tolerance = fit_tolerance(generator, streams, val_steps, sigma, seed=args.seed)
    print(f"[generator] {seconds:.0f} s, final {history[-1]}")

    def measure(on, steps, labels, noise):
        return evaluate_generator(
            generator, on, steps, labels, population, noise, samples=args.samples, seed=args.seed
        )

    evaluation = {
        "validation_walking": measure(streams, val_steps, events, None),
        "validation_walking_noisy_history": measure(streams, val_steps, events, sigma),
        "gait": measure(gait, gait_steps, gait_events, None),
        "gait_noisy_history": measure(gait, gait_steps, gait_events, sigma),
    }
    chained = chained_gait(
        generator, start, command, sigma, 900, clock.commit, dt_star, seed=args.seed
    )
    reference = gait_reference(gait_z[100 + clock.history :].astype(float), dt_star)
    deviation = {
        key: abs(chained["signature"][key] - reference["signature"][key])
        / reference["signature"][key]
        for key in ("stride_period", "stride_length")
    }
    deviation["rhythm_period"] = (
        abs(chained["rhythm_period"] - reference["rhythm_period"]) / reference["rhythm_period"]
    )
    calibration = evaluation["validation_walking"]["calibration_error"]
    passes = all(value <= CALIBRATION_LIMIT for value in calibration)
    # Added after the blind test, where every plan was calibrated while their chain
    # stepped 17-20 % fast: how far the chain walks from the teacher, with its own
    # standard error. A measurement, not a gate; the body's gates decide.
    chain_measured = fidelity()

    path = args.logdir / args.name / "generator.pt"
    meta = {
        "name": args.name,
        "dt_star": dt_star,
        "leg_length": units.length,
        "total_mass": units.mass,
        "control_dt": float(archive["control_dt"]),
        "captures": [_display(p) for p in paths],
        "teacher_sha256": str(archive["teacher_sha256"]),
    }
    save_contract(
        path,
        Contract(generator, encoder, sigma, meta),
        config.dims,
        clock,
        G1_OBSERVATION.encoder_size,
    )

    report = {
        "name": args.name,
        "data": {
            "training_captures": meta["captures"],
            "drivers": drivers,
            "gait_capture": _display(args.gait),
            "train_steps": len(train_steps),
            "validation_steps": len(val_steps),
            "gait_steps": len(gait_steps),
            "validation_envs": args.val_envs,
            "teacher_sha256": meta["teacher_sha256"],
            "dt_star": dt_star,
        },
        "clock": {
            "step": clock.step,
            "history": clock.history,
            "horizon": clock.horizon,
            "commit": clock.commit,
            "view": clock.view,
        },
        "config": {
            "encoder_epochs": config.encoder_epochs,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "free_bits": config.free_bits,
            "intent": config.dims.intent,
            "kind": config.dims.kind,
            "noise_scale": args.noise_scale,
            "memory": config.dims.memory,
            "channels": config.dims.channels,
            "flow_hidden": config.dims.flow_hidden,
            "flow_steps": config.dims.flow_steps,
            "warp_loads": config.dims.warp_loads,
            "seed": config.seed,
        },
        "encoder": {
            "seconds": encoder_seconds,
            "source": _display(args.encoder_from) if args.encoder_from else "trained here",
            "final_anchor_loss": final_anchor,
            "sigma": sigma.tolist(),
            "sigma_median": float(sigma.median()),
            "sigma_on_this_task_median": float(sigma_on_task.median()),
            "sigma_from_body": bool(args.sigma_from_body),
        },
        "generator": {
            "seconds": seconds,
            "final": history[-1],
            "chains": {"config": vars(chain), "rounds": chain_log},
            "kind": generator.kind,
            "tolerance_by_k": tolerance.mean(dim=-1).tolist(),
            "tolerance": tolerance.tolist(),
        },
        "evaluation": evaluation,
        "chained": {"generator": chained, "teacher": reference, "deviation": deviation},
        "gate": {
            "rule": f"calibration_error <= {CALIBRATION_LIMIT} for each foot, held-out walking, "
            "clean history",
            "calibration_error": calibration,
            "passes": passes,
        },
        "chain_fidelity": chain_measured,
        "artifact": _display(path),
    }
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, block in evaluation.items():
        print(
            f"  {name:34s} calibration {np.round(block['calibration_error'], 3)}  "
            f"timing gain {np.round(block['timing_nll_gain'], 3)} nats  "
            f"R2 mean@K {block['r2_sample_mean_by_k'][-1]:.3f} "
            f"(persistence {block['r2_persistence_by_k'][-1]:.3f})  "
            f"KL {sum(block['kl_per_intent_dim'] or [0.0]):.2f}"
        )
    print(
        f"  chained: period {chained['signature']['stride_period']:.3f} "
        f"(teacher {reference['signature']['stride_period']:.3f}), rhythm "
        f"{chained['rhythm_period']:.3f} ({reference['rhythm_period']:.3f}), swing median "
        f"{chained['segments']['swing']['median']} ({reference['segments']['swing']['median']}), "
        f"stable {chained['environments_stable']}/{chained['environments']}"
    )
    print(
        f"  chain fidelity: score {chain_measured['score']:.3f} +- "
        f"{chain_measured['score_se']:.3f} (period "
        f"{chain_measured['deviation']['stride_period']:.3f}, stride "
        f"{chain_measured['deviation']['stride_length']:.3f})"
    )
    print(f"gate {'PASSES' if passes else 'FAILS'}; written {_display(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
