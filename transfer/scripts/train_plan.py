"""Train the shared skill with a contact schedule, and gate it before the simulator.

Stage A trains the G1 encoder on the anchor alone. Stage B then trains two
variants on the same rollouts, seed and noise, each with the shared contact
planner in the loop (:mod:`humanoid_transfer.transfer.latent.plan`):

``plan``
    Memory, planner, policy and dynamics shared; one decoder for all joints.
``plan_limbs``
    The same with a decoder per limb.

With ``--consistency`` (the default, 0.02: the same order as the imitation
loss), the policy is also held to plans drawn from the planner through the
latent dynamics -- the contacts ``T`` predicts under its actions must change
when the plan says. Without it, the first run found the plan calibrated but
unused.

Rollouts whose ``driver`` is the teacher give the policy the schedule that
actually followed, and teach the planner; rollouts a student drove give the
policy a schedule drawn from the planner and teach the planner nothing.

**Gate, fixed before the first run.** On held-out walking of the first
capture, through the deployed path (encoder, memory, planner):

1. calibration error at most ``CALIBRATION_LIMIT`` for each foot -- a drawn
   schedule is only the teacher's if the planner's probabilities are;
2. imitation with the schedule that followed at least ``PLAN_USE_MIN`` above
   imitation with the schedule held at its mean -- a schedule the policy
   ignores cannot fix its timing.

Results go to ``transfer/results/latent_g1_plan.json``; weights and exports
to ``logs/latent/<variant><tag>/`` (untracked): ``policy.pt`` draws its
schedule, ``policy_frozen_plan.pt`` holds it at its mean,
``policy_bypass.pt`` holds ``u`` at its mean.

Usage::

    python transfer/scripts/train_plan.py \
        --data g1/rollouts/teacher_dart.npz --data g1/rollouts/dagger_1.npz ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import TIME, UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.events import contact_events  # noqa: E402
from humanoid_transfer.transfer.latent.model import (  # noqa: E402
    G1_OBSERVATION,
    FrozenActionPolicy,
)
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_memory import (  # noqa: E402
    MemoryModel,
    config_dict,
    estimates,
    fit_skill_normaliser,
    leakage_probes,
    new_decoder,
    new_skill,
    residual_sigma,
    train_encoder,
)
from humanoid_transfer.transfer.latent.train_plan import (  # noqa: E402
    PlannedModel,
    column_rollouts,
    evaluate_plan,
    hindsight_plans,
    new_planner,
    plan_config,
    plan_following,
    population_hazard,
    runtime_imitation,
    schedule_inputs,
    train_planned,
)

CALIBRATION_LIMIT = 0.05
PLAN_USE_MIN = 0.02

BODY_JOINT_POSITIONS = slice(12, 35)
"""The 23 body joints' positions in the teacher's observation (hands excluded)."""

PROBE_FIT_STEPS = 100_000


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _check_export(policy, scripted, obs: torch.Tensor, done: torch.Tensor) -> float:
    """Largest difference between the eager and scripted policies, on the same draws."""
    worst = 0.0
    with torch.no_grad():
        for t in range(obs.shape[0]):
            torch.manual_seed(t)
            eager = policy(obs[t])
            torch.manual_seed(t)
            worst = max(worst, float((eager - scripted(obs[t])).abs().max()))
            policy.reset(done[t])
            scripted.reset(done[t])
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Training capture; repeat to aggregate. The first one is split for validation "
        "and must be teacher-driven. Default: g1/rollouts/teacher_dart.npz.",
    )
    parser.add_argument("--gait", type=Path, default=REPO_ROOT / "g1/rollouts/teacher.npz")
    parser.add_argument("--val-envs", type=int, default=16)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--encoder-epochs", type=int, default=150)
    parser.add_argument("--skill-epochs", type=int, default=150)
    parser.add_argument("--consistency", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=Path, default=REPO_ROOT / "logs/latent")
    parser.add_argument(
        "--results", type=Path, default=REPO_ROOT / "transfer/results/latent_g1_plan.json"
    )
    args = parser.parse_args()

    paths = args.data or [REPO_ROOT / "g1/rollouts/teacher_dart.npz"]
    archive = np.load(paths[0])
    units = UnitSystem(float(archive["leg_length"]), float(archive["total_mass"]))
    dt_star = float(units.to_dimensionless(float(archive["control_dt"]), TIME))
    joint_names = [str(name) for name in archive["joint_names"]]
    layout = G1_OBSERVATION
    drivers = [str(np.load(path)["driver"]) for path in paths]
    if drivers[0] != "teacher":
        raise SystemExit("the first capture is the validation split and must be teacher-driven")

    streams = load_streams(paths).to(args.device)
    gait = load_streams([args.gait]).to(args.device)
    events = contact_events(streams)
    gait_events = contact_events(gait)
    teacher_driven = torch.tensor([d == "teacher" for d in drivers], device=args.device)
    hindsight = teacher_driven[streams.capture]
    envs = int(archive["done"].shape[1])
    capture_steps = int(archive["done"].shape[0])
    held_out = (streams.capture == 0) & (streams.env >= envs - args.val_envs)

    base = plan_config(
        encoder_epochs=args.encoder_epochs,
        skill_epochs=args.skill_epochs,
        seed=args.seed,
        consistency=args.consistency,
        temperature=args.temperature,
    )
    trustworthy = streams.samples()
    train_steps = trustworthy[~held_out[trustworthy]]
    val_steps = trustworthy[held_out[trustworthy]]
    walking = val_steps[~streams.falling[val_steps]]
    rollout_steps = streams.samples(future=base.horizon)
    rollout_steps = rollout_steps[~held_out[rollout_steps]]
    gait_steps = gait.samples()
    teacher_train = train_steps[hindsight[train_steps]]

    report: dict = {
        "data": {
            "training_captures": [_display(p) for p in paths],
            "drivers": drivers,
            "gait_capture": _display(args.gait),
            "train_steps": len(train_steps),
            "train_steps_teacher_driven": len(teacher_train),
            "validation_steps": len(val_steps),
            "validation_falling_steps": len(val_steps) - len(walking),
            "validation_envs": args.val_envs,
            "teacher_sha256": str(archive["teacher_sha256"]),
            "dt_star": dt_star,
        },
        "gate": {
            "rule": f"calibration_error <= {CALIBRATION_LIMIT} for each foot and "
            f"plan_use >= {PLAN_USE_MIN}, held-out walking, deployed path",
        },
        "variants": {},
    }

    # Stage A, once: the encoder and its error, shared by both variants.
    skill_norm = fit_skill_normaliser(streams, train_steps, base.dims)
    lender = new_skill(streams, train_steps, skill_norm, base, dt_star)
    started = time.time()
    encoder, encoder_history = train_encoder(streams, train_steps, layout, lender, base)
    sigma = residual_sigma(encoder, lender, layout, streams, walking)
    report["encoder"] = {
        "train_seconds": round(time.time() - started, 1),
        "final_anchor_loss": encoder_history[-1]["anchor"],
        "sigma_normalised": sigma.tolist(),
        "sigma_median": float(sigma.median()),
    }
    print(f"[encoder] {report['encoder']['train_seconds']:.0f} s, sigma median "
          f"{report['encoder']['sigma_median']:.3f}")  # fmt: skip
    estimate = estimates(encoder, layout, streams)
    gait_estimate = estimates(encoder, layout, gait)

    val_envs = torch.arange(envs - args.val_envs, envs, device=args.device)
    val_obs, val_action, _, val_valid = column_rollouts(streams, capture_steps, val_envs)
    val_done = torch.as_tensor(archive["done"][:, envs - args.val_envs :], device=args.device)

    variants = {"plan": base, "plan_limbs": replace(base, limbs=True)}
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    probe_fit = train_steps[torch.randperm(len(train_steps), generator=generator)[:PROBE_FIT_STEPS]]
    teacher_fit = probe_fit[hindsight[probe_fit]]

    for name, config in variants.items():
        started = time.time()
        skill = new_skill(streams, train_steps, skill_norm, config, dt_star)
        planner = new_planner(events, teacher_train, config, dt_star)
        decoder = new_decoder(streams, train_steps, layout, config, joint_names)
        history = train_planned(
            streams, events, hindsight, rollout_steps, layout, skill, planner, decoder, sigma,
            config, dt_star,
        )  # fmt: skip
        seconds = time.time() - started
        model = PlannedModel(encoder, skill, planner, decoder, layout, sigma).eval()

        population = population_hazard(model, streams, events, teacher_train, dt_star)
        mean_plan = hindsight_plans(model, streams, events, teacher_fit, dt_star).mean(dim=0)
        with torch.no_grad():
            memory, _, _ = schedule_inputs(model, streams, estimate, probe_fit)
            plans = hindsight_plans(model, streams, events, probe_fit, dt_star)
            mean_action = model.skill.act(
                estimate[probe_fit], memory, streams.command[probe_fit], plans
            ).mean(dim=0)

        references = (population, mean_plan, mean_action)
        metrics = {
            split: evaluate_plan(model, *source, dt_star, *references)
            for split, source in (
                ("validation", (streams, events, val_steps, estimate)),
                ("validation_walking", (streams, events, walking, estimate)),
                ("gait", (gait, gait_events, gait_steps, gait_estimate)),
            )
        }
        metrics["plan_following"] = {
            split: plan_following(
                model,
                source_streams,
                source_events,
                steps,
                source_estimate,
                dt_star,
                mean_plan,
                config.horizon,
                config.temperature,
                seed=args.seed,
            )  # fmt: skip
            for split, (source_streams, source_events, steps, source_estimate) in (
                ("validation_walking", (streams, events, walking, estimate)),
                ("gait", (gait, gait_events, gait_steps, gait_estimate)),
            )
        }
        metrics["runtime_validation_walking"] = {
            mode: runtime_imitation(
                model.policy(units.velocity, units.time, dt_star, stochastic=mode == "drawn").to(
                    args.device
                ),
                val_obs,
                val_done,
                val_action,
                val_valid,
                seed=args.seed,
            )  # fmt: skip
            for mode in ("drawn", "median")
        }
        leakage = leakage_probes(
            MemoryModel(encoder, skill, decoder, layout, sigma),
            streams, probe_fit, walking, estimate, BODY_JOINT_POSITIONS,
        )  # fmt: skip

        walk = metrics["validation_walking"]
        gate = {
            "calibration_error": walk["calibration_error"],
            "plan_use": walk["plan_use"],
            "passes": max(walk["calibration_error"]) <= CALIBRATION_LIMIT
            and walk["plan_use"] >= PLAN_USE_MIN,
        }

        out = args.logdir / f"{name}{args.tag}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state": model.state_dict(), "config": config_dict(config), "dt_star": dt_star},
            out / "model.pt",
        )
        model.to("cpu")
        example = gait.obs[: 40 * 8].reshape(40, 8, -1).cpu()  # any sequence will do
        example_done = torch.zeros(40, 8, dtype=torch.bool)
        example_done[20, 3] = True
        policy = model.policy(units.velocity, units.time, dt_star)
        scripted = torch.jit.script(model.policy(units.velocity, units.time, dt_star))
        export_error = _check_export(policy, scripted, example, example_done)
        assert export_error < 1e-4, export_error
        scripted.save(str(out / "policy.pt"))
        torch.jit.script(
            model.policy(units.velocity, units.time, dt_star, plan=mean_plan.cpu())
        ).save(str(out / "policy_frozen_plan.pt"))
        frozen = FrozenActionPolicy(model.decoder, layout, mean_action.cpu()).eval()
        torch.jit.trace(frozen, example[0]).save(str(out / "policy_bypass.pt"))
        model.to(args.device)

        report["variants"][name] = {
            "config": config_dict(config),
            "train_seconds": round(seconds, 1),
            "final_losses": {k: v for k, v in history[-1].items() if k != "epoch"},
            "gate": gate,
            "metrics": metrics,
            "leakage_validation_walking": leakage,
            "export_max_error": export_error,
            "policy_export": _display(out / "policy.pt"),
            "frozen_plan_export": _display(out / "policy_frozen_plan.pt"),
            "bypass_export": _display(out / "policy_bypass.pt"),
        }
        runtime = metrics["runtime_validation_walking"]
        print(f"[{name}] {seconds:.0f} s  gate {'PASSES' if gate['passes'] else 'FAILS'}")
        print(f"  calibration error  {[round(e, 3) for e in walk['calibration_error']]}")
        print(f"  nll gain (nats)    {[round(g, 3) for g in walk['nll_gain_nats']]}"
              f"  median-bin R2 {[round(g, 3) for g in walk['median_bin_r2']]}")  # fmt: skip
        print(f"  imitation R2       followed {walk['imitation_r2_followed_plan']:.3f}  "
              f"drawn {walk['imitation_r2_drawn_plan']:.3f}  "
              f"frozen {walk['imitation_r2_frozen_plan']:.3f}  "
              f"(use {walk['plan_use']:+.3f})")  # fmt: skip
        print(f"  running policy     drawn {runtime['drawn']:.3f}  median {runtime['median']:.3f}")
        follow = metrics["plan_following"]["validation_walking"]
        print(f"  plan followed      at change {follow['agreement_at_planned_change']:.3f}  "
              f"withheld {follow['agreement_at_planned_change_plan_withheld']:.3f}")  # fmt: skip
        print(f"  bypass u           {walk['bypass_action_r2']:.3f}")
        print(f"  joint probes       excess {leakage['excess_mlp']:+.3f}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {_display(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
