"""Train the shared skill with a memory that reads only the physical state.

Stage A trains the G1 encoder on the anchor alone, once. Stage B then trains
two variants on the same rollouts, seed and noise:

``memory``
    Memory, policy and latent dynamics shared; one decoder for all joints.
``memory_limbs``
    The same with a decoder per limb, each reading its own share of ``u``.

Both are evaluated on held-out environments of the first capture and on the
clean teacher gait, and exported as scripted policies with their history
buffer (observation in, joint targets out, ``reset(done)`` at episode ends)
for ``g1/scripts/capture_rollouts.py --driver-policy``, together with the two
bypass policies: ``u`` held at its mean, and ``φ`` held at its mean.

Results go to ``transfer/results/latent_g1_memory.json``; weights and exports
to ``logs/latent/<variant><tag>/`` (untracked).

Usage::

    python transfer/scripts/train_memory.py \
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
from humanoid_transfer.transfer.latent.model import (  # noqa: E402
    G1_OBSERVATION,
    FrozenActionPolicy,
)
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402
from humanoid_transfer.transfer.latent.train_memory import (  # noqa: E402
    MemoryConfig,
    MemoryModel,
    act,
    config_dict,
    estimates,
    evaluate_memory,
    fit_skill_normaliser,
    leakage_probes,
    new_decoder,
    new_skill,
    residual_sigma,
    train_encoder,
    train_skill,
)

BODY_JOINT_POSITIONS = slice(12, 35)
"""The 23 body joints' positions in the teacher's observation (hands excluded)."""

PROBE_FIT_STEPS = 100_000


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _check_export(policy, scripted, obs: torch.Tensor, done: torch.Tensor) -> float:
    """Largest difference between the eager and scripted policies over a rollout."""
    worst = 0.0
    with torch.no_grad():
        for t in range(obs.shape[0]):
            worst = max(worst, float((policy(obs[t]) - scripted(obs[t])).abs().max()))
            policy.reset(done[t])
            scripted.reset(done[t])
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        help="Training capture; repeat to aggregate. The first one is split for validation. "
        "Default: g1/rollouts/teacher_dart.npz.",
    )
    parser.add_argument("--gait", type=Path, default=REPO_ROOT / "g1/rollouts/teacher.npz")
    parser.add_argument("--val-envs", type=int, default=16)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--encoder-epochs", type=int, default=150)
    parser.add_argument("--skill-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=Path, default=REPO_ROOT / "logs/latent")
    parser.add_argument(
        "--results", type=Path, default=REPO_ROOT / "transfer/results/latent_g1_memory.json"
    )
    args = parser.parse_args()

    paths = args.data or [REPO_ROOT / "g1/rollouts/teacher_dart.npz"]
    archive = np.load(paths[0])
    units = UnitSystem(float(archive["leg_length"]), float(archive["total_mass"]))
    dt_star = float(units.to_dimensionless(float(archive["control_dt"]), TIME))
    joint_names = [str(name) for name in archive["joint_names"]]
    layout = G1_OBSERVATION

    streams = load_streams(paths).to(args.device)
    gait = load_streams([args.gait]).to(args.device)
    envs = int(archive["done"].shape[1])
    held_out = (streams.capture == 0) & (streams.env >= envs - args.val_envs)

    base = MemoryConfig(
        encoder_epochs=args.encoder_epochs, skill_epochs=args.skill_epochs, seed=args.seed
    )
    trustworthy = streams.samples()
    train_steps = trustworthy[~held_out[trustworthy]]
    val_steps = trustworthy[held_out[trustworthy]]
    walking = val_steps[~streams.falling[val_steps]]
    rollout_steps = streams.samples(future=base.horizon)
    rollout_steps = rollout_steps[~held_out[rollout_steps]]
    gait_steps = gait.samples()

    report: dict = {
        "data": {
            "training_captures": [_display(p) for p in paths],
            "gait_capture": _display(args.gait),
            "train_steps": len(train_steps),
            "validation_steps": len(val_steps),
            "validation_falling_steps": len(val_steps) - len(walking),
            "validation_envs": args.val_envs,
            "teacher_sha256": str(archive["teacher_sha256"]),
            "dt_star": dt_star,
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

    variants = {"memory": base, "memory_limbs": replace(base, limbs=True)}
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    probe_fit = train_steps[torch.randperm(len(train_steps), generator=generator)[:PROBE_FIT_STEPS]]

    for name, config in variants.items():
        started = time.time()
        skill = new_skill(streams, train_steps, skill_norm, config, dt_star)
        decoder = new_decoder(streams, train_steps, layout, config, joint_names)
        history = train_skill(streams, rollout_steps, layout, skill, decoder, sigma, config)
        seconds = time.time() - started
        model = MemoryModel(encoder, skill, decoder, layout, sigma).eval()

        mean_memory, mean_action = (
            values.mean(dim=0) for values in act(model, streams, estimate, probe_fit)
        )
        metrics = {
            "validation": evaluate_memory(
                model, streams, val_steps, estimate, mean_memory, mean_action
            ),
            "validation_walking": evaluate_memory(
                model, streams, walking, estimate, mean_memory, mean_action
            ),
            "gait": evaluate_memory(
                model, gait, gait_steps, gait_estimate, mean_memory, mean_action
            ),
        }
        leakage = leakage_probes(model, streams, probe_fit, walking, estimate, BODY_JOINT_POSITIONS)

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
            model.policy(units.velocity, units.time, dt_star, memory=mean_memory.cpu())
        ).save(str(out / "policy_frozen_memory.pt"))
        frozen = FrozenActionPolicy(model.decoder, layout, mean_action.cpu()).eval()
        torch.jit.trace(frozen, example[0]).save(str(out / "policy_bypass.pt"))
        model.to(args.device)

        report["variants"][name] = {
            "config": config_dict(config),
            "train_seconds": round(seconds, 1),
            "final_losses": {k: v for k, v in history[-1].items() if k != "epoch"},
            "metrics": metrics,
            "leakage_validation_walking": leakage,
            "export_max_error": export_error,
            "policy_export": _display(out / "policy.pt"),
            "bypass_export": _display(out / "policy_bypass.pt"),
            "frozen_memory_export": _display(out / "policy_frozen_memory.pt"),
        }
        val, walk = metrics["validation_walking"], metrics["gait"]
        print(f"[{name}] {seconds:.0f} s")
        print(
            f"  imitation R2      walking {val['imitation_r2']:.3f}  gait "
            f"{walk['imitation_r2']:.3f}  (true state {val['imitation_r2_true_state']:.3f})"
        )
        print(f"  bypass u / phi    {val['bypass_action_r2']:.3f} / {val['bypass_memory_r2']:.3f}")
        print(
            f"  anchor R2         median {val['anchor_r2_median']:.3f}  "
            f"min {val['anchor_r2_min']:.3f}"
        )
        print(f"  joint probes      policy input {leakage['policy_input']['mlp_r2_median']:.3f}"
              f"  physical history {leakage['physical_history']['mlp_r2_median']:.3f}"
              f"  (excess {leakage['excess_mlp']:+.3f})")  # fmt: skip

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {_display(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
