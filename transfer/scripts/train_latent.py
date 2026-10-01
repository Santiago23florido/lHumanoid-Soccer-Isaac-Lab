"""Factor the teacher through the shared latent space, with and without the anchor.

Trains three models on the same rollouts with the same seed and capacity:

``anchored``
    The contribution: ``ẑ*`` is pinned to the analytic skill state and the
    latent dynamics predict the physical next state.
``anchored_open``
    The same with a weak bottleneck on the free part ``ζ``, to show what it
    lets through: the body's own configuration.
``free``
    The ablation: identical architecture, no anchor. The usual learned latent.

Both are evaluated on held-out environments of the training capture and on the
clean teacher gait, and exported as TorchScript policies (observation in,
joint targets out) that ``g1/scripts/eval_latent.py`` runs closed-loop.

Results go to ``transfer/results/latent_g1.json``; weights and exports to
``logs/latent/<variant>/`` (untracked, like every training output).

Usage::

    python transfer/scripts/train_latent.py
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

from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.dataset import load_transitions  # noqa: E402
from humanoid_transfer.transfer.latent.losses import LossWeights  # noqa: E402
from humanoid_transfer.transfer.latent.model import G1_OBSERVATION  # noqa: E402
from humanoid_transfer.transfer.latent.train import (  # noqa: E402
    TrainConfig,
    config_dict,
    evaluate,
    train,
)

BODY_JOINT_POSITIONS = slice(12, 35)
"""The 23 body joints' positions in the teacher's observation (hands excluded)."""


def _display(path: Path) -> str:
    """A repository-relative path where possible, for reports that get committed."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "g1/rollouts/teacher_dart.npz")
    parser.add_argument("--gait", type=Path, default=REPO_ROOT / "g1/rollouts/teacher.npz")
    parser.add_argument("--val-envs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=Path, default=REPO_ROOT / "logs/latent")
    parser.add_argument(
        "--results", type=Path, default=REPO_ROOT / "transfer/results/latent_g1.json"
    )
    args = parser.parse_args()

    data = load_transitions(args.data)
    gait = load_transitions(args.gait)
    envs = int(data.env.max()) + 1
    held_out = data.env >= envs - args.val_envs
    train_set, val_set = data.subset(~held_out), data.subset(held_out)

    archive = np.load(args.data)
    units = UnitSystem(float(archive["leg_length"]), float(archive["total_mass"]))

    report: dict = {
        "data": {
            "training_capture": _display(args.data),
            "gait_capture": _display(args.gait),
            "train_transitions": len(train_set),
            "validation_transitions": len(val_set),
            "validation_envs": args.val_envs,
            "validation_falling_transitions": int(val_set.falling.sum()),
            "gait_transitions": len(gait),
            "teacher_sha256": str(archive["teacher_sha256"]),
        },
        "variants": {},
    }

    base = TrainConfig(epochs=args.epochs, seed=args.seed)
    variants = {
        "anchored": base,
        "anchored_open": replace(base, weights=LossWeights(kl=1e-3)),
        "free": replace(base, weights=LossWeights(anchored=False, anchor=0.0)),
    }
    for name, config in variants.items():
        started = time.time()
        model, history = train(train_set, G1_OBSERVATION, config, device=args.device)
        seconds = time.time() - started
        metrics = {
            split: evaluate(model, subset, train_set, BODY_JOINT_POSITIONS, config.weights)
            for split, subset in (
                ("validation", val_set),
                ("validation_walking", val_set.subset(~val_set.falling)),
                ("gait", gait),
            )
        }

        out = args.logdir / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "config": config_dict(config)}, out / "model.pt")
        policy = model.policy(units.velocity, units.time).to("cpu").eval()
        example = gait.obs[:8]
        traced = torch.jit.trace(policy, example)
        assert torch.allclose(traced(example), policy(example), atol=1e-5)
        traced.save(str(out / "policy.pt"))

        report["variants"][name] = {
            "config": config_dict(config),
            "train_seconds": round(seconds, 1),
            "final_losses": {k: v for k, v in history[-1].items() if k != "epoch"},
            "metrics": metrics,
            "policy_export": _display(out / "policy.pt"),
        }

        val, walk = metrics["validation"], metrics["gait"]
        print(f"[{name}] {seconds:.0f} s")
        print(f"  imitation R2   val {val['imitation_r2']:.3f}  gait {walk['imitation_r2']:.3f}")
        print(f"  bypass R2      val {val['bypass_r2']:.3f}  gait {walk['bypass_r2']:.3f}")
        if val["anchor_r2_median"] is not None:
            print(
                f"  anchor R2      median {val['anchor_r2_median']:.3f}  "
                f"min {val['anchor_r2_min']:.3f}"
            )
        print(
            f"  probes         skill state {val['probe_skill_state_r2_median']:.3f}  "
            f"body joints {val['probe_body_joints_r2_median']:.3f}"
        )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
