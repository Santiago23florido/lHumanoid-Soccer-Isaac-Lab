"""Gate G-A: one shared PACT state encoder against one MLP per body.

Both read the same inputs (joint angles and speeds, projected gravity and base
angular velocity, over ``--history`` steps, speeds in each body's own time unit)
and regress the true ``z*``, each column in units of its standard deviation pooled
over the training bodies. Training uses environments 0-23 of every capture, the
test environments 24-31. The gate (``docs/report/research_notes/icml/
preregistration.md``) passes when, on every training body, PACT's held-out RMSE is
at most the per-body MLP's plus two bootstrap standard errors of the paired
difference. With ``--hold-out <body>``, that body is left out of training and its
error is reported against a constant (training-mean) prediction.

Usage::

    python transfer/scripts/pact_gate_a.py --bodies g1,nao,h1,cassie
    python transfer/scripts/pact_gate_a.py --bodies g1,h1,cassie,nao --hold-out nao
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

import torch  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.pact.data import (  # noqa: E402
    load_body,
    split_by_environment,
)
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import (  # noqa: E402
    EmbodimentEncoder,
    PactDims,
    StateEncoder,
    stack_bodies,
)
from humanoid_transfer.transfer.latent.pact.tokens import merge  # noqa: E402
from torch import nn  # noqa: E402


class SharedEncoder(nn.Module):
    """The two encoders of PACT that map a body's observations to the anchored z*."""

    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        self.embodiment = EmbodimentEncoder(dims)
        self.state = StateEncoder(dims)

    def forward(self, bodies, joint_obs, imu):
        embodiment, descriptor = self.embodiment(bodies)
        out = self.state(bodies, descriptor, embodiment, joint_obs, imu)
        return merge(out["centroid"], out["contacts"])


def per_body_mlp(inputs: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(inputs, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(),
        nn.Linear(512, SKILL_LAYOUT.size),
    )  # fmt: skip


def field_rmse(errors: np.ndarray) -> dict[str, float]:
    """Per-field RMSE of normalised errors ``(N, 27)``, plus ``all``."""
    out = {"all": float(np.sqrt(np.mean(errors**2)))}
    for field in SKILL_LAYOUT.fields:
        block = errors[:, SKILL_LAYOUT.slice_of(field.name)]
        out[field.name] = float(np.sqrt(np.mean(block**2)))
    return out


def bootstrap_rmse(
    sq_a: np.ndarray, sq_b: np.ndarray | None, groups: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float | None, float | None]:
    """RMSE of ``a`` and its SE over resampled groups; same for ``a - b`` if given."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    sum_a = np.bincount(inverse, weights=sq_a)
    count = np.bincount(inverse).astype(float)
    sum_b = np.bincount(inverse, weights=sq_b) if sq_b is not None else None
    rmse_a = np.sqrt(sum_a.sum() / count.sum())
    stats_a, stats_d = [], []
    for _ in range(draws):
        pick = rng.integers(0, len(unique), len(unique))
        n = count[pick].sum()
        a = np.sqrt(sum_a[pick].sum() / n)
        stats_a.append(a)
        if sum_b is not None:
            stats_d.append(a - np.sqrt(sum_b[pick].sum() / n))
    if sum_b is None:
        return float(rmse_a), float(np.std(stats_a)), None, None
    diff = rmse_a - np.sqrt(sum_b.sum() / count.sum())
    return float(rmse_a), float(np.std(stats_a)), float(diff), float(np.std(stats_d))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default=None)
    parser.add_argument("--captures", default="logs/pact/rollouts/{body}/flat_*.npz")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument(
        "--mlp-steps",
        type=int,
        default=None,
        help="Steps of each per-body MLP (default --steps). --steps 24000 --mlp-steps 6000 gives "
        "both the same samples per body with four bodies.",
    )
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_gate_a.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    names = [b for b in args.bodies.split(",") if b]
    train_names = [b for b in names if b != args.hold_out]

    descriptors, contacts, windows = [], [], []
    for body in names:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        descriptor = describe(card, leg_length=entry.get("leg_length_unit"))
        paths = sorted(REPO.glob(args.captures.format(body=body)))
        if not paths:
            raise SystemExit(f"no captures for {body}: {args.captures.format(body=body)}")
        descriptors.append(descriptor)
        contacts.append(contact_joints(card))
        windows.append(
            load_body(body, paths, args.history, descriptor.leg_length, descriptor.joint_names)
        )
        print(f"{body}: {len(paths)} captures, {len(windows[-1].z)} windows")

    table = stack_bodies(descriptors, contacts).to(device)
    count = table.mask.shape[1]
    test = [split_by_environment(w.group) for w in windows]
    train_index = [names.index(b) for b in train_names]
    pooled = np.concatenate([windows[i].z[~test[i]] for i in train_index])
    mean, std = pooled.mean(0), np.maximum(pooled.std(0), 1e-3)

    def tensors(i: int, rows: np.ndarray):
        w = windows[i]
        obs = np.zeros((len(rows), count, args.history, 2), dtype=np.float32)
        obs[:, : w.joint_obs.shape[1]] = w.joint_obs[rows]
        return (
            torch.as_tensor(obs, device=device),
            torch.as_tensor(w.imu[rows], device=device),
            torch.as_tensor((w.z[rows] - mean) / std, device=device),
        )

    rng = np.random.default_rng(args.seed)
    train_rows = {i: np.nonzero(~test[i])[0] for i in train_index}

    # The shared encoder: every batch draws its samples evenly from the training bodies.
    dims = PactDims(observation=2, history=args.history, width=args.width)
    shared = SharedEncoder(dims).to(device)
    optimiser = torch.optim.AdamW(shared.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.steps)
    start = time.time()
    per = max(1, args.batch // len(train_index))
    for step in range(args.steps):
        parts = [tensors(i, rng.choice(train_rows[i], per)) for i in train_index]
        which = torch.cat(
            [torch.full((per,), i, device=device, dtype=torch.long) for i in train_index]
        )
        obs, imu, target = (torch.cat([p[k] for p in parts]) for k in range(3))
        loss = (shared(table.take(which), obs, imu) - target).square().mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(shared.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"shared step {step:5d} loss {loss.item():.4f} ({time.time() - start:.0f} s)")
    parameters = sum(p.numel() for p in shared.parameters())

    def predict_shared(i: int, rows: np.ndarray) -> np.ndarray:
        shared.eval()
        out = []
        with torch.no_grad():
            for chunk in np.array_split(rows, max(1, len(rows) // 2048)):
                obs, imu, _ = tensors(i, chunk)
                which = torch.full((len(chunk),), i, device=device, dtype=torch.long)
                out.append(shared(table.take(which), obs, imu).cpu().numpy())
        return np.concatenate(out)

    report: dict = {
        "gate": "G-A",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "shared_parameters": parameters,
        "bodies": {},
    }
    verdict = True
    for i, body in enumerate(names):
        rows = np.nonzero(test[i])[0]
        truth = (windows[i].z[rows] - mean) / std
        err_shared = predict_shared(i, rows) - truth
        groups = windows[i].group[rows]
        entry: dict = {"test_windows": int(len(rows)), "shared": field_rmse(err_shared)}
        if body == args.hold_out:
            err_const = -truth
            rmse, se, diff, se_diff = bootstrap_rmse(
                (err_shared**2).mean(1), (err_const**2).mean(1), groups, args.bootstrap, args.seed
            )
            entry.update(
                held_out=True,
                constant=field_rmse(err_const),
                shared_rmse=rmse,
                shared_se=se,
                shared_minus_constant=diff,
                shared_minus_constant_se=se_diff,
            )
        else:
            w = windows[i]
            inputs = w.joint_obs.shape[1] * args.history * 2 + args.history * 6
            mlp = per_body_mlp(inputs).to(device)
            opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            mlp_steps = args.mlp_steps or args.steps
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, mlp_steps)
            n_joints = w.joint_obs.shape[1]

            def flat(obs, imu, n=n_joints):
                return torch.cat([obs[:, :n].flatten(1), imu.flatten(1)], dim=1)

            for _ in range(mlp_steps):
                obs, imu, target = tensors(i, rng.choice(train_rows[i], args.batch))
                loss = (mlp(flat(obs, imu)) - target).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            mlp.eval()
            with torch.no_grad():
                obs, imu, _ = tensors(i, rows)
                err_mlp = mlp(flat(obs, imu)).cpu().numpy() - truth
            rmse, se, diff, se_diff = bootstrap_rmse(
                (err_shared**2).mean(1), (err_mlp**2).mean(1), groups, args.bootstrap, args.seed
            )
            passes = diff <= 2 * se_diff
            verdict &= passes
            entry.update(
                held_out=False,
                per_body_mlp=field_rmse(err_mlp),
                per_body_mlp_parameters=sum(p.numel() for p in mlp.parameters()),
                shared_rmse=rmse,
                shared_se=se,
                shared_minus_mlp=diff,
                shared_minus_mlp_se=se_diff,
                passes=bool(passes),
            )
        report["bodies"][body] = entry
        print(
            f"{body:7s} shared {entry['shared']['all']:.4f}"
            + (
                f"  constant {entry['constant']['all']:.4f}"
                if entry["held_out"]
                else f"  mlp {entry['per_body_mlp']['all']:.4f}"
                f"  diff {entry['shared_minus_mlp']:+.4f} +- {entry['shared_minus_mlp_se']:.4f}"
                f"  {'PASS' if entry['passes'] else 'FAIL'}"
            )
        )
    report["passes"] = bool(verdict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"G-A {'PASSES' if verdict else 'FAILS'}; written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
