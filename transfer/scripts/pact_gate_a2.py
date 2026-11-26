"""Gate G-A2: the state encoder with the analytic anchors, against one MLP per body.

Registered in ``docs/report/research_notes/icml/preregistration.md`` before it ran.
Three arms, each the same optimiser steps, each seeing the same samples of each body
(the shared arm draws ``--batch`` split over the training bodies; a per-body arm draws
that share from its own body):

- ``mlp``: one MLP per body, all 27 fields of ``z*`` learned (G-A's control);
- ``mlp_k``: one MLP per body reading the analytic anchors too; the 21 exact fields are
  analytic, the CoM velocity, the loads and a height correction are learned;
- ``pact_k``: one shared PACT encoder whose entity queries read the anchors, same split.

The exact fields come from ``pact/anchors.py`` at each window's last step. A predicted
``z*`` is assembled from them: foot velocity = analytic relative velocity + predicted
CoM velocity; height = analytic height above the lowest sole + predicted correction.
Metric and bootstrap as G-A (held-out RMSE over all 27 fields, pooled units).

Usage::

    python transfer/scripts/pact_gate_a2.py --bodies g1,nao,h1,cassie
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
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import torch  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.pact.anchors import KinematicAnchors  # noqa: E402
from humanoid_transfer.transfer.latent.pact.data import (  # noqa: E402
    load_body,
    split_by_environment,
)
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import (  # noqa: E402
    EmbodimentEncoder,
    PactDims,
    StateEncoder,
    stack_bodies,
)
from humanoid_transfer.transfer.latent.pact.tokens import split  # noqa: E402
from pact_check_anchors import sole_override  # noqa: E402
from pact_gate_a import bootstrap_rmse, field_rmse, per_body_mlp  # noqa: E402
from torch import nn  # noqa: E402

S = {f.name: SKILL_LAYOUT.slice_of(f.name) for f in SKILL_LAYOUT.fields}
LEARNED = 6
"""Learned outputs: CoM velocity (3), height correction (1), loads (2)."""


def analytic_z(anchors: KinematicAnchors, w, time_unit: float, chunk: int = 8192) -> np.ndarray:
    """``(N, 27)``: the exact fields from the windows' last step; CoM velocity and loads 0,
    foot velocity relative to the CoM, height above the lowest sole."""
    out = np.zeros((len(w.z), SKILL_LAYOUT.size), dtype=np.float32)
    for start in range(0, len(w.z), chunk):
        rows = slice(start, start + chunk)
        q = torch.as_tensor(w.joint_obs[rows, :, -1, 0])
        qd = torch.as_tensor(w.joint_obs[rows, :, -1, 1] / time_unit)
        g = torch.as_tensor(w.imu[rows, -1, :3])
        omega = torch.as_tensor(w.imu[rows, -1, 3:] / time_unit)
        with torch.no_grad():
            a = anchors(q, qd, g, omega)
        z = out[rows]
        z[:, S["height"]] = a["height"].numpy()
        z[:, S["gravity"]] = a["gravity"].numpy()
        z[:, S["base_angular_velocity"]] = a["base_angular_velocity"].numpy()
        z[:, S["foot_position"]] = a["foot_position"].numpy().reshape(-1, 6)
        z[:, S["foot_velocity"]] = a["foot_velocity_relative"].numpy().reshape(-1, 6)
        z[:, S["angular_momentum"]] = a["angular_momentum"].numpy()
    return out


def targets_of(z_true: np.ndarray, z_analytic: np.ndarray) -> np.ndarray:
    """``(N, 6)``: CoM velocity, height correction, loads."""
    return np.concatenate(
        [
            z_true[:, S["com_velocity"]],
            z_true[:, S["height"]] - z_analytic[:, S["height"]],
            z_true[:, S["foot_load"]],
        ],
        axis=1,
    )


def assemble(z_analytic: torch.Tensor, learned: torch.Tensor) -> torch.Tensor:
    """The predicted ``z*`` from the exact fields and the 6 learned numbers."""
    z = z_analytic.clone()
    com_vel = learned[:, 0:3]
    z[:, S["com_velocity"]] = com_vel
    z[:, S["height"]] = z_analytic[:, S["height"]] + learned[:, 3:4]
    z[:, S["foot_load"]] = learned[:, 4:6]
    z[:, S["foot_velocity"]] = z_analytic[:, S["foot_velocity"]] + com_vel.repeat(1, 2)
    return z


class SharedAnchored(nn.Module):
    def __init__(self, dims: PactDims) -> None:
        super().__init__()
        self.embodiment = EmbodimentEncoder(dims)
        self.state = StateEncoder(dims)

    def forward(self, bodies, joint_obs, imu, anchor_in):
        embodiment, descriptor = self.embodiment(bodies)
        centroid, contacts = split(anchor_in)
        out = self.state(bodies, descriptor, embodiment, joint_obs, imu, (centroid, contacts))
        # centroid head: [height, com_velocity(3), ...]; contact head: [..., load]
        return torch.cat(
            [out["centroid"][:, 1:4], out["centroid"][:, 0:1], out["contacts"][:, :, 6]], dim=1
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default=None)
    parser.add_argument("--captures", default="logs/pact/rollouts/{body}/flat_*.npz")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--arms", default="mlp,mlp_k,pact_k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_gate_a2.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    arms = args.arms.split(",")
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    names = [b for b in args.bodies.split(",") if b]
    train_index = [i for i, b in enumerate(names) if b != args.hold_out]

    descriptors, contacts, windows, analytic = [], [], [], []
    for body in names:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        unit = entry.get("leg_length_unit")
        descriptor = describe(card, leg_length=unit)
        paths = sorted(REPO.glob(args.captures.format(body=body)))
        w = load_body(body, paths, args.history, descriptor.leg_length, descriptor.joint_names)
        anchors = KinematicAnchors(card, unit, sole_override(body))
        analytic.append(analytic_z(anchors, w, float(np.sqrt(descriptor.leg_length / GRAVITY))))
        descriptors.append(descriptor)
        contacts.append(contact_joints(card))
        windows.append(w)
        print(f"{body}: {len(w.z)} windows")

    test = [split_by_environment(w.group) for w in windows]
    pooled = np.concatenate([windows[i].z[~test[i]] for i in train_index])
    z_mean, z_std = pooled.mean(0), np.maximum(pooled.std(0), 1e-3)
    a_pool = np.concatenate([analytic[i][~test[i]] for i in train_index])
    a_mean, a_std = a_pool.mean(0), np.maximum(a_pool.std(0), 1e-3)
    t_pool = np.concatenate(
        [targets_of(windows[i].z[~test[i]], analytic[i][~test[i]]) for i in train_index]
    )
    t_mean, t_std = t_pool.mean(0), np.maximum(t_pool.std(0), 1e-3)

    table = stack_bodies(descriptors, contacts).to(device)
    count = table.mask.shape[1]
    as_t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)  # noqa: E731

    def batch(i: int, rows: np.ndarray):
        w = windows[i]
        obs = np.zeros((len(rows), count, args.history, 2), dtype=np.float32)
        obs[:, : w.joint_obs.shape[1]] = w.joint_obs[rows]
        a = analytic[i][rows]
        return (
            as_t(obs),
            as_t(w.imu[rows]),
            as_t((a - a_mean) / a_std),  # anchors as inputs, normalised
            as_t(a),  # anchors in z* units, to assemble the prediction
            as_t((w.z[rows] - z_mean) / z_std),  # full target, for mlp
            as_t((targets_of(w.z[rows], a) - t_mean) / t_std),  # learned target
        )

    rng = np.random.default_rng(args.seed)
    train_rows = {i: np.nonzero(~test[i])[0] for i in train_index}
    per = max(1, args.batch // len(train_index))
    t_mean_t, t_std_t, z_mean_t, z_std_t = as_t(t_mean), as_t(t_std), as_t(z_mean), as_t(z_std)
    predictions: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in arms}
    params: dict[str, int] = {}

    def train(model: nn.Module, step_fn, label: str) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        start = time.time()
        for step in range(args.steps):
            loss = step_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 2000 == 0 or step == args.steps - 1:
                print(
                    f"{label} step {step:5d} loss {loss.item():.5f} ({time.time() - start:.0f} s)"
                )

    def chunks(rows):
        return np.array_split(rows, max(1, len(rows) // 2048))

    if "pact_k" in arms:
        dims = PactDims(observation=2, history=args.history, width=args.width, anchor_inputs=True)
        shared = SharedAnchored(dims).to(device)

        def step_shared():
            parts = [batch(i, rng.choice(train_rows[i], per)) for i in train_index]
            which = torch.cat(
                [torch.full((per,), i, device=device, dtype=torch.long) for i in train_index]
            )
            obs, imu, a_in, _, _, target = (torch.cat([p[k] for p in parts]) for k in range(6))
            return (shared(table.take(which), obs, imu, a_in) - target).square().mean()

        train(shared, step_shared, "pact_k")
        shared.eval()
        params["pact_k"] = sum(p.numel() for p in shared.parameters())
        for i in range(len(names)):
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(test[i])[0]):
                    obs, imu, a_in, a_z, _, _ = batch(i, rows)
                    which = torch.full((len(rows),), i, device=device, dtype=torch.long)
                    learned = shared(table.take(which), obs, imu, a_in) * t_std_t + t_mean_t
                    out.append(((assemble(a_z, learned) - z_mean_t) / z_std_t).cpu().numpy())
            predictions["pact_k"][i] = np.concatenate(out)

    for arm in [a for a in arms if a in ("mlp", "mlp_k")]:
        for i in train_index:
            w = windows[i]
            n_in = w.joint_obs.shape[1] * args.history * 2 + args.history * 6
            if arm == "mlp_k":
                n_in += SKILL_LAYOUT.size
            model = per_body_mlp(n_in).to(device)
            if arm == "mlp_k":
                model[-1] = nn.Linear(512, LEARNED).to(device)
            n_j = w.joint_obs.shape[1]

            def features(obs, imu, a_in, arm=arm, n_j=n_j):
                parts = [obs[:, :n_j].flatten(1), imu.flatten(1)]
                if arm == "mlp_k":
                    parts.append(a_in)
                return torch.cat(parts, dim=1)

            def step_mlp(i=i, model=model, arm=arm, features=features):
                obs, imu, a_in, _, full, learned = batch(i, rng.choice(train_rows[i], per))
                target = learned if arm == "mlp_k" else full
                return (model(features(obs, imu, a_in)) - target).square().mean()

            train(model, step_mlp, f"{arm}/{names[i]}")
            model.eval()
            params[f"{arm}/{names[i]}"] = sum(p.numel() for p in model.parameters())
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(test[i])[0]):
                    obs, imu, a_in, a_z, _, _ = batch(i, rows)
                    y = model(features(obs, imu, a_in))
                    if arm == "mlp_k":
                        y = (assemble(a_z, y * t_std_t + t_mean_t) - z_mean_t) / z_std_t
                    out.append(y.cpu().numpy())
            predictions[arm][i] = np.concatenate(out)

    report = {
        "gate": "G-A2",
        "config": {k: str(v) for k, v in vars(args).items()},
        "parameters": params,
        "bodies": {},
    }
    verdict = True
    for i, body in enumerate(names):
        rows = np.nonzero(test[i])[0]
        truth = (windows[i].z[rows] - z_mean) / z_std
        groups = windows[i].group[rows]
        entry = {"held_out": body == args.hold_out}
        errors = {arm: predictions[arm][i] - truth for arm in arms if i in predictions[arm]}
        for arm, err in errors.items():
            entry[arm] = field_rmse(err)
        if "pact_k" in errors and "mlp" in errors:
            _, _, diff, se = bootstrap_rmse(
                (errors["pact_k"] ** 2).mean(1),
                (errors["mlp"] ** 2).mean(1),
                groups,
                args.bootstrap,
                args.seed,
            )
            entry.update(pact_k_minus_mlp=diff, pact_k_minus_mlp_se=se, passes=bool(diff <= 2 * se))
            verdict &= entry["passes"]
        if "pact_k" in errors and "mlp_k" in errors:
            _, _, diff, se = bootstrap_rmse(
                (errors["pact_k"] ** 2).mean(1),
                (errors["mlp_k"] ** 2).mean(1),
                groups,
                args.bootstrap,
                args.seed,
            )
            entry.update(pact_k_minus_mlp_k=diff, pact_k_minus_mlp_k_se=se)
        report["bodies"][body] = entry
        print(
            body,
            {arm: round(entry[arm]["all"], 4) for arm in errors},
            {k: round(v, 4) for k, v in entry.items() if "minus" in k},
        )
    report["passes"] = bool(verdict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"G-A2 {'PASSES' if verdict else 'FAILS'}; written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
