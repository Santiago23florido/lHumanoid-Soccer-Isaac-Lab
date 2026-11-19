"""Gate G-D0: one decoder for every body -- the four trackers cloned offline into PACT.

Registered in ``docs/report/research_notes/icml/preregistration.md`` before it ran. On the
co-execution map captures (``pact_coexecute.sh``: each body's tracker on the 71 gait
probes), one :class:`~humanoid_transfer.transfer.latent.pact.policy.PactPolicy` -- G-A3's
``pact_kd`` encoder and the joint decoder -- learns every body's joint targets from what
the body senses and the plan its tracker was shown:

- per joint and step (history 4): ``q - q0`` (rad), ``q̇ sqrt(L/g)``, and the target in
  force, ``q_target - q0`` (rad; the tracker's last action, in radians);
- the IMU, and G-A3's ``pact_kd`` anchor inputs (kinematic anchors, odometry, drive loads);
- the plan: the tracker's view decoded into raw ``z*`` with its generator's normaliser
  (``plan_from_view``; the decoded estimate is checked against the capture's), each step
  as its value and its difference to the estimate, with its path and its time ahead in
  units of ``sqrt(L/g)``.

The label is the target the tracker set at that step, ``q_target(t + 1) - q0``, on the
joints whose target moves in the capture. Arms:

- ``pact_d``: one model for every body, BC loss (MSE over one pooled scale) plus G-A3's
  anchor loss on the encoder's six learned fields;
- ``mlp_d``: one MLP per body on the same inputs, flattened (the in-body reference);
- ``hold``: the target in force kept (no learning; what persistence alone explains).

Test set: probes with index % 4 == 0 (as G-A3). Metric per body: RMSE (rad) over the
controlled joints and ``R² = 1 - MSE / Var(label)``; bootstrap over test environments.
The trained ``pact_d`` and its normalisers are saved for G-D1 (``logs/pact/models/``).

Usage::

    python transfer/scripts/pact_gate_d.py
    python transfer/scripts/pact_gate_d.py --hold-out nao --output <json>
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
from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.data import _window_ends, load_body  # noqa: E402
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import (  # noqa: E402
    SKILL,
    PactPolicy,
    plan_from_view,
)
from pact_check_anchors import sole_override  # noqa: E402
from pact_gate_a import bootstrap_rmse  # noqa: E402
from pact_gate_a2 import analytic_z  # noqa: E402
from pact_gate_a3 import (  # noqa: E402
    DRIVES,
    PROBES,
    physics_bases,
    residual_targets,
    with_physics,
)
from torch import nn  # noqa: E402

CONTRACTS = "g1=walk_dagger3e,nao=walk_dagger2e,h1=walk_dagger3e,cassie=walk_dagger3e"
"""The contract each body's co-executions ran under (``pact_coexecute.sh``)."""


class BodyData:
    """One body's samples: inputs, labels and its constants."""

    def __init__(self, body, args, registry):
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        unit = entry.get("leg_length_unit")
        self.descriptor = describe(card, leg_length=unit)
        self.contacts = contact_joints(card)
        path = REPO / args.captures.format(body=body)
        h = args.history
        w = load_body(
            body,
            [path],
            h,
            self.descriptor.leg_length,
            self.descriptor.joint_names,
            extra=("joint_pos_target",),
            constants=DRIVES + ("default_joint_pos",),
        )
        anchors = KinematicAnchors(card, unit, sole_override(body))
        drive = DriveLoads(card, anchors)
        drive.set_drives(*(w.constants[k] for k in DRIVES))
        self.time_unit = float(np.sqrt(self.descriptor.leg_length / GRAVITY))
        analytic = analytic_z(anchors, w, self.time_unit)
        physics = physics_bases(drive, w, analytic, self.time_unit)

        data = np.load(path)
        done = data["done"]
        steps = done.shape[0]
        t, env = _window_ends(done, h)
        # Also clean at t - h (the target in force at the window's first step was set inside
        # the episode) and at t (the label, set at t, is read at t + 1).
        keep = (t - h >= 0) & (t + 1 < steps) & ~done[t, env]
        keep &= ~done[np.maximum(t - h, 0), env]
        keep = np.nonzero(keep)[0][:: args.stride]
        t, env = t[keep], env[keep]

        q0 = data["default_joint_pos"].astype(np.float32)
        q = data["joint_pos"]
        target = data["joint_pos_target"]
        rows = t[:, None] + np.arange(-h + 1, 1)[None]
        cols = env[:, None]
        self.joint_obs = (
            np.stack(
                [
                    q[rows, cols] - q0,
                    data["joint_vel"][rows, cols] * self.time_unit,
                    target[rows, cols] - q0,
                ],
                axis=-1,
            )
            .transpose(0, 2, 1, 3)
            .astype(np.float32)
        )  # (N, J, h, 3)
        self.imu = w.imu[keep]
        self.label = (target[t + 1, env] - q0).astype(np.float32)
        self.hold = (target[t, env] - q0).astype(np.float32)
        self.controlled = target.reshape(-1, target.shape[-1]).std(0) > 1e-6
        self.anchor_in = with_physics(analytic[keep], physics[keep])
        self.learned = residual_targets(w.z[keep], analytic[keep], physics[keep])
        self.z = w.z[keep]

        contract = dict(c.split("=") for c in args.contracts.split(","))[body]
        norm = torch.load(
            REPO / f"logs/contract/{contract}/generator.pt", map_location="cpu", weights_only=False
        )["generator"]
        mean, std = norm["skill_norm.mean"].numpy(), norm["skill_norm.std"].numpy()
        obs = data["policy_obs"]
        estimate = data["estimate"]
        self.view = self._view(obs, estimate, mean, std)
        tail = obs[t, env, obs.shape[-1] - (self.view * (SKILL + 3) + SKILL) :]
        upcoming, est, self.path = plan_from_view(tail, self.view, mean, std)
        error = float(np.abs(est - estimate[t, env]).max())
        if error > 1e-3:
            raise ValueError(f"{body}: decoded estimate differs from the capture's by {error}")
        self.plan = upcoming.astype(np.float32)
        self.relative = (upcoming - est[:, None]).astype(np.float32)
        self.path = self.path.astype(np.float32)
        dt_star = float(data["control_dt"]) / self.time_unit
        self.times = (np.arange(1, self.view + 1) * dt_star).astype(np.float32)
        self.group = env.astype(np.int64)
        probe = data["probe_index"][env]
        if not np.array_equal(probe, env % PROBES):
            raise ValueError(f"{body}: environments are not laid out probe by probe")
        self.test = probe % 4 == 0
        self.name = body

    @staticmethod
    def _view(obs, estimate, mean, std) -> int:
        """The view length whose decoded estimate matches the capture's (rows 20-200)."""
        for view in range(1, 20):
            width = view * (SKILL + 3) + SKILL
            if width > obs.shape[-1]:
                break
            start = obs.shape[-1] - width + view * SKILL
            zhat = obs[20:200, :, start : start + SKILL] * std + mean
            if np.abs(zhat - estimate[20:200]).max() < 1e-3:
                return view
        raise ValueError("no plan view decodes to the capture's estimate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default="")
    parser.add_argument("--captures", default="logs/pact/coexec/{body}_map.npz")
    parser.add_argument("--contracts", default=CONTRACTS)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th window")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--arms", default="hold,pact_d,mlp_d")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_gate_d0.json")
    parser.add_argument("--save", type=Path, default=None, help="where to save pact_d")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    arms = args.arms.split(",")
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    names = [b for b in args.bodies.split(",") if b]
    bodies = []
    for body in names:
        start = time.time()
        bodies.append(BodyData(body, args, registry))
        b = bodies[-1]
        print(
            f"{body}: {len(b.label)} samples, view {b.view}, controlled "
            f"{int(b.controlled.sum())}/{len(b.controlled)} ({time.time() - start:.0f} s)",
            flush=True,
        )
    train_index = [i for i, b in enumerate(names) if b != args.hold_out]
    train_rows = {i: np.nonzero(~b.test)[0] for i, b in enumerate(bodies)}

    def pooled(key):
        return np.concatenate([getattr(bodies[i], key)[train_rows[i]] for i in train_index])

    z = pooled("z")
    z_mean, z_std = z.mean(0), np.maximum(z.std(0), 1e-3)
    a = pooled("anchor_in")
    a_mean, a_std = a.mean(0), np.maximum(a.std(0), 1e-3)
    t_all = pooled("learned")
    t_mean, t_std = t_all.mean(0), np.maximum(t_all.std(0), 1e-3)
    label_scale = float(
        np.sqrt(
            np.mean([np.mean(bodies[i].label[:, bodies[i].controlled] ** 2) for i in train_index])
        )
    )
    as_t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)  # noqa: E731
    table = stack_bodies([b.descriptor for b in bodies], [b.contacts for b in bodies]).to(device)
    count = table.mask.shape[1]
    steps_max = max(b.view for b in bodies)
    controlled = torch.zeros(len(bodies), count, dtype=torch.bool, device=device)
    for i, b in enumerate(bodies):
        controlled[i, : len(b.controlled)] = torch.as_tensor(b.controlled, device=device)

    def batch(i: int, rows: np.ndarray) -> dict[str, torch.Tensor]:
        b = bodies[i]
        n, j, v = len(rows), b.joint_obs.shape[1], b.view
        obs = np.zeros((n, count, args.history, 3), np.float32)
        obs[:, :j] = b.joint_obs[rows]
        plan = np.zeros((n, steps_max, SKILL), np.float32)
        plan[:, :v] = (b.plan[rows] - z_mean) / z_std
        rel = np.zeros((n, steps_max, SKILL), np.float32)
        rel[:, :v] = b.relative[rows] / z_std
        path = np.zeros((n, steps_max, 3), np.float32)
        path[:, :v] = b.path[rows]
        times = np.zeros((n, steps_max), np.float32)
        times[:, :v] = b.times
        mask = np.zeros((n, steps_max), bool)
        mask[:, :v] = True
        label = np.zeros((n, count), np.float32)
        label[:, :j] = b.label[rows]
        return {
            "joint_obs": as_t(obs),
            "imu": as_t(b.imu[rows]),
            "anchor_in": as_t((b.anchor_in[rows] - a_mean) / a_std),
            "plan": as_t(plan),
            "plan_relative": as_t(rel),
            "plan_time": as_t(times),
            "plan_path": as_t(path),
            "plan_mask": torch.as_tensor(mask, device=device),
            "label": as_t(label),
            "learned": as_t((b.learned[rows] - t_mean) / t_std),
        }

    def fit(model: nn.Module, step_fn, label: str) -> float:
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
        return time.time() - start

    def chunks(rows):
        return np.array_split(rows, max(1, len(rows) // 2048))

    predictions: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in arms}
    params: dict[str, int] = {}
    seconds: dict[str, float] = {}
    if "hold" in arms:
        for i, b in enumerate(bodies):
            predictions["hold"][i] = b.hold[b.test]

    if "pact_d" in arms:
        dims = PactDims(
            observation=3,
            history=args.history,
            width=args.width,
            anchor_inputs=True,
            timed_plan=True,
        )
        model = PactPolicy(dims).to(device)
        per = max(1, args.batch // len(train_index))

        def step_pact():
            parts = [batch(i, rng.choice(train_rows[i], per)) for i in train_index]
            x = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            which = torch.cat(
                [torch.full((per,), i, device=device, dtype=torch.long) for i in train_index]
            )
            label, learned = x.pop("label"), x.pop("learned")
            out_learned, out = model(table.take(which), **x)
            moved = controlled[which]
            bc = ((out - label) / label_scale).square()[moved].mean()
            anchor = (out_learned - learned).square().mean()
            return bc + args.anchor_weight * anchor

        seconds["pact_d"] = fit(model, step_pact, "pact_d")
        model.eval()
        params["pact_d"] = sum(p.numel() for p in model.parameters())
        for i, b in enumerate(bodies):
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(b.test)[0]):
                    x = batch(i, rows)
                    x.pop("label"), x.pop("learned")
                    which = torch.full((len(rows),), i, device=device, dtype=torch.long)
                    out.append(
                        model(table.take(which), **x)[1][:, : len(b.controlled)].cpu().numpy()
                    )
            predictions["pact_d"][i] = np.concatenate(out)
        save = args.save or REPO / (
            f"logs/pact/models/d0_{args.hold_out or 'all'}_s{args.seed}_p{args.steps}.pt"
        )
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": model.state_dict(),
                "dims": vars(dims),
                "bodies": names,
                "hold_out": args.hold_out,
                "z_mean": z_mean,
                "z_std": z_std,
                "a_mean": a_mean,
                "a_std": a_std,
                "t_mean": t_mean,
                "t_std": t_std,
                "label_scale": label_scale,
                "views": {b.name: b.view for b in bodies},
            },
            save,
        )
        print(f"saved {save}")

    if "mlp_d" in arms:
        for i, b in enumerate(bodies):
            j, v = b.joint_obs.shape[1], b.view
            inputs = j * args.history * 3 + args.history * 6 + SKILL + v * (2 * SKILL + 3)
            moved = np.nonzero(b.controlled)[0]
            model = nn.Sequential(
                nn.Linear(inputs, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(),
                nn.Linear(512, len(moved)),
            ).to(device)  # fmt: skip

            def features(rows, b=b):
                return torch.cat(
                    [
                        as_t(b.joint_obs[rows]).flatten(1),
                        as_t(b.imu[rows]).flatten(1),
                        as_t((b.anchor_in[rows] - a_mean) / a_std),
                        as_t((b.plan[rows] - z_mean) / z_std).flatten(1),
                        as_t(b.relative[rows] / z_std).flatten(1),
                        as_t(b.path[rows]).flatten(1),
                    ],
                    dim=1,
                )

            def step_mlp(i=i, b=b, model=model, features=features, moved=moved):
                rows = rng.choice(train_rows[i], max(1, args.batch // len(train_index)))
                target = as_t(b.label[rows][:, moved])
                return ((model(features(rows)) - target) / label_scale).square().mean()

            seconds[f"mlp_d/{b.name}"] = fit(model, step_mlp, f"mlp_d/{b.name}")
            model.eval()
            params[f"mlp_d/{b.name}"] = sum(p.numel() for p in model.parameters())
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(b.test)[0]):
                    full = np.zeros((len(rows), len(b.controlled)), np.float32)
                    full[:, moved] = model(features(rows)).cpu().numpy()
                    out.append(full)
            predictions["mlp_d"][i] = np.concatenate(out)

    report = {
        "gate": "G-D0",
        "config": {k: str(v) for k, v in vars(args).items()},
        "parameters": params,
        "train_seconds": {k: round(v, 1) for k, v in seconds.items()},
        "label_scale_rad": label_scale,
        "bodies": {},
    }
    for i, b in enumerate(bodies):
        moved = b.controlled
        truth = b.label[b.test][:, moved]
        groups = b.group[b.test]
        variance = float(truth.var(0).mean())
        entry: dict = {
            "held_out": b.name == args.hold_out,
            "view": b.view,
            "controlled_joints": int(moved.sum()),
            "test_samples": int(b.test.sum()),
            "label_variance_rad2": variance,
        }
        sq = {
            arm: ((p[i][:, moved] - truth) ** 2).mean(1) for arm, p in predictions.items() if i in p
        }
        for arm, err in sq.items():
            rmse, se, _, _ = bootstrap_rmse(err, None, groups, args.bootstrap, args.seed)
            entry[arm] = {"rmse_rad": rmse, "se": se, "r2": 1.0 - float(err.mean()) / variance}
        for a_arm, b_arm in (("pact_d", "mlp_d"), ("pact_d", "hold"), ("mlp_d", "hold")):
            if a_arm in sq and b_arm in sq:
                _, _, diff, se = bootstrap_rmse(
                    sq[a_arm], sq[b_arm], groups, args.bootstrap, args.seed
                )
                entry[f"{a_arm}_minus_{b_arm}"] = diff
                entry[f"{a_arm}_minus_{b_arm}_se"] = se
        report["bodies"][b.name] = entry
        print(
            b.name,
            "(held out)" if entry["held_out"] else "",
            {arm: (round(entry[arm]["rmse_rad"], 4), round(entry[arm]["r2"], 3)) for arm in sq},
            flush=True,
        )
    if "pact_d" in arms:
        r2 = {name: report["bodies"][name]["pact_d"]["r2"] for name in names}
        report["proceed"] = bool(min(r2.values()) >= 0.9)
        print(f"G-D0: pact_d R² {r2}; proceed to G-D1: {report['proceed']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
