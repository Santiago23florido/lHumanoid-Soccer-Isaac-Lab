"""Gate G-A3: the state encoder on physics for the loads and the CoM velocity, on a held-out body.

Registered in ``docs/report/research_notes/icml/preregistration.md`` before it ran. G-A2's
shared encoder did not reach the NAO when trained on the G1, H1 and Cassie (CoM velocity
and loads). Here the two learned quantities get physics bases too, from the card and the
drives alone (``pact/anchors.py``):

- loads: ``DriveLoads`` -- each joint's PD torque (recorded gains and targets) balanced by
  the leg's weight and the ground's wrench at the sole;
- CoM velocity: odometry -- minus each foot's velocity relative to the CoM (exact from the
  anchors), weighted by those loads (a loaded foot stands still).

Arms, all on the co-execution captures of the four trackers (``pact_coexecute.sh``, map
stage), split by gait probe (probes with index % 4 == 0 are the test set):

- ``physics``: no learning at all, the physics bases as the estimate;
- ``pact_k``: G-A2's shared encoder (kinematic anchors; CoM velocity, loads, height learned);
- ``pact_kd``: the same encoder reading the physics bases too; the loads are learned as a
  correction to the drive loads, the CoM velocity directly (odometry is an input only: on
  the training bodies' expert captures it errs by several times the CoM velocity's spread,
  in transitions and wherever a loaded foot rolls);
- ``mlp_k``: one MLP per body, trained on that body (the in-body reference).

The shared arms train on every body but ``--hold-out``. Metric: RMSE over the 27 fields in
pooled units (pooled over the training bodies' training rows); bootstrap over test
environments.

Usage::

    python transfer/scripts/pact_gate_a3.py --bodies g1,nao,h1,cassie --hold-out nao
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
from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.data import load_body  # noqa: E402
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from pact_check_anchors import sole_override  # noqa: E402
from pact_gate_a import bootstrap_rmse, field_rmse, per_body_mlp  # noqa: E402
from pact_gate_a2 import LEARNED, S, SharedAnchored, analytic_z, assemble, targets_of  # noqa: E402
from torch import nn  # noqa: E402

DRIVES = ("drive_stiffness", "drive_damping", "drive_effort_limit")
PROBES = 71


def physics_bases(drive: DriveLoads, w, analytic: np.ndarray, time_unit: float) -> np.ndarray:
    """``(N, 5)``: odometry CoM velocity (3, ``z*`` units) and drive loads (2, ``m g``) at
    each window's last step."""
    q = torch.as_tensor(w.joint_obs[:, :, -1, 0])
    qd = torch.as_tensor(w.joint_obs[:, :, -1, 1] / time_unit)
    g = torch.as_tensor(w.imu[:, -1, :3])
    target = torch.as_tensor(w.extra["joint_pos_target"])
    loads = []
    with torch.no_grad():
        for rows in np.array_split(np.arange(len(q)), max(1, len(q) // 4096)):
            loads.append(drive(q[rows], qd[rows], g[rows], target=target[rows])["loads"].numpy())
    loads = np.concatenate(loads)
    weights = np.clip(loads, 0.0, None)
    total = weights.sum(-1, keepdims=True)
    weights = np.where(total > 0.05, weights / np.maximum(total, 1e-9), 0.5)
    relative = analytic[:, S["foot_velocity"]].reshape(-1, 2, 3)
    odometry = -(weights[..., None] * relative).sum(1)
    return np.concatenate([odometry, loads], axis=1).astype(np.float32)


def with_physics(analytic: np.ndarray, physics: np.ndarray) -> np.ndarray:
    """The analytic ``z*`` with the physics bases in the CoM velocity and load slots, and
    each foot's velocity completed with the odometry."""
    z = analytic.copy()
    z[:, S["com_velocity"]] = physics[:, :3]
    z[:, S["foot_load"]] = physics[:, 3:]
    z[:, S["foot_velocity"]] = analytic[:, S["foot_velocity"]] + np.tile(physics[:, :3], 2)
    return z


def learned_base(physics: np.ndarray) -> np.ndarray:
    """``(N, 6)``: what ``pact_kd``'s outputs correct -- nothing for the CoM velocity and
    the height correction, the drive loads for the loads."""
    zeros = np.zeros((len(physics), 4), np.float32)
    return np.concatenate([zeros, physics[:, 3:]], axis=1)


def residual_targets(z_true: np.ndarray, analytic: np.ndarray, physics: np.ndarray):
    """``(N, 6)``: what ``pact_kd`` learns (G-A2's targets less :func:`learned_base`)."""
    return targets_of(z_true, analytic) - learned_base(physics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default="nao")
    parser.add_argument("--captures", default="logs/pact/coexec/{body}_map.npz")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--arms", default="physics,pact_k,pact_kd,mlp_k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_gate_a3.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    arms = args.arms.split(",")
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    names = [b for b in args.bodies.split(",") if b]
    train_index = [i for i, b in enumerate(names) if b != args.hold_out]

    descriptors, contacts, windows, analytic, physics = [], [], [], [], []
    for body in names:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        unit = entry.get("leg_length_unit")
        descriptor = describe(card, leg_length=unit)
        paths = [REPO / args.captures.format(body=body)]
        w = load_body(
            body,
            paths,
            args.history,
            descriptor.leg_length,
            descriptor.joint_names,
            extra=("joint_pos_target",),
            constants=DRIVES + ("default_joint_pos",),
        )
        anchors = KinematicAnchors(card, unit, sole_override(body))
        drive = DriveLoads(card, anchors)
        drive.set_drives(*(w.constants[k] for k in DRIVES))
        time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
        a = analytic_z(anchors, w, time_unit)
        descriptors.append(descriptor)
        contacts.append(contact_joints(card))
        windows.append(w)
        analytic.append(a)
        physics.append(physics_bases(drive, w, a, time_unit))
        print(f"{body}: {len(w.z)} windows", flush=True)

    test = [((w.group % 1000) % PROBES) % 4 == 0 for w in windows]
    pooled = np.concatenate([windows[i].z[~test[i]] for i in train_index])
    z_mean, z_std = pooled.mean(0), np.maximum(pooled.std(0), 1e-3)
    as_t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)  # noqa: E731
    z_mean_t, z_std_t = as_t(z_mean), as_t(z_std)
    table = stack_bodies(descriptors, contacts).to(device)
    count = table.mask.shape[1]
    train_rows = {i: np.nonzero(~test[i])[0] for i in range(len(names))}
    per = max(1, args.batch // len(train_index))
    predictions: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in arms}
    params: dict[str, int] = {}

    def fit(model: nn.Module, step_fn, label: str) -> None:
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

    if "physics" in arms:
        for i in range(len(names)):
            rows = np.nonzero(test[i])[0]
            z = with_physics(analytic[i][rows], physics[i][rows])
            predictions["physics"][i] = (z - z_mean) / z_std

    for arm in [a for a in arms if a in ("pact_k", "pact_kd")]:
        uses_physics = arm == "pact_kd"
        inputs = [
            with_physics(a, p) if uses_physics else a
            for a, p in zip(analytic, physics, strict=True)
        ]
        targets = [
            residual_targets(w.z, a, p) if uses_physics else targets_of(w.z, a)
            for w, a, p in zip(windows, analytic, physics, strict=True)
        ]
        a_pool = np.concatenate([inputs[i][train_rows[i]] for i in train_index])
        a_mean, a_std = a_pool.mean(0), np.maximum(a_pool.std(0), 1e-3)
        t_pool = np.concatenate([targets[i][train_rows[i]] for i in train_index])
        t_mean, t_std = t_pool.mean(0), np.maximum(t_pool.std(0), 1e-3)
        t_mean_t, t_std_t = as_t(t_mean), as_t(t_std)

        def batch(i: int, rows: np.ndarray, inputs=inputs, targets=targets, a_mean=a_mean,
                  a_std=a_std, t_mean=t_mean, t_std=t_std):  # fmt: skip
            w = windows[i]
            obs = np.zeros((len(rows), count, args.history, 2), dtype=np.float32)
            obs[:, : w.joint_obs.shape[1]] = w.joint_obs[rows]
            return (
                as_t(obs),
                as_t(w.imu[rows]),
                as_t((inputs[i][rows] - a_mean) / a_std),
                as_t((targets[i][rows] - t_mean) / t_std),
            )

        dims = PactDims(observation=2, history=args.history, width=args.width, anchor_inputs=True)
        shared = SharedAnchored(dims).to(device)

        def step_shared(batch=batch, shared=shared):
            parts = [batch(i, rng.choice(train_rows[i], per)) for i in train_index]
            which = torch.cat(
                [torch.full((per,), i, device=device, dtype=torch.long) for i in train_index]
            )
            obs, imu, a_in, target = (torch.cat([p[k] for p in parts]) for k in range(4))
            return (shared(table.take(which), obs, imu, a_in) - target).square().mean()

        fit(shared, step_shared, arm)
        shared.eval()
        params[arm] = sum(p.numel() for p in shared.parameters())
        for i in range(len(names)):
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(test[i])[0]):
                    obs, imu, a_in, _ = batch(i, rows)
                    which = torch.full((len(rows),), i, device=device, dtype=torch.long)
                    learned = shared(table.take(which), obs, imu, a_in) * t_std_t + t_mean_t
                    if uses_physics:
                        learned = learned + as_t(learned_base(physics[i][rows]))
                    z = assemble(as_t(analytic[i][rows]), learned)
                    out.append(((z - z_mean_t) / z_std_t).cpu().numpy())
            predictions[arm][i] = np.concatenate(out)

    if "mlp_k" in arms:
        for i in range(len(names)):
            w, a = windows[i], analytic[i]
            a_mean_i, a_std_i = a[train_rows[i]].mean(0), np.maximum(a[train_rows[i]].std(0), 1e-3)
            t = targets_of(w.z, a)
            t_mean_i, t_std_i = t[train_rows[i]].mean(0), np.maximum(t[train_rows[i]].std(0), 1e-3)
            n_j = w.joint_obs.shape[1]
            model = per_body_mlp(n_j * args.history * 2 + args.history * 6 + SKILL_LAYOUT.size)
            model[-1] = nn.Linear(512, LEARNED)
            model = model.to(device)

            def features(rows, w=w, a=a, a_mean_i=a_mean_i, a_std_i=a_std_i, n_j=n_j):
                return torch.cat(
                    [
                        as_t(w.joint_obs[rows]).flatten(1),
                        as_t(w.imu[rows]).flatten(1),
                        as_t((a[rows] - a_mean_i) / a_std_i),
                    ],
                    dim=1,
                )

            def step_mlp(i=i, model=model, features=features, t=t, t_mean_i=t_mean_i,
                         t_std_i=t_std_i):  # fmt: skip
                rows = rng.choice(train_rows[i], args.batch // len(train_index))
                target = as_t((t[rows] - t_mean_i) / t_std_i)
                return (model(features(rows)) - target).square().mean()

            fit(model, step_mlp, f"mlp_k/{names[i]}")
            model.eval()
            params[f"mlp_k/{names[i]}"] = sum(p.numel() for p in model.parameters())
            out = []
            with torch.no_grad():
                for rows in chunks(np.nonzero(test[i])[0]):
                    learned = model(features(rows)) * as_t(t_std_i) + as_t(t_mean_i)
                    z = assemble(as_t(a[rows]), learned)
                    out.append(((z - z_mean_t) / z_std_t).cpu().numpy())
            predictions["mlp_k"][i] = np.concatenate(out)

    report = {
        "gate": "G-A3",
        "config": {k: str(v) for k, v in vars(args).items()},
        "parameters": params,
        "bodies": {},
    }
    for i, body in enumerate(names):
        rows = np.nonzero(test[i])[0]
        truth = (windows[i].z[rows] - z_mean) / z_std
        groups = windows[i].group[rows]
        entry: dict = {"held_out": body == args.hold_out}
        errors = {arm: predictions[arm][i] - truth for arm in arms if i in predictions[arm]}
        for arm, err in errors.items():
            entry[arm] = field_rmse(err)
        for a_arm, b_arm in (("pact_kd", "pact_k"), ("pact_kd", "mlp_k"), ("physics", "pact_k")):
            if a_arm in errors and b_arm in errors:
                _, _, diff, se = bootstrap_rmse(
                    (errors[a_arm] ** 2).mean(1),
                    (errors[b_arm] ** 2).mean(1),
                    groups,
                    args.bootstrap,
                    args.seed,
                )
                entry[f"{a_arm}_minus_{b_arm}"] = diff
                entry[f"{a_arm}_minus_{b_arm}_se"] = se
        report["bodies"][body] = entry
        print(
            body,
            "(held out)" if entry["held_out"] else "",
            {arm: round(entry[arm]["all"], 4) for arm in errors},
            {k: round(v, 4) for k, v in entry.items() if "minus" in k},
        )
    held = report["bodies"].get(args.hold_out, {})
    if "pact_kd_minus_pact_k" in held:
        passes = held["pact_kd_minus_pact_k"] < -2 * held["pact_kd_minus_pact_k_se"]
        report["passes"] = bool(passes)
        print(f"G-A3 {'PASSES' if passes else 'FAILS'} on the held-out {args.hold_out}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
