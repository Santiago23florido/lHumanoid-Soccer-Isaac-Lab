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

DAgger (G-D1): each round's captures (``capture_contract.py --pact-model``) join the map
captures, and the previous round's model is the start::

    python transfer/scripts/pact_gate_d.py --dagger "logs/pact/dagger/<tag>/r*/{body}.npz"
        --init logs/pact/models/<previous>.pt --steps 8000 --save logs/pact/models/<next>.pt

(one command; ``pact_dagger.sh`` runs the rounds).

Windows: a DAgger capture holds many episode starts (PACT falls where the tracker did not),
and the driver acts there with its history refilled by the first reading. Windows that
cross the start of an episode begun inside the capture therefore repeat that reading
(:func:`window_rows`) and are trained on; the offline metric keeps G-D0's complete windows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import torch  # noqa: E402
from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.mirror import SampleMirror  # noqa: E402
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import (  # noqa: E402
    PLAN_LIMIT,
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


def unclamped(estimate: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """``(...)`` bool: rows whose estimate the tracker's view holds unclamped. A body far out
    of the plan's range (fallen) is seen clamped at ``PLAN_LIMIT`` normalised units; the
    clamped view is what the tracker, and so its label, saw: those rows are kept."""
    return (np.abs((estimate - mean) / std) < PLAN_LIMIT).all(-1)


def window_rows(
    done: np.ndarray, history: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(t, env, rows (N, history), start (N,))``: every step with a label in its episode
    (``t + 1`` in the capture, no reset at ``t``) whose window either lies inside one
    episode (no reset in ``[t - history, t]``, ``t >= history``: G-D0's windows) or crosses
    the start of an episode that began inside the capture (``start``). The rows before the
    start repeat its first reading: the driver refills its history at a reset, as the
    observation manager's history buffers do, so a policy acts there and must learn there.
    An episode that began before the capture has no known start and keeps G-D0's rule.
    Ordered by step, then environment."""
    steps, envs = done.shape
    first = np.full((steps, envs), -1, dtype=np.int64)
    current = np.full(envs, -1, dtype=np.int64)
    for s in range(steps):
        first[s] = current
        current = np.where(done[s], s + 1, current)
    ended = np.zeros((steps + 1, envs), dtype=np.int64)
    ended[1:] = np.cumsum(done, axis=0)
    tt = np.arange(steps)
    resets = ended[tt + 1] - ended[np.maximum(tt - history, 0)]  # resets in [t - h, t]
    full = (tt[:, None] >= history) & (resets == 0)
    starting = (first >= 0) & (tt[:, None] - first < history) & ~done.astype(bool)
    valid = (full | starting) & (tt[:, None] + 1 < steps)
    t, env = np.nonzero(valid)
    base = np.maximum(first[t, env], 0)
    rows = np.maximum(t[:, None] + np.arange(-history + 1, 1)[None], base[:, None])
    return t, env, rows, starting[t, env]


class BodyData:
    """One body's samples, from its map capture and any DAgger captures: inputs and labels.

    In a map capture the tracker drove, and the label is the target it set (read at the
    next step); in a DAgger capture (``capture_contract.py --pact-model``) PACT drove and
    the label is the tracker's own action on that state, ``q0 + scale * a`` on the action's
    joints. The offline metric reads the map capture's test probes only (``evaluate``).
    """

    def __init__(self, body, args, registry):
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        unit = entry.get("leg_length_unit")
        self.descriptor = describe(card, leg_length=unit)
        self.contacts = contact_joints(card)
        self.anchors = KinematicAnchors(card, unit, sole_override(body))
        self.time_unit = float(np.sqrt(self.descriptor.leg_length / GRAVITY))
        self.card, self.name = card, body
        contract = dict(c.split("=") for c in args.contracts.split(","))[body]
        norm = torch.load(
            REPO / f"logs/contract/{contract}/generator.pt", map_location="cpu", weights_only=False
        )["generator"]
        self.skill = norm["skill_norm.mean"].numpy(), norm["skill_norm.std"].numpy()
        paths = [REPO / args.captures.format(body=body)]
        if args.dagger:
            paths += sorted(REPO.glob(args.dagger.format(body=body)))
        parts = [
            self._samples(path, args, args.stride if i == 0 else args.dagger_stride)
            for i, path in enumerate(paths)
        ]
        for i, part in enumerate(parts):
            part["group"] = i * 1000 + part["group"]
            part["evaluate"] = part["test"] & (i == 0) & ~part["start"]
        for key in parts[0]:
            if key not in ("view", "times", "controlled"):
                setattr(self, key, np.concatenate([part[key] for part in parts]))
        self.view, self.times = parts[0]["view"], parts[0]["times"]
        self.controlled = parts[0]["controlled"]
        self.captures = [str(path.relative_to(REPO)) for path in paths]
        self.capture_samples = [int(len(part["label"])) for part in parts]
        q0 = np.asarray(np.load(paths[0])["default_joint_pos"], dtype=np.float32)
        self.mirror = SampleMirror.for_body(REPO, body, self.descriptor.joint_names, q0)

    def _samples(self, path: Path, args, stride: int) -> dict:
        h = args.history
        data = np.load(path)
        names = tuple(str(n) for n in data["joint_names"])
        if names != tuple(self.descriptor.joint_names):
            raise ValueError(f"{path}: joints {names[:4]}... differ from the card's")
        done = data["done"]
        t, env, rows, start = window_rows(done, h)
        keep = np.arange(len(t))[::stride]
        t, env, rows, start = t[keep], env[keep], rows[keep], start[keep]

        q0 = data["default_joint_pos"].astype(np.float32)
        q, target = data["joint_pos"], data["joint_pos_target"]
        qd = data["joint_vel"] * self.time_unit
        cols = env[:, None]
        q_w, qd_w = q[rows, cols], qd[rows, cols]  # (N, h, J)
        out: dict = {}
        out["joint_obs"] = (
            np.stack([q_w - q0, qd_w, target[rows, cols] - q0], axis=-1)
            .transpose(0, 2, 1, 3)
            .astype(np.float32)
        )  # (N, J, h, 3)
        imu = np.concatenate(
            [data["projected_gravity"], data["base_ang_vel"] * self.time_unit], axis=-1
        )
        out["imu"] = imu[rows, cols].astype(np.float32)
        w = SimpleNamespace(
            joint_obs=np.stack([q_w, qd_w], axis=-1).transpose(0, 2, 1, 3).astype(np.float32),
            imu=out["imu"],
            z=data["skill_state"][t, env].astype(np.float32),
            extra={"joint_pos_target": target[t, env].astype(np.float32)},
        )
        drive = DriveLoads(self.card, self.anchors)
        drive.set_drives(*(np.asarray(data[k]) for k in DRIVES))
        analytic = analytic_z(self.anchors, w, self.time_unit)
        physics = physics_bases(drive, w, analytic, self.time_unit)

        if "action_joints" in data.files:
            joints = data["action_joints"]
            label = np.zeros((len(t), len(q0)), np.float32)
            label[:, joints] = data["action_scale"] * data["action_teacher"][t, env]
            controlled = np.zeros(len(q0), bool)
            controlled[joints] = True
        else:
            label = target[t + 1, env] - q0
            controlled = target.reshape(-1, target.shape[-1]).std(0) > 1e-6
        out["label"] = label.astype(np.float32)
        out["hold"] = (target[t, env] - q0).astype(np.float32)
        out["controlled"] = controlled
        out["anchor_in"] = with_physics(analytic, physics)
        out["learned"] = residual_targets(w.z, analytic, physics)
        out["z"] = w.z
        out["start"] = start
        mean, std = self.skill
        obs = data["policy_obs"]
        estimate = data["estimate"]
        view = self._view(obs, estimate, mean, std)
        tail = obs[t, env, obs.shape[-1] - (view * (SKILL + 3) + SKILL) :]
        upcoming, est, path_ahead = plan_from_view(tail, view, mean, std)
        inside = unclamped(estimate[t, env], mean, std)
        error = float(np.abs(est - estimate[t, env])[inside].max()) if inside.any() else 0.0
        if error > 1e-3:
            raise ValueError(f"{path}: decoded estimate differs from the capture's by {error}")
        out["clamped"] = ~inside
        out["plan"] = upcoming.astype(np.float32)
        out["relative"] = (upcoming - est[:, None]).astype(np.float32)
        out["path"] = path_ahead.astype(np.float32)
        out["view"] = view
        dt_star = float(data["control_dt"]) / self.time_unit
        out["times"] = (np.arange(1, view + 1) * dt_star).astype(np.float32)
        out["group"] = env.astype(np.int64)
        if "probe_index" in data.files:
            probe = data["probe_index"][env]
            if not np.array_equal(probe, env % PROBES):
                raise ValueError(f"{path}: environments are not laid out probe by probe")
            out["test"] = probe % 4 == 0
        else:  # a capture in the training task's own plans (pact_dagger.sh, PACT_DAGGER_ENV=train)
            out["test"] = np.zeros(len(env), bool)
        return out

    @staticmethod
    def _view(obs, estimate, mean, std) -> int:
        """The view length whose decoded estimate matches the capture's (rows 20-200, where
        the view is not clamped)."""
        rows = slice(20, 200)
        inside = unclamped(estimate[rows], mean, std)
        for view in range(1, 20):
            width = view * (SKILL + 3) + SKILL
            if width > obs.shape[-1]:
                break
            start = obs.shape[-1] - width + view * SKILL
            zhat = obs[rows, :, start : start + SKILL] * std + mean
            if inside.any() and np.abs(zhat - estimate[rows])[inside].max() < 1e-3:
                return view
        raise ValueError("no plan view decodes to the capture's estimate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default="")
    parser.add_argument("--captures", default="logs/pact/coexec/{body}_map.npz")
    parser.add_argument(
        "--dagger",
        default="",
        help="glob of a body's DAgger captures, e.g. logs/pact/dagger/r*/{body}.npz",
    )
    parser.add_argument("--dagger-stride", type=int, default=2)
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="start pact_d from this checkpoint and its normalisers",
    )
    parser.add_argument("--contracts", default=CONTRACTS)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="keep every n-th window")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="train pact_d on each sample or its mirror image, at random (pact/mirror.py)",
    )
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

    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    if init is not None:
        z_mean, z_std, a_mean, a_std, t_mean, t_std = (
            np.asarray(init[k]) for k in ("z_mean", "z_std", "a_mean", "a_std", "t_mean", "t_std")
        )
    else:
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
    if init is not None:
        label_scale = float(init["label_scale"])
    as_t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)  # noqa: E731
    table = stack_bodies([b.descriptor for b in bodies], [b.contacts for b in bodies]).to(device)
    count = table.mask.shape[1]
    steps_max = max(b.view for b in bodies)
    controlled = torch.zeros(len(bodies), count, dtype=torch.bool, device=device)
    for i, b in enumerate(bodies):
        controlled[i, : len(b.controlled)] = torch.as_tensor(b.controlled, device=device)

    def batch(i: int, rows: np.ndarray, augment: bool = False) -> dict[str, torch.Tensor]:
        b = bodies[i]
        n, j, v = len(rows), b.joint_obs.shape[1], b.view
        raw = {
            "joint_obs": b.joint_obs[rows],
            "imu": b.imu[rows],
            "anchor_in": b.anchor_in[rows],
            "plan": b.plan[rows],
            "relative": b.relative[rows],
            "path": b.path[rows],
            "label": b.label[rows],
            "learned": b.learned[rows],
        }
        if augment and args.mirror:  # each sample or its mirror image, at random
            flip = rng.random(n) < 0.5
            m = b.mirror
            for key, reflect in (
                ("joint_obs", m.joint_obs),
                ("imu", m.imu),
                ("anchor_in", m.skill),
                ("plan", m.skill),
                ("relative", m.skill),
                ("path", m.path),
                ("label", m.positions),
                ("learned", m.learned),
            ):
                raw[key] = raw[key].copy()
                raw[key][flip] = reflect(raw[key][flip])
        obs = np.zeros((n, count, args.history, 3), np.float32)
        obs[:, :j] = raw["joint_obs"]
        plan = np.zeros((n, steps_max, SKILL), np.float32)
        plan[:, :v] = (raw["plan"] - z_mean) / z_std
        rel = np.zeros((n, steps_max, SKILL), np.float32)
        rel[:, :v] = raw["relative"] / z_std
        path = np.zeros((n, steps_max, 3), np.float32)
        path[:, :v] = raw["path"]
        times = np.zeros((n, steps_max), np.float32)
        times[:, :v] = b.times
        mask = np.zeros((n, steps_max), bool)
        mask[:, :v] = True
        label = np.zeros((n, count), np.float32)
        label[:, :j] = raw["label"]
        return {
            "joint_obs": as_t(obs),
            "imu": as_t(raw["imu"]),
            "anchor_in": as_t((raw["anchor_in"] - a_mean) / a_std),
            "plan": as_t(plan),
            "plan_relative": as_t(rel),
            "plan_time": as_t(times),
            "plan_path": as_t(path),
            "plan_mask": torch.as_tensor(mask, device=device),
            "label": as_t(label),
            "learned": as_t((raw["learned"] - t_mean) / t_std),
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
            predictions["hold"][i] = b.hold[b.evaluate]

    if "pact_d" in arms:
        dims = PactDims(
            observation=3,
            history=args.history,
            width=args.width,
            anchor_inputs=True,
            timed_plan=True,
        )
        model = PactPolicy(dims).to(device)
        if init is not None:
            if PactDims(**init["dims"]) != dims:
                raise ValueError("--init was trained with other dimensions")
            model.load_state_dict(init["state"])
        per = max(1, args.batch // len(train_index))

        def step_pact():
            parts = [batch(i, rng.choice(train_rows[i], per), augment=True) for i in train_index]
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
                for rows in chunks(np.nonzero(b.evaluate)[0]):
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
                for rows in chunks(np.nonzero(b.evaluate)[0]):
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
        truth = b.label[b.evaluate][:, moved]
        groups = b.group[b.evaluate]
        variance = float(truth.var(0).mean())
        entry: dict = {
            "held_out": b.name == args.hold_out,
            "view": b.view,
            "controlled_joints": int(moved.sum()),
            "test_samples": int(b.evaluate.sum()),
            "captures": dict(zip(b.captures, b.capture_samples, strict=True)),
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
