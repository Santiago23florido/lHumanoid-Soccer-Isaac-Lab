"""Gate G-N1: adding a body cheaply -- few-shot adaptation of PACT to a held-out body.

Registered in ``docs/report/research_notes/icml/preregistration.md`` before it ran. A model
pretrained without one body (exactly as G-A3's encoder or G-X's translator) is adapted to
that body with k of its training probes (encoder, co-execution map captures) or k of its
training chains (translator, replay captures):

- ``zero``: no adaptation (the zero-shot reading);
- ``etune``: a residual on the body's embodiment tokens ``E_b`` (for the encoder also on
  its per-joint descriptor tokens), everything else frozen;
- ``bitfit``: ``etune`` plus every bias and LayerNorm parameter;
- ``full``: every parameter;
- ``scratch``: the same architecture from a random init, on the same k only.

Budgets are nested (the first k of one permutation), the same draw for every arm. Each
adaptation runs a fixed number of steps, no early stopping. Metric as G-A3 (encoder) or
G-X (translator) on the held-out body's test probes or chains. Cost: tuned parameters
(counting only the held-out body's own tokens), adaptation seconds, simulated seconds of
the body's data. Every arm is also compared, paired over the same bootstrap draws, with
the budget's best arm (``minus_best``), which the registered selection rule reads.
Pretrained models are cached in ``logs/pact/models/``.

Usage::

    python transfer/scripts/pact_gate_n1.py --part translator --hold-out nao
    python transfer/scripts/pact_gate_n1.py --part encoder --hold-out nao --encoder pact_kd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import torch  # noqa: E402
from humanoid_transfer.transfer.latent.pact.coexec import pair, windows  # noqa: E402
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.tokens import split  # noqa: E402
from humanoid_transfer.transfer.latent.pact.translate import CrossBodyTranslator  # noqa: E402
from torch import nn  # noqa: E402

ARMS = ("zero", "etune", "bitfit", "full", "scratch")
LR = {"etune": 1e-3, "bitfit": 1e-3, "full": 1e-4, "scratch": 3e-4}
MODELS = REPO / "logs" / "pact" / "models"


def data_key(args) -> str:
    """A cached pretrained model is reused only for the same bodies and captures."""
    text = "|".join((args.bodies, args.encoder_captures, args.translator_captures))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def nested_budgets(units: np.ndarray, budgets: list[str], seed: int) -> dict[str, np.ndarray]:
    """The first k units of one permutation for every k ("all" = every unit)."""
    order = np.random.default_rng(seed).permutation(units)
    return {k: order if k == "all" else order[: int(k)] for k in budgets}


def bias_and_norm(model: nn.Module) -> list[nn.Parameter]:
    """Every bias and every LayerNorm parameter (BitFit's set), once each."""
    chosen: dict[int, nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters(recurse=False):
                chosen[id(p)] = p
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            chosen[id(p)] = p
    return list(chosen.values())


def select(arm: str, wrapper: nn.Module) -> list[nn.Parameter]:
    """The parameters an arm tunes; every other parameter of ``wrapper`` is frozen."""
    residual = [wrapper.delta_e] + ([wrapper.delta_d] if hasattr(wrapper, "delta_d") else [])
    if arm == "zero":
        tuned: list[nn.Parameter] = []
    elif arm == "etune":
        tuned = residual
    elif arm == "bitfit":
        tuned = residual + bias_and_norm(wrapper.base)
    else:  # full, scratch
        tuned = residual + list(wrapper.base.parameters())
    ids = {id(p) for p in tuned}
    for p in wrapper.parameters():
        p.requires_grad_(id(p) in ids)
    return tuned


def counted(arm: str, wrapper: nn.Module, own_joints: int | None) -> int:
    """Tuned parameters, the residual counted on the held-out body's own tokens only."""
    tuned = select(arm, wrapper)
    total = sum(p.numel() for p in tuned)
    if own_joints is not None and arm != "zero":
        padded = wrapper.delta_d.shape[0]
        total -= (padded - own_joints) * wrapper.delta_d.shape[1]
    return total


# --------------------------------------------------------------------------- translator


class TranslatorResidual(nn.Module):
    """A translator whose held-out body's ``E_b`` carries a learned residual."""

    def __init__(self, base: CrossBodyTranslator, body: int, tokens: int, width: int) -> None:
        super().__init__()
        self.base = base
        self.body = body
        self.delta_e = nn.Parameter(torch.zeros(tokens, width))

    def bodies(self, table):
        tokens = self.base.bodies(table)
        own = (torch.arange(len(tokens), device=tokens.device) == self.body).float()
        return tokens + own[:, None, None] * self.delta_e[None]

    def forward(self, source, source_body, target_body, tokens):
        return self.base(source, source_body, target_body, tokens)


def run_translator(args, device) -> dict:
    import pact_gate_x as gx

    names = [b for b in args.bodies.split(",") if b]
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    captures = [np.load(REPO / args.translator_captures.format(body=b)) for b in names]
    step = max(gx.body_dt(c) for c in captures)
    paired = pair(captures, step)
    test_chain = paired.chains % 5 == 0
    train_valid = paired.valid & ~test_chain[:, None]
    flat = paired.z[:, train_valid]
    mean, std = flat.reshape(-1, flat.shape[-1]).mean(0), flat.reshape(-1, flat.shape[-1]).std(0)
    std = np.maximum(std, 1e-3)
    data = torch.as_tensor((paired.z - mean) / std, dtype=torch.float32, device=device)
    window = args.window
    train_t = torch.as_tensor(windows(train_valid, window, args.stride), device=device)
    test_rows = windows(paired.valid & test_chain[:, None], window, args.stride)
    test_t = torch.as_tensor(test_rows, device=device)

    descriptors, contacts = [], []
    for body in names:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        descriptors.append(describe(card, leg_length=entry.get("leg_length_unit")))
        contacts.append(contact_joints(card))
    table = stack_bodies(descriptors, contacts).to(device)
    dims = PactDims(width=args.width)
    h = names.index(args.hold_out)
    seen = [b for b in range(len(names)) if b != h]
    held_pairs = [(s, h) for s in seen]

    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"n1_translator_{args.hold_out}_s{args.seed}_p{args.pretrain_steps}_{data_key(args)}.pt"
    pretrained = CrossBodyTranslator(dims, window).to(device)
    if path.exists():
        pretrained.load_state_dict(torch.load(path, map_location=device))
        print(f"pretrained translator loaded from {path}")
    else:
        settings = SimpleNamespace(steps=args.pretrain_steps, batch=256, lr=3e-4, window=window)
        gx.train(pretrained, data, train_t, seen, table, settings, device, "pretrain")
        torch.save(pretrained.state_dict(), path)
    state = {k: v.clone() for k, v in pretrained.state_dict().items()}

    _, chain_rows = np.unique(test_rows[:, 0], return_inverse=True)
    held = int(chain_rows.max()) + 1
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, held, size=(args.bootstrap, held))
    counts = np.stack([np.bincount(d, minlength=held) for d in draws]).astype(np.float64)

    def evaluate(model) -> np.ndarray:
        model.eval()
        e = gx.errors(model, data, test_t, held_pairs, table, window)
        return gx.target_values(e, held_pairs, h)

    copy_values = gx.target_values(
        gx.errors(None, data, test_t, held_pairs, table, window), held_pairs, h
    )
    seconds_per_instant = step * math.sqrt(float(captures[h]["leg_length"]) / GRAVITY)
    units = np.nonzero(~test_chain)[0]
    budgets = nested_budgets(units, args.budgets.split(","), args.seed)
    seen_t = torch.as_tensor(seen, device=device)

    def adapt(model, tuned, rows, arm) -> float:
        if not tuned:
            return 0.0
        optimiser = torch.optim.AdamW(tuned, lr=LR[arm], weight_decay=0.0)
        start = time.time()
        model.train()
        for _ in range(args.adapt_steps):
            pick = rows[torch.randint(len(rows), (args.batch,), device=device)]
            other = seen_t[torch.randint(len(seen), (args.batch,), device=device)]
            own = torch.full_like(other, h)
            as_source = torch.rand(args.batch, device=device) < 0.5
            s, t = torch.where(as_source, own, other), torch.where(as_source, other, own)
            source, target = gx.gather(data, pick, s, window), gx.gather(data, pick, t, window)
            loss = (model(source, s, t, model.bodies(table)) - target).pow(2).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuned, 1.0)
            optimiser.step()
        return time.time() - start

    report = {
        "part": "translator",
        "grid_step": step,
        "copy": gx.rmse(copy_values),
        "budgets": {},
    }
    for k, chosen in budgets.items():
        valid_k = train_valid & np.isin(np.arange(len(paired.chains)), chosen)[:, None]
        rows_k = windows(valid_k, window, args.stride)
        rows_t = torch.as_tensor(rows_k, device=device)
        entry = {
            "units": int(len(chosen)),
            "windows": int(len(rows_k)),
            "sim_seconds": float(valid_k.sum() * seconds_per_instant),
            "arms": {},
        }
        values = {}
        for arm in args.arms.split(","):
            torch.manual_seed(args.seed)
            base = CrossBodyTranslator(dims, window).to(device)
            if arm != "scratch":
                base.load_state_dict(state)
            model = TranslatorResidual(base, h, dims.body_tokens, dims.width).to(device)
            n_params = counted(arm, model, None)
            tuned = select(arm, model)
            seconds = adapt(model, tuned, rows_t, arm) if len(rows_k) else 0.0
            with torch.no_grad():
                values[arm] = evaluate(model)
            entry["arms"][arm] = {
                "rmse": gx.rmse(values[arm]),
                "tuned_parameters": int(n_params),
                "adapt_seconds": round(seconds, 1),
            }
            print(f"k={k} {arm}: {entry['arms'][arm]}", flush=True)
        boot = {arm: gx.bootstrap_rmse(v, chain_rows, counts) for arm, v in values.items()}
        entry["best"] = best = min(values, key=lambda a: entry["arms"][a]["rmse"])
        for arm in values:
            for name, ref in (("zero", "zero"), ("scratch", "scratch"), ("best", best)):
                if ref in values and arm != ref:
                    row = entry["arms"][arm]
                    row[f"minus_{name}"] = row["rmse"] - entry["arms"][ref]["rmse"]
                    row[f"minus_{name}_se"] = float(np.std(boot[arm] - boot[ref]))
        report["budgets"][k] = entry
    return report


# ------------------------------------------------------------------------------ encoder


class EncoderResidual(nn.Module):
    """G-A2's shared encoder whose ``E_b`` and descriptor tokens carry learned residuals."""

    def __init__(self, base, tokens: int, joints: int, width: int) -> None:
        super().__init__()
        self.base = base
        self.delta_e = nn.Parameter(torch.zeros(tokens, width))
        self.delta_d = nn.Parameter(torch.zeros(joints, width))

    def forward(self, bodies, joint_obs, imu, anchor_in):
        embodiment, descriptor = self.base.embodiment(bodies)
        embodiment = embodiment + self.delta_e
        descriptor = descriptor + self.delta_d
        centroid, contacts = split(anchor_in)
        out = self.base.state(bodies, descriptor, embodiment, joint_obs, imu, (centroid, contacts))
        # The output layout of pact_gate_a2.SharedAnchored.forward (checked at start-up).
        return torch.cat(
            [out["centroid"][:, 1:4], out["centroid"][:, 0:1], out["contacts"][:, :, 6]], dim=1
        )


def run_encoder(args, device) -> dict:
    import pact_gate_a3 as a3
    from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
    from humanoid_transfer.transfer.latent.pact.data import load_body
    from pact_check_anchors import sole_override
    from pact_gate_a import bootstrap_rmse, field_rmse
    from pact_gate_a2 import SharedAnchored, analytic_z, assemble, targets_of

    names = [b for b in args.bodies.split(",") if b]
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))
    h = names.index(args.hold_out)
    train_index = [i for i in range(len(names)) if i != h]
    descriptors, contacts, wins, analytic, physics, control_dt = [], [], [], [], [], []
    for body in names:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        unit = entry.get("leg_length_unit")
        descriptor = describe(card, leg_length=unit)
        path = REPO / args.encoder_captures.format(body=body)
        w = load_body(
            body,
            [path],
            args.history,
            descriptor.leg_length,
            descriptor.joint_names,
            extra=("joint_pos_target",),
            constants=a3.DRIVES + ("default_joint_pos",),
        )
        anchors = KinematicAnchors(card, unit, sole_override(body))
        drive = DriveLoads(card, anchors)
        drive.set_drives(*(w.constants[k] for k in a3.DRIVES))
        time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
        a = analytic_z(anchors, w, time_unit)
        descriptors.append(descriptor)
        contacts.append(contact_joints(card))
        wins.append(w)
        analytic.append(a)
        physics.append(a3.physics_bases(drive, w, a, time_unit))
        control_dt.append(float(np.load(path)["control_dt"]))
        print(f"{body}: {len(w.z)} windows", flush=True)

    probe = [(w.group % 1000) % a3.PROBES for w in wins]
    test = [p % 4 == 0 for p in probe]
    train_rows = {i: np.nonzero(~test[i])[0] for i in range(len(names))}
    pooled = np.concatenate([wins[i].z[~test[i]] for i in train_index])
    z_mean, z_std = pooled.mean(0), np.maximum(pooled.std(0), 1e-3)

    def as_t(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

    complete = {"pact_kd": a3.with_physics, "pact_kl": a3.with_loads}.get(args.encoder)
    uses_physics = args.encoder in ("pact_kd", "pact_kl")
    inputs = [complete(a, p) if complete else a for a, p in zip(analytic, physics, strict=True)]
    targets = [
        a3.residual_targets(w.z, a, p) if uses_physics else targets_of(w.z, a)
        for w, a, p in zip(wins, analytic, physics, strict=True)
    ]
    a_pool = np.concatenate([inputs[i][train_rows[i]] for i in train_index])
    a_mean, a_std = a_pool.mean(0), np.maximum(a_pool.std(0), 1e-3)
    t_pool = np.concatenate([targets[i][train_rows[i]] for i in train_index])
    t_mean, t_std = t_pool.mean(0), np.maximum(t_pool.std(0), 1e-3)
    table = stack_bodies(descriptors, contacts).to(device)
    count = table.mask.shape[1]
    dims = PactDims(observation=2, history=args.history, width=args.width, anchor_inputs=True)

    def batch(i: int, rows: np.ndarray):
        w = wins[i]
        obs = np.zeros((len(rows), count, args.history, 2), dtype=np.float32)
        obs[:, : w.joint_obs.shape[1]] = w.joint_obs[rows]
        return (
            as_t(obs),
            as_t(w.imu[rows]),
            as_t((inputs[i][rows] - a_mean) / a_std),
            as_t((targets[i][rows] - t_mean) / t_std),
        )

    rng = np.random.default_rng(args.seed)
    MODELS.mkdir(parents=True, exist_ok=True)
    path = (
        MODELS
        / f"n1_encoder_{args.encoder}_{args.hold_out}_s{args.seed}_p{args.pretrain_steps}_{data_key(args)}.pt"
    )
    torch.manual_seed(args.seed)
    pretrained = SharedAnchored(dims).to(device)
    if path.exists():
        pretrained.load_state_dict(torch.load(path, map_location=device))
        print(f"pretrained encoder loaded from {path}")
    else:  # exactly G-A3's shared-arm training
        optimiser = torch.optim.AdamW(pretrained.parameters(), lr=3e-4, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.pretrain_steps)
        per = max(1, args.pretrain_batch // len(train_index))
        start = time.time()
        for step in range(args.pretrain_steps):
            parts = [batch(i, rng.choice(train_rows[i], per)) for i in train_index]
            which = torch.cat(
                [torch.full((per,), i, device=device, dtype=torch.long) for i in train_index]
            )
            obs, imu, a_in, target = (torch.cat([p[k] for p in parts]) for k in range(4))
            loss = (pretrained(table.take(which), obs, imu, a_in) - target).square().mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(pretrained.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            if step % 2000 == 0 or step == args.pretrain_steps - 1:
                print(
                    f"pretrain step {step:5d} loss {loss.item():.5f} ({time.time() - start:.0f} s)"
                )
        torch.save(pretrained.state_dict(), path)
    state = {k: v.clone() for k, v in pretrained.state_dict().items()}

    test_rows = np.nonzero(test[h])[0]
    truth = (wins[h].z[test_rows] - z_mean) / z_std
    groups = wins[h].group[test_rows]
    own_joints = wins[h].joint_obs.shape[1]

    # The residual wrapper reproduces SharedAnchored's output exactly with zero residuals.
    check = EncoderResidual(pretrained, dims.body_tokens, count, dims.width).to(device).eval()
    pretrained.eval()
    with torch.no_grad():
        obs, imu, a_in, _ = batch(h, test_rows[:64])
        which = torch.full((len(obs),), h, device=device, dtype=torch.long)
        bodies = table.take(which)
        gap = check(bodies, obs, imu, a_in) - pretrained(bodies, obs, imu, a_in)
        if gap.abs().max().item() > 1e-5:
            raise RuntimeError(f"EncoderResidual differs from SharedAnchored by {gap.abs().max()}")

    def predict(model) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for rows in np.array_split(test_rows, max(1, len(test_rows) // 2048)):
                obs, imu, a_in, _ = batch(h, rows)
                which = torch.full((len(rows),), h, device=device, dtype=torch.long)
                learned = model(table.take(which), obs, imu, a_in) * as_t(t_std) + as_t(t_mean)
                if uses_physics:
                    learned = learned + as_t(a3.learned_base(physics[h][rows]))
                z = assemble(as_t(analytic[h][rows]), learned)
                out.append(((z - as_t(z_mean)) / as_t(z_std)).cpu().numpy())
        return np.concatenate(out) - truth

    def adapt(model, tuned, rows_k, arm) -> float:
        if not tuned:
            return 0.0
        optimiser = torch.optim.AdamW(tuned, lr=LR[arm], weight_decay=0.0)
        draw = np.random.default_rng(args.seed + 1)  # the same batches for every arm
        start = time.time()
        model.train()
        for _ in range(args.adapt_steps):
            rows = draw.choice(rows_k, args.batch)
            obs, imu, a_in, target = batch(h, rows)
            which = torch.full((len(rows),), h, device=device, dtype=torch.long)
            loss = (model(table.take(which), obs, imu, a_in) - target).square().mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(tuned, 1.0)
            optimiser.step()
        return time.time() - start

    units = np.unique(probe[h][train_rows[h]])
    budgets = nested_budgets(units, args.budgets.split(","), args.seed)
    report = {"part": "encoder", "encoder": args.encoder, "budgets": {}}
    for k, chosen in budgets.items():
        rows_k = train_rows[h][np.isin(probe[h][train_rows[h]], chosen)]
        entry = {
            "units": int(len(chosen)),
            "windows": int(len(rows_k)),
            "sim_seconds": float(len(rows_k) * control_dt[h]),
            "arms": {},
        }
        errors = {}
        for arm in args.arms.split(","):
            torch.manual_seed(args.seed)
            base = SharedAnchored(dims).to(device)
            if arm != "scratch":
                base.load_state_dict(state)
            model = EncoderResidual(base, dims.body_tokens, count, dims.width).to(device)
            n_params = counted(arm, model, own_joints)
            tuned = select(arm, model)
            seconds = adapt(model, tuned, rows_k, arm)
            errors[arm] = predict(model)
            entry["arms"][arm] = {
                "rmse": field_rmse(errors[arm]),
                "tuned_parameters": int(n_params),
                "adapt_seconds": round(seconds, 1),
            }
            print(f"k={k} {arm}: all {entry['arms'][arm]['rmse']['all']:.4f}", flush=True)
        entry["best"] = best = min(errors, key=lambda a: entry["arms"][a]["rmse"]["all"])
        for arm in errors:
            for name, ref in (("zero", "zero"), ("scratch", "scratch"), ("best", best)):
                if ref in errors and arm != ref:
                    _, _, diff, se = bootstrap_rmse(
                        (errors[arm] ** 2).mean(1),
                        (errors[ref] ** 2).mean(1),
                        groups,
                        args.bootstrap,
                        args.seed,
                    )
                    entry["arms"][arm][f"minus_{name}"] = diff
                    entry["arms"][arm][f"minus_{name}_se"] = se
        report["budgets"][k] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", choices=("encoder", "translator"), required=True)
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--hold-out", default="nao")
    parser.add_argument("--encoder", default="pact_kd", choices=("pact_k", "pact_kd", "pact_kl"))
    parser.add_argument("--budgets", default="1,4,16,all")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--adapt-steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=None, help="Adaptation batch.")
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--pretrain-batch", type=int, default=512)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--encoder-captures", default="logs/pact/coexec/{body}_map.npz")
    parser.add_argument("--translator-captures", default="logs/pact/coexec/{body}_replay.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch is None:
        args.batch = 512 if args.part == "encoder" else 256
    if args.pretrain_steps is None:
        args.pretrain_steps = 24000 if args.part == "encoder" else 12000
    output = args.output or (
        REPO / f"transfer/results/icml_gate_n1_{args.part}_{args.hold_out}.json"
    )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run = run_encoder if args.part == "encoder" else run_translator
    report = {"gate": "G-N1", "config": {k: str(v) for k, v in vars(args).items()}}
    report.update(run(args, device))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
