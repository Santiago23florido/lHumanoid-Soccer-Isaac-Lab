"""Gate G-X: masked cross-body prediction on co-executions.

Registered in ``docs/report/research_notes/icml/preregistration.md`` (made precise before
its data existed). Every body's tracker replayed the same plans
(``transfer/scripts/pact_coexecute.sh``); their ``z*`` are paired on one grid of instants
(``pact/coexec.py``). A target body's window is predicted from ONE other body's window:

- ``pact_x``: the translator (``pact/translate.py``) with both bodies' ``E_b``;
- ``copy`` (control i): the source's window itself, what a body-blind contract assumes;
- ``no_body`` (control ii): the same translator without any ``E_b``.

Metric: RMSE over the 27 fields, pooled-std units, on held-out chains (index % 5 == 0), per
target body, averaged over its sources; bootstrap over held-out chains. Pass: ``pact_x``
better than both controls by > 2 SE on every target body. Reported, not gated: the
held-out body (``--hold-out``, default the NAO), never a source or target in training.

Usage::

    python transfer/scripts/pact_gate_x.py --bodies g1,nao,h1,cassie
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

import torch  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.pact.coexec import pair, windows  # noqa: E402
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.translate import CrossBodyTranslator  # noqa: E402

FIELDS = [f.name for f in SKILL_LAYOUT.fields]


def body_dt(capture) -> float:
    return float(capture["control_dt"]) / math.sqrt(float(capture["leg_length"]) / GRAVITY)


def train(model, data, rows, bodies, table, args, device, tag: str):
    """Masked prediction: a random window, a random ordered pair of distinct ``bodies``."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bodies_t = torch.as_tensor(bodies, device=device)
    start = time.time()
    for step in range(args.steps):
        pick = rows[torch.randint(len(rows), (args.batch,), device=device)]
        first = torch.randint(len(bodies), (args.batch,), device=device)
        other = torch.randint(len(bodies) - 1, (args.batch,), device=device)
        other = other + (other >= first).long()  # a different body, uniformly
        s, t = bodies_t[first], bodies_t[other]
        source, target = gather(data, pick, s, args.window), gather(data, pick, t, args.window)
        tokens = model.bodies(table)
        loss = (model(source, s, t, tokens) - target).pow(2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 2000 == 0 or step == args.steps - 1:
            print(f"{tag} step {step:5d} loss {loss.item():.5f} ({time.time() - start:.0f} s)")
    return model


def gather(data: torch.Tensor, pick: torch.Tensor, body: torch.Tensor, window: int):
    """``data (B, K, G, skill)``, ``pick (N, 2)`` (chain row, first instant), ``body (N,)``."""
    offsets = torch.arange(window, device=data.device)
    return data[body[:, None], pick[:, :1], pick[:, 1:] + offsets[None]]


@torch.no_grad()
def errors(model, data, rows, pairs, table, window: int, batch: int = 1024) -> np.ndarray:
    """``(len(rows), len(pairs), skill)`` mean squared error per window, pair and component;
    ``model`` None copies the source."""
    out = np.zeros((len(rows), len(pairs), data.shape[-1]))
    tokens = None if model is None else model.bodies(table)
    for p, (s, t) in enumerate(pairs):
        for i in range(0, len(rows), batch):
            pick = rows[i : i + batch]
            s_t = torch.full((len(pick),), s, device=data.device)
            t_t = torch.full((len(pick),), t, device=data.device)
            source, target = gather(data, pick, s_t, window), gather(data, pick, t_t, window)
            predicted = source if model is None else model(source, s_t, t_t, tokens)
            out[i : i + batch, p] = (predicted - target).pow(2).mean(dim=1).cpu().numpy()
    return out


def target_values(err: np.ndarray, pairs, target: int) -> np.ndarray:
    """``(windows,)``: each held-out window's mean squared error for ``target``, over its
    sources and the 27 fields."""
    columns = [p for p, (_, t) in enumerate(pairs) if t == target]
    return err[:, columns].mean(axis=(1, 2))


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(values.mean()))


def bootstrap_rmse(values: np.ndarray, chain_rows: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``counts (draws, chains)``: how often each held-out chain is drawn; the RMSE of each
    draw over the windows of the chains drawn."""
    chains = counts.shape[1]
    total = np.bincount(chain_rows, weights=values, minlength=chains)
    number = np.bincount(chain_rows, minlength=chains)
    return np.sqrt((counts @ total) / np.maximum(counts @ number, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default="g1,nao,h1,cassie")
    parser.add_argument("--captures", default="logs/pact/coexec/{body}_replay.npz")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--hold-out", default="nao", help="Reported, not gated; '' to skip.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_gate_x.json")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    names = [b for b in args.bodies.split(",") if b]
    registry = json.loads((REPO / "transfer/bodies/bodies.json").read_text("utf-8"))

    captures = [np.load(REPO / args.captures.format(body=b)) for b in names]
    step = max(body_dt(c) for c in captures)
    paired = pair(captures, step)
    test_chain = paired.chains % 5 == 0
    print(
        f"grid step {step:.4f}; chains {len(paired.chains)} ({test_chain.sum()} held out); "
        f"paired instants {int(paired.valid.sum())}"
    )
    train_valid = paired.valid & ~test_chain[:, None]
    flat = paired.z[:, train_valid]  # (B, M, skill)
    mean, std = flat.reshape(-1, flat.shape[-1]).mean(0), flat.reshape(-1, flat.shape[-1]).std(0)
    std = np.maximum(std, 1e-3)
    data = torch.as_tensor((paired.z - mean) / std, dtype=torch.float32, device=device)

    train_rows = windows(paired.valid & ~test_chain[:, None], args.window, args.stride)
    test_rows = windows(paired.valid & test_chain[:, None], args.window, args.stride)
    print(f"windows: train {len(train_rows)}, test {len(test_rows)}")
    if len(test_rows) == 0 or len(train_rows) == 0:
        raise SystemExit("no paired windows")
    train_t = torch.as_tensor(train_rows, device=device)
    test_t = torch.as_tensor(test_rows, device=device)

    descriptors, contacts = [], []
    for body in names:
        entry = registry[body]
        card = json.loads((REPO / entry["card"]).read_text("utf-8"))
        descriptors.append(describe(card, leg_length=entry.get("leg_length_unit")))
        contacts.append(contact_joints(card))
    table = stack_bodies(descriptors, contacts).to(device)
    dims = PactDims(width=args.width)

    every = list(range(len(names)))
    pairs = [(s, t) for t in every for s in every if s != t]
    models = {
        "pact_x": CrossBodyTranslator(dims, args.window).to(device),
        "no_body": CrossBodyTranslator(dims, args.window, use_body=False).to(device),
    }
    for tag, model in models.items():
        train(model, data, train_t, every, table, args, device, tag).eval()
    errs = {"copy": errors(None, data, test_t, pairs, table, args.window)}
    for tag, model in models.items():
        errs[tag] = errors(model, data, test_t, pairs, table, args.window)

    _, chain_rows = np.unique(test_rows[:, 0], return_inverse=True)
    held = int(chain_rows.max()) + 1
    draws = rng.integers(0, held, size=(args.bootstrap, held))
    counts = np.stack([np.bincount(d, minlength=held) for d in draws]).astype(np.float64)
    report = {
        "gate": "G-X",
        "config": {k: str(v) for k, v in vars(args).items()},
        "grid_step": step,
        "chains": int(len(paired.chains)),
        "held_out_chains": int(test_chain.sum()),
        "windows": {"train": int(len(train_rows)), "test": int(len(test_rows))},
        "parameters": {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
        "bodies": {},
    }
    passes = True
    for b, body in enumerate(names):
        values = {arm: target_values(e, pairs, b) for arm, e in errs.items()}
        row = {arm: rmse(v) for arm, v in values.items()}
        boot = {arm: bootstrap_rmse(v, chain_rows, counts) for arm, v in values.items()}
        for control in ("copy", "no_body"):
            row[f"pact_x_minus_{control}"] = row["pact_x"] - row[control]
            row[f"pact_x_minus_{control}_se"] = float(np.std(boot["pact_x"] - boot[control]))
        row["passes"] = bool(
            row["pact_x_minus_copy"] < -2 * row["pact_x_minus_copy_se"]
            and row["pact_x_minus_no_body"] < -2 * row["pact_x_minus_no_body_se"]
        )
        passes &= row["passes"]
        columns = [p for p, (_, t) in enumerate(pairs) if t == b]
        row["fields"] = {
            arm: {
                f: float(np.sqrt(e[:, columns][..., SKILL_LAYOUT.slice_of(f)].mean()))
                for f in FIELDS
            }
            for arm, e in errs.items()
        }
        report["bodies"][body] = row
        print(body, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})

    if args.hold_out and args.hold_out in names:
        h = names.index(args.hold_out)
        seen = [b for b in every if b != h]
        model = CrossBodyTranslator(dims, args.window).to(device)
        # The windows are shared instants; the held-out body is never drawn as source or target.
        train(model, data, train_t, seen, table, args, device, "hold")
        model.eval()
        held_pairs = [(s, h) for s in seen]
        e_model = errors(model, data, test_t, held_pairs, table, args.window)
        e_copy = errors(None, data, test_t, held_pairs, table, args.window)
        v_model, v_copy = (
            target_values(e_model, held_pairs, h),
            target_values(e_copy, held_pairs, h),
        )
        b_model = bootstrap_rmse(v_model, chain_rows, counts)
        b_copy = bootstrap_rmse(v_copy, chain_rows, counts)
        report["held_out_body"] = {
            "body": args.hold_out,
            "pact_x": rmse(v_model),
            "copy": rmse(v_copy),
            "pact_x_minus_copy": rmse(v_model) - rmse(v_copy),
            "pact_x_minus_copy_se": float(np.std(b_model - b_copy)),
            "pact_x_trained_with_it": report["bodies"][args.hold_out]["pact_x"],
        }
        print("held out", report["held_out_body"])
    report["passes"] = passes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"G-X {'PASSES' if passes else 'FAILS'}; written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
