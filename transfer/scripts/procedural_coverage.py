"""Does the plan family cover a task? Gait signatures against the ranges, and windows against both.

Zero-shot transfer can only be expected for a task whose plans lie where the
tracker practised (``transfer/docs/skill_space.md`` §8). Two measurements,
neither of which trains anything:

**Coverage (H1, pre-registered).** Each capture's gait is summarised by the
parameters the family is drawn in -- forward and sideways speed, turn, stride
period, the two duties, step width, CoM height, foot clearance, lean -- and
checked against :class:`~humanoid_transfer.transfer.latent.procedural.GaitRanges`
and its feasibility constraints, with the shortest swing of the body that
would carry it out. A task outside them is outside the family by design.

**Style (diagnostic).** The family is written by a model; a task's plans are
written by its teacher. A tracker could learn to tell them apart and treat a
real plan as the walk's because it does not look procedural (shortcut
learning, Geirhos et al. 2020). Windows of ``z*`` as a tracker sees them --
the state, and the next 13 samples relative to it -- are taken from the walk
teacher and from random procedural schedules, and a logistic probe is fitted
to tell them apart (with every field, and with the fields the biped derives
only). Then, for another task's windows: how walk-like the probe finds them,
and whether each is nearer to a procedural window or to a walk window.

Usage::

    python transfer/scripts/procedural_coverage.py --walk g1/rollouts/closed_loop_teacher.npz \
        --task flat_forward=g1/rollouts/flat_play_forward.npz ... \
        --output transfer/results/procedural_coverage.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.procedural import (  # noqa: E402
    SWING_STEPS,
    GaitRanges,
    PointMassBiped,
    feasible,
    standing_skill,
)

SETTLE = 100
WINDOW = 13
STEP = 0.07742576129865368
"""The contract's grid, the G1's control step, in ``T``."""

BODY_STEPS = {"g1": STEP, "nao": 0.12579}
DERIVED = [
    i
    for name in ("height", "com_velocity", "foot_position", "foot_velocity", "foot_load")
    for i in range(SKILL_LAYOUT.slice_of(name).start, SKILL_LAYOUT.slice_of(name).stop)
]
"""The fields the biped derives from its dynamics (not the ones calibrated from walks)."""


def load(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.load(REPO_ROOT / path)
    dt_star = float(data["control_dt"]) / math.sqrt(float(data["leg_length"]) / 9.81)
    return data["skill_state"][SETTLE:], data["done"][SETTLE:], dt_star


def parameters(z: np.ndarray, done: np.ndarray, dt_star: float) -> dict:
    """The family's parameters, read off a capture."""
    f = SKILL_LAYOUT.unpack(z)
    signature = gait_signature(z, done, dt_star, threshold=0.01)
    feet = f["foot_position"]
    loads = f["foot_load"]
    in_air = loads < 0.01
    clearance = (f["height"][..., None] + feet[..., 2])[in_air]
    gravity_x = np.clip(f["gravity"][..., 0], -1.0, 1.0)
    return {
        "forward": float(f["com_velocity"][..., 0].mean()),
        "lateral": float(f["com_velocity"][..., 1].mean()),
        "turn": float(f["base_angular_velocity"][..., 2].mean()),
        "period": signature.stride_period,
        "duty": [float(v) for v in signature.duty_factor],
        "width": float(np.abs(feet[..., 0, 1] - feet[..., 1, 1]).mean()),
        "height": float(f["height"].mean()),
        "clearance": float(np.percentile(clearance, 90)) if clearance.size else 0.0,
        "lean": float(np.arcsin(gravity_x).mean()),
    }


def check(params: dict, ranges: GaitRanges, body_step: float) -> dict:
    inside = {}
    for key in ("forward", "lateral", "turn", "period", "width", "height", "clearance", "lean"):
        low, high = getattr(ranges, key)
        inside[key] = bool(low <= params[key] <= high)
    low, high = ranges.duty
    asymmetry = abs(params["duty"][0] - params["duty"][1])
    inside["duty"] = bool(all(low <= d <= high for d in params["duty"]))
    inside["duty_asymmetry"] = bool(asymmetry <= 2 * ranges.duty_asymmetry)
    t = torch.tensor
    min_swing = max(ranges.min_swing, SWING_STEPS * body_step)
    inside["feasible"] = bool(
        feasible(
            t([params["period"]]),
            t([params["duty"]]),
            t([params["forward"]]),
            t([params["lateral"]]),
            t([params["width"]]),
            ranges,
            min_swing,
        )[0]
    )
    return inside


def windows(z: np.ndarray, done: np.ndarray, stride: int = 3) -> np.ndarray:
    """``(M, 27 + 13 x 27)``: the state, then the next samples relative to it, within episodes."""
    rows = []
    steps = z.shape[0]
    for start in range(0, steps - WINDOW - 1, stride):
        span = done[start : start + WINDOW + 1].any(axis=0)
        now = z[start, ~span]
        ahead = z[start + 1 : start + WINDOW + 1, ~span] - now[None]
        rows.append(np.concatenate([now, ahead.transpose(1, 0, 2).reshape(len(now), -1)], axis=1))
    return np.concatenate(rows)


def procedural_windows(n: int, steps: int, device: str) -> np.ndarray:
    biped = PointMassBiped(n, GaitRanges(), body_step=STEP, substep=STEP / 4, horizon=1.0,
                           device=device, seed=13)  # fmt: skip
    biped.set_offsets(torch.zeros(1, dtype=torch.float64))
    ids = torch.arange(n, device=device)
    biped.anchor(ids, standing_skill(n, device))
    z = []
    for _ in range(steps):
        biped.advance()
        z.append(biped.plan(ids)[2].cpu().numpy())
    z = np.stack(z)[SETTLE:]
    done = np.repeat(biped.fallen.cpu().numpy()[None], z.shape[0], axis=0)
    return windows(z, done)


def probe(a: np.ndarray, b: np.ndarray, columns: np.ndarray, device: str) -> tuple:
    """Logistic probe, ``a`` (label 1) against ``b``: held-out AUC, and the fitted model."""
    rng = np.random.default_rng(0)
    x = np.concatenate([a, b])[:, columns]
    y = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    mean, std = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / std
    cut = int(0.8 * len(x))
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model = torch.nn.Linear(x.shape[1], 1).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    weight = torch.tensor(len(b) / len(a), device=device)
    for _ in range(400):
        optimiser.zero_grad()
        logits = model(xt[:cut]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yt[:cut], pos_weight=weight
        )
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        scores = model(xt[cut:]).squeeze(-1).cpu().numpy()
    truth = y[cut:]
    positive, negative = scores[truth == 1], scores[truth == 0]
    auc = float((positive[:, None] > negative[None, :]).mean())

    def walkness(c: np.ndarray) -> float:
        with torch.no_grad():
            u = torch.tensor((c[:, columns] - mean) / std, dtype=torch.float32, device=device)
            return float(torch.sigmoid(model(u)).mean())

    return auc, walkness, mean, std


def nearest_side(task: np.ndarray, walk: np.ndarray, family: np.ndarray, columns, mean, std,
                 device: str) -> float:  # fmt: skip
    """Share of a task's windows nearer to a procedural window than to any walk window."""

    def standard(x):
        return torch.tensor((x[:, columns] - mean) / std, dtype=torch.float32, device=device)

    t, w, f = standard(task), standard(walk), standard(family)
    nearer = []
    for chunk in torch.split(t, 512):
        to_walk = torch.cdist(chunk, w).amin(dim=1)
        to_family = torch.cdist(chunk, f).amin(dim=1)
        nearer.append((to_family < to_walk).float())
    return float(torch.cat(nearer).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[], help="name=path/to/capture.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ranges = GaitRanges()

    captures = {
        "walk": args.walk,
        **{k: Path(v) for k, _, v in (t.partition("=") for t in args.task)},
    }
    report: dict = {
        "ranges": {k: getattr(ranges, k) for k in ranges.__dataclass_fields__},
        "tasks": {},
    }
    real_windows = {}
    for name, path in captures.items():
        z, done, dt_star = load(path)
        params = parameters(z, done, dt_star)
        coverage = {body: check(params, ranges, step) for body, step in BODY_STEPS.items()}
        report["tasks"][name] = {
            "capture": str(path),
            "parameters": params,
            "inside": coverage,
            "covered": {body: all(v.values()) for body, v in coverage.items()},
        }
        real_windows[name] = windows(z, done)
        shown = {
            k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3))
            for k, v in params.items()
        }
        print(name, shown, "covered", report["tasks"][name]["covered"])

    family = procedural_windows(1024, 900, args.device)
    walk = real_windows["walk"]
    rng = np.random.default_rng(1)
    family = family[rng.permutation(len(family))[: 4 * len(walk)]]
    style = {}
    for label, columns in (
        ("all_fields", np.arange(family.shape[1])),
        (
            "derived_fields",
            np.concatenate([[i] + [27 * (k + 1) + i for k in range(WINDOW)] for i in DERIVED]),
        ),  # fmt: skip
    ):
        auc, walkness, mean, std = probe(walk, family, columns, args.device)
        entry = {"auc_walk_vs_family": auc, "walkness": {}, "nearer_family_than_walk": {}}
        for name, task in real_windows.items():
            if name == "walk":
                continue
            entry["walkness"][name] = walkness(task)
            entry["nearer_family_than_walk"][name] = nearest_side(
                task[rng.permutation(len(task))[:4000]], walk, family, columns, mean, std,
                args.device,
            )  # fmt: skip
        entry["walkness"]["family"] = walkness(family[:4000])
        style[label] = entry
        print(label, json.dumps(entry, indent=None))
    report["style"] = style
    report["windows"] = {"walk": len(walk), "family": len(family)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
