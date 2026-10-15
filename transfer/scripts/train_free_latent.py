"""Learn the free latent part from both bodies, and test what it carries across them.

F5 of ``transfer/docs/skill_space.md`` §9 (:mod:`~humanoid_transfer.transfer.latent.free_latent`).
The data are captures of each body's tracker following the same fixed gaits of
the procedural family (``capture_contract.py --gait-probes``), which record each
foot's measured balance and the biped's gait phase. Two states are a pair when
they follow the same probe at the same phase (one of ``--phase-bins``): the
same planned gait at the same moment, each dynamically valid on its own body.

**Round 2 (the user's choice after round 1).** Round 1 read both bodies' balance
with one predictor; paired states are held up differently on the two bodies,
so it could not pass (§9.1). Here ``ζ`` stays shared, aligned and blind to the
body -- what a plan carries from one body to another -- while each body reads
its physics from ``(z*, ζ)`` with its own head, trained on its own procedural
data (the body side; no task's data). Each foot's centre of pressure is
measured as a fraction of its own support polygon. Pre-registered:

1. **Shared.** Paired states' ``ζ`` agree, and a classifier tells the bodies apart
   on held-out pairs no better than chance, within twice its standard error.
2. **Informative.** On each body's held-out probes, its head with ``ζ`` predicts
   its balance better than its head with ``ζ`` set to zero, by more than twice the
   bootstrap standard error over probes.
3. **Carried.** On held-out pairs, the NAO's head reading the NAO's ``z*`` and the
   *G1's* ``ζ`` -- what a plan written from the G1 would hand the NAO -- predicts the
   NAO's balance better than with ``ζ`` set to zero, by the same margin.

Usage::

    python transfer/scripts/train_free_latent.py \
        --g1 g1/rollouts/zs_r3_6194_sym_f5_map.npz --g1 g1/rollouts/zs_r3_6194_sym_f5_corners.npz \
        --nao nao/rollouts/zs_r1_3498_sym_f5_map.npz \
        --nao nao/rollouts/zs_r1_3498_sym_f5_corners.npz \
        --output logs/free_latent/f5_r2/model.pt --results transfer/results/free_latent_f5_r2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.similarity import TIME, UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.free_latent import (  # noqa: E402
    BodyClassifier,
    BodyStems,
    DualAscent,
    FreeLatentDims,
    FreeLatentEncoder,
    PhysicsPredictor,
    accuracy_excess,
    alignment_loss,
    entity_features,
    gaussian_kl,
    prediction_loss,
    reverse_gradient,
)

SLICES = {field.name: SKILL_LAYOUT.slice_of(field.name) for field in SKILL_LAYOUT.fields}
LEGS = {
    "g1": ("left_hip", "left_knee", "left_ankle"),
    "nao": ("LHip", "LKnee", "LAnkle"),
}
"""Name prefixes of each body's left-leg joints; the right leg's mirror them."""

POLYGONS = {
    "g1": {
        "x": (-0.101555, 0.101555),
        "left": (-0.032735, 0.032735),
        "right": (-0.032735, 0.032735),
    },
    "nao": {"x": (-0.0607, 0.1033), "left": (-0.0397, 0.0526), "right": (-0.0526, 0.0397)},
}
"""Each foot's support polygon about its sole point, metres: the G1's collision box underside
(``g1/body/body_card.json``), the NAO's measured foot (``nao_kinematics.SUPPORT_POLYGON_*``)."""


def polygon_fraction(balance: np.ndarray, body: str, length: float) -> np.ndarray:
    """The centre of pressure, from leg lengths about the sole point to a fraction of the foot.

    -1 and +1 are the polygon's edges along each axis; the forces are left as they are.
    """
    out = balance.copy()
    polygon = POLYGONS[body]
    for i, side in enumerate(("left", "right")):
        for axis, bounds in ((0, polygon["x"]), (1, polygon[side])):
            centre, half = (bounds[0] + bounds[1]) / 2, (bounds[1] - bounds[0]) / 2
            column = 2 * i + axis
            out[..., column] = (balance[..., column] * length - centre) / half
    return out


@dataclass
class BodyData:
    """One body's steps, on the device: features by entity, ``z*``, targets, pairing keys."""

    features: dict[str, torch.Tensor]
    skill: torch.Tensor
    balance: torch.Tensor
    falls: torch.Tensor
    probe: torch.Tensor
    phase_bin: torch.Tensor


def _legs(joint_names: list[str], body: str) -> tuple[list[int], list[int]]:
    left_prefixes = LEGS[body]
    right_prefixes = tuple(
        p.replace("left", "right") if body == "g1" else "R" + p[1:] for p in left_prefixes
    )
    left = [i for i, n in enumerate(joint_names) if n.startswith(left_prefixes)]
    right = [i for i, n in enumerate(joint_names) if n.startswith(right_prefixes)]
    if len(left) == 0 or len(right) == 0:
        raise ValueError(f"no leg joints found for {body} in {joint_names}")
    return left, right


def load_body(paths: list[Path], body: str, bins: int, norm, device: str) -> BodyData:
    """Every step that follows a step of the same episode, from every capture of one body."""
    parts: dict[str, list] = {k: [] for k in ("x", "z", "b", "f", "p", "k")}
    joint_legs = action_legs = None
    joints = 0
    probe_offset = 0
    for path in paths:
        data = np.load(path)
        names = [str(n) for n in data["joint_names"]]
        joints = len(names)
        actions = data["action_executed"].shape[-1]
        joint_legs = _legs(names, body)
        # The actions drive the actuated joints in their own order; on both bodies the legs'
        # joints are named the same in the action, so the same prefixes find them there.
        action_names = names if actions == joints else _actuated_names(body, names)
        action_legs = _legs(action_names, body)
        done = data["done"]
        fresh = np.zeros_like(done)
        fresh[1:] = done[:-1]
        keep = ~fresh
        units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
        dt_star = float(units.to_dimensionless(float(data["control_dt"]), TIME))
        horizon = max(1, round(1.0 / dt_star))
        terminated = data["terminated"]
        soon = np.zeros_like(terminated)
        for k in range(horizon):
            soon[: len(soon) - k] |= terminated[k:]
        probe = np.broadcast_to(data["probe_index"][None, :], done.shape) + probe_offset
        probe_offset += int(data["probe_index"].max()) + 1
        phase = np.clip((data["biped_phase"] * bins).astype(np.int64), 0, bins - 1)
        parts["x"].append(data["encoder_input"][keep])
        parts["z"].append(data["skill_state"][keep])
        parts["b"].append(polygon_fraction(data["balance"], body, float(data["leg_length"]))[keep])
        parts["f"].append(soon[keep])
        parts["p"].append(probe[keep])
        parts["k"].append(phase[keep])

    def cat(key, dtype):
        return torch.as_tensor(np.concatenate(parts[key]), dtype=dtype, device=device)

    encoder_input, skill = cat("x", torch.float32), cat("z", torch.float32)
    features = entity_features(encoder_input, skill, joint_legs, action_legs, joints, SLICES)
    return BodyData(
        features=features,
        skill=norm(skill),
        balance=cat("b", torch.float32),
        falls=cat("f", torch.bool),
        probe=cat("p", torch.long),
        phase_bin=cat("k", torch.long),
    )


def _actuated_names(body: str, names: list[str]) -> list[str]:
    if body != "nao":
        return names
    from humanoid_transfer.nao.assets import nao_kinematics as nk

    return list(nk.ACTUATED_JOINTS)


def pair_index(a: BodyData, b: BodyData, bins: int) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """``{cell: (steps of a, steps of b)}`` for every (probe, phase bin) both bodies visited."""
    cells = {}
    key_a, key_b = a.probe * bins + a.phase_bin, b.probe * bins + b.phase_bin
    for cell in torch.unique(key_a).tolist():
        rows_b = (key_b == cell).nonzero().squeeze(-1)
        if len(rows_b) == 0:
            continue
        cells[cell] = ((key_a == cell).nonzero().squeeze(-1), rows_b)
    return cells


def sample_pairs(cells, count: int, generator: torch.Generator, device: str):
    keys = list(cells)
    chosen = torch.randint(len(keys), (count,), generator=generator)
    left, right = [], []
    for c in chosen.tolist():
        rows_a, rows_b = cells[keys[c]]
        left.append(rows_a[torch.randint(len(rows_a), (1,), generator=generator)])
        right.append(rows_b[torch.randint(len(rows_b), (1,), generator=generator)])
    return torch.cat(left).to(device), torch.cat(right).to(device)


def take(data: BodyData, rows: torch.Tensor) -> dict:
    return {
        "features": {k: v[rows] for k, v in data.features.items()},
        "skill": data.skill[rows],
        "balance": data.balance[rows],
        "falls": data.falls[rows],
    }


def held_out_probes(data: BodyData, share: float, seed: int) -> torch.Tensor:
    probes = torch.unique(data.probe)
    generator = torch.Generator().manual_seed(seed)
    order = probes[torch.randperm(len(probes), generator=generator)]
    return order[: max(1, round(share * len(order)))]


def per_probe_error(predictor, skill, free, balance, falls, probe) -> torch.Tensor:
    """``(probes, 2)``: each probe's balance mean squared error and fall cross-entropy."""
    with torch.no_grad():
        predicted, logit = predictor(skill, free)
        squared = (predicted - balance).pow(2).mean(dim=-1)
        fall = F.binary_cross_entropy_with_logits(logit, falls.float(), reduction="none")
    rows = []
    for p in torch.unique(probe):
        mask = probe == p
        rows.append(torch.stack([squared[mask].mean(), fall[mask].mean()]))
    return torch.stack(rows)


def encode_all(encoder, stem, features: dict, chunk: int = 8192) -> torch.Tensor:
    """``ζ``'s posterior means for every row, a chunk at a time (a fused attention kernel
    rejects a launch over hundreds of thousands of rows)."""
    count = next(iter(features.values())).shape[0]
    means = []
    with torch.no_grad():
        for start in range(0, count, chunk):
            part = {k: v[start : start + chunk] for k, v in features.items()}
            means.append(encoder(stem(part))[0])
    return torch.cat(means)


def verdict(predictor, baseline, data: dict, free: torch.Tensor, probes, seed: int) -> dict:
    """Per held-out probe: the predictor with ``ζ`` against the ``z*``-alone baseline."""
    with_free = (
        per_probe_error(predictor, data["skill"], free, data["balance"], data["falls"], probes)
        .cpu()
        .numpy()
    )
    alone = (
        per_probe_error(
            baseline, data["skill"], torch.zeros_like(free), data["balance"], data["falls"], probes
        )
        .cpu()
        .numpy()
    )
    result = {}
    for i, name in enumerate(("balance_mse", "fall_cross_entropy")):
        diff = with_free[:, i] - alone[:, i]
        se = bootstrap_se(diff, 2000, seed)
        result[name] = {
            "z_star_alone": float(alone[:, i].mean()),
            "with_free": float(with_free[:, i].mean()),
            "difference": float(diff.mean()),
            "se": se,
            "better": bool(diff.mean() < -2 * se),
        }
    return result


def bootstrap_se(diff: np.ndarray, draws: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diff), size=(draws, len(diff)))
    return float(diff[picks].mean(axis=1).std())


def standardise(data: BodyData, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each body's balance in units of its own spread over its training steps."""
    mean = data.balance[rows].mean(dim=0)
    std = data.balance[rows].std(dim=0).clamp(min=1e-6)
    data.balance = (data.balance - mean) / std
    return mean, std


def paired_rows(cells: dict, generator: torch.Generator, device: str):
    """One pair per cell: the held-out pairs the alignment and carry checks are scored on."""
    left, right, cell_ids = [], [], []
    for cell, (rows_a, rows_b) in cells.items():
        count = min(len(rows_a), len(rows_b), 8)
        left.append(rows_a[torch.randperm(len(rows_a), generator=generator)[:count]])
        right.append(rows_b[torch.randperm(len(rows_b), generator=generator)[:count]])
        cell_ids.append(torch.full((count,), cell))
    return torch.cat(left).to(device), torch.cat(right).to(device), torch.cat(cell_ids).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--g1", type=Path, action="append", required=True)
    parser.add_argument("--nao", type=Path, action="append", required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "logs/contract/walk_dagger3e/generator.pt",
        help="Whose normalisation of z* to read it in.",
    )
    parser.add_argument("--phase-bins", type=int, default=20)
    parser.add_argument("--held-out", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--kl-budget", type=float, default=4.0, help="Nats per state.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from humanoid_transfer.transfer.latent.contract import load_contract

    torch.manual_seed(args.seed)
    device = args.device
    norm = load_contract(args.contract, device).generator.skill_norm
    bodies = {
        "g1": load_body(args.g1, "g1", args.phase_bins, norm, device),
        "nao": load_body(args.nao, "nao", args.phase_bins, norm, device),
    }
    g1, nao = bodies["g1"], bodies["nao"]
    held = held_out_probes(g1, args.held_out, args.seed).to(device)
    train = {k: (~torch.isin(d.probe, held)).nonzero().squeeze(-1) for k, d in bodies.items()}
    test = {k: torch.isin(d.probe, held).nonzero().squeeze(-1) for k, d in bodies.items()}
    stats = {k: standardise(d, train[k]) for k, d in bodies.items()}
    all_cells = pair_index(g1, nao, args.phase_bins)
    held_cell = {
        c: bool(torch.isin(torch.tensor(c // args.phase_bins), held.cpu())) for c in all_cells
    }
    cells = {c: v for c, v in all_cells.items() if not held_cell[c]}
    test_cells = {c: v for c, v in all_cells.items() if held_cell[c]}
    print(
        f"g1 {len(g1.skill)} steps, nao {len(nao.skill)}, paired cells {len(cells)} "
        f"(+{len(test_cells)} held out), held-out probes {len(held)}"
    )

    dims = FreeLatentDims(balance=g1.balance.shape[-1])
    stems = torch.nn.ModuleDict(
        {
            name: BodyStems({k: v.shape[-1] for k, v in data.features.items()}, dims.width)
            for name, data in bodies.items()
        }
    ).to(device)
    encoder = FreeLatentEncoder(dims).to(device)
    heads = torch.nn.ModuleDict({name: PhysicsPredictor(dims) for name in bodies}).to(device)
    baselines = torch.nn.ModuleDict({name: PhysicsPredictor(dims) for name in bodies}).to(device)
    classifier = BodyClassifier(dims).to(device)
    optimiser = torch.optim.Adam(
        [*stems.parameters(), *encoder.parameters(), *heads.parameters()], lr=args.learning_rate
    )
    side = torch.optim.Adam(
        [*baselines.parameters(), *classifier.parameters()], lr=args.learning_rate
    )
    kl_multiplier, body_multiplier = DualAscent(rate=0.01, value=0.1), DualAscent(rate=0.05)
    generator = torch.Generator().manual_seed(args.seed)
    log = []
    for step in range(args.steps):
        total = torch.zeros((), device=device)
        base_total = torch.zeros((), device=device)
        row = {"step": step}
        for name, data in bodies.items():
            pick = torch.randint(len(train[name]), (args.batch,), generator=generator)
            batch = take(data, train[name][pick.to(device)])
            mean, log_variance = encoder(stems[name](batch["features"]))
            free = mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
            losses = prediction_loss(
                heads[name], batch["skill"], free, batch["balance"], batch["falls"]
            )
            kl = gaussian_kl(mean, log_variance).mean()
            total = total + losses["balance"] + losses["falls"] + kl_multiplier.value * kl
            base = prediction_loss(
                baselines[name], batch["skill"], torch.zeros_like(free), batch["balance"],
                batch["falls"],
            )  # fmt: skip
            base_total = base_total + base["balance"] + base["falls"]
            row[f"{name}_balance"] = float(losses["balance"])
            row[f"{name}_kl"] = float(kl)
        pair_a, pair_b = sample_pairs(cells, args.batch, generator, device)
        free_a, _ = encoder(stems["g1"](take(g1, pair_a)["features"]))
        free_b, _ = encoder(stems["nao"](take(nao, pair_b)["features"]))
        align = alignment_loss(free_a, free_b)
        logits = classifier(reverse_gradient(torch.cat([free_a, free_b]), body_multiplier.value))
        labels = torch.cat([torch.ones(len(free_a)), torch.zeros(len(free_b))]).to(device)
        body_loss = F.binary_cross_entropy_with_logits(logits, labels)
        total = total + align + body_loss
        optimiser.zero_grad(set_to_none=True)
        side.zero_grad(set_to_none=True)
        (total + base_total).backward()
        optimiser.step()
        side.step()
        kl_multiplier.step((row["g1_kl"] + row["nao_kl"]) / 2 - args.kl_budget)
        excess, se = accuracy_excess(logits.detach(), labels)
        body_multiplier.step(excess - 2 * se)
        if step % 500 == 0 or step == args.steps - 1:
            row.update({"align": float(align), "body_excess": excess,
                        "kl_multiplier": kl_multiplier.value,
                        "body_multiplier": body_multiplier.value})  # fmt: skip
            log.append(row)
            print(" ".join(f"{k} {v:.4g}" for k, v in row.items()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dims": asdict(dims),
            "stems": stems.state_dict(),
            "encoder": encoder.state_dict(),
            "heads": heads.state_dict(),
            "baselines": baselines.state_dict(),
            "balance": {k: [v.cpu() for v in s] for k, s in stats.items()},
            "polygons": POLYGONS,
        },
        args.output,
    )
    encoder.eval()
    report: dict = {
        "g1": [str(p) for p in args.g1],
        "nao": [str(p) for p in args.nao],
        "dims": asdict(dims),
        "held_out_probes": held.cpu().tolist(),
        "log": log,
    }
    # 2. Informative, on each body's held-out probes.
    informative = {}
    for name, data in bodies.items():
        part = take(data, test[name])
        free = encode_all(encoder, stems[name], part["features"])
        informative[name] = verdict(
            heads[name], baselines[name], part, free, data.probe[test[name]], args.seed
        )
    report["informative"] = informative
    # 1. Shared, and 3. carried, on held-out pairs.
    rows_a, rows_b, cell_ids = paired_rows(test_cells, generator, device)
    pair_a, pair_b = take(g1, rows_a), take(nao, rows_b)
    free_a = encode_all(encoder, stems["g1"], pair_a["features"])
    free_b = encode_all(encoder, stems["nao"], pair_b["features"])
    with torch.no_grad():
        logits = classifier(torch.cat([free_a, free_b]))
    labels = torch.cat([torch.ones(len(free_a)), torch.zeros(len(free_b))]).to(device)
    excess, se = accuracy_excess(logits, labels)
    report["shared"] = {
        "alignment": float(alignment_loss(free_a, free_b)),
        "classifier_excess": excess,
        "classifier_se": se,
        "body_blind": bool(excess <= 2 * se),
        "pairs": len(rows_a),
    }
    probes = cell_ids // args.phase_bins
    carried = verdict(heads["nao"], baselines["nao"], pair_b, free_a, probes, args.seed)
    own = verdict(heads["nao"], baselines["nao"], pair_b, free_b, probes, args.seed)
    report["carried"] = {"g1_zeta": carried, "nao_zeta": own}
    report["passes"] = {
        "shared": report["shared"]["body_blind"],
        "informative": all(v["balance_mse"]["better"] for v in informative.values()),
        "carried": carried["balance_mse"]["better"],
    }
    report["fall_rate"] = {k: float(d.falls.float().mean()) for k, d in bodies.items()}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, v in informative.items():
        b = v["balance_mse"]
        print(
            f"{name} held-out: z* alone {b['z_star_alone']:.4f}, with zeta {b['with_free']:.4f} "
            f"({b['difference']:+.4f} ± {b['se']:.4f})"
        )
    for name, v in report["carried"].items():
        b = v["balance_mse"]
        print(
            f"NAO head on held-out pairs, {name}: z* alone {b['z_star_alone']:.4f}, "
            f"with {b['with_free']:.4f} ({b['difference']:+.4f} ± {b['se']:.4f})"
        )
    print(f"classifier on held-out pairs: {excess:+.3f} ± {se:.3f}; passes {report['passes']}")
    return 0 if math.isfinite(log[-1]["g1_balance"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
