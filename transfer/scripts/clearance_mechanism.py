"""Why a tracker lifts its feet short of its plans: three checks on probe captures.

Read with ``sole_offset.py``. Per capture of the family's
probes (grid and corners), on the samples between valid steps, plan ``contract_due[t]``
against the state at ``t + 1`` (``common.contact.capture_fidelity``'s alignment):

* **swing time** -- per planned swing, the body's samples in the air (the swing's window
  widened by two samples each way) over the planned swing's length, and the clearance
  ratio of the swings the body spent fully in the air;
* **the foot-position term by axis** -- the tracking reward's recorded per-field error
  (``contract_fields``) is a weighted sum of the components' squared errors, weights the
  tolerances; fitted per plan age by least squares (R² reported), it splits into each
  foot's x, y and z, with each component's median error in its own tolerances;
* **what a higher lift costs** -- within each probe, the Spearman correlation of a swing's
  clearance ratio with every field's mean error over that swing, averaged over probes.

    python transfer/scripts/clearance_mechanism.py \
        --capture nao/rollouts/zs_r1_3498_sym_f5_map.npz \
        --capture nao/rollouts/zs_r1_3498_sym_f5_corners.npz \
        --capture g1/rollouts/zs_warm_r3_logload_sym_map.npz \
        --capture g1/rollouts/zs_warm_r3_logload_sym_corners.npz \
        --output transfer/results/clearance_mechanism.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import foot_height  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)

FIELDS = [field.name for field in SKILL_LAYOUT.fields]
AXES = ["left_x", "left_y", "left_z", "right_x", "right_y", "right_z"]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def swings(z, due, fields, valid, probe_of_env):
    """One row per planned swing: probe, asked length, body air samples, clearances, errors."""
    hb, ha = foot_height(z), foot_height(due)
    cb = SKILL_LAYOUT.unpack(z)["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    ca = SKILL_LAYOUT.unpack(due)["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    steps, envs = valid.shape
    rows = []
    for e in range(envs):
        v = valid[:, e]
        for s in range(2):
            stance_a, stance_b = ca[:, e, s] & v, cb[:, e, s] & v
            if not stance_a.any() or not stance_b.any():
                continue
            level_a = np.median(ha[:, e, s][stance_a])
            level_b = np.median(hb[:, e, s][stance_b])
            for a, b in _runs(~ca[:, e, s] & v):
                if a < 5 or b > steps - 5 or b - a < 2:
                    continue
                window = slice(a - 2, b + 2)
                asked = ha[a:b, e, s].max() - level_a
                reached = hb[window, e, s].max() - level_b
                rows.append(
                    {
                        "probe": int(probe_of_env[e]),
                        "env": e,
                        "side": s,
                        "start": a,
                        "length": b - a,
                        "air": int((~cb[window, e, s]).sum()),
                        "asked": float(asked),
                        "reached": float(reached),
                        "ratio": float(reached / max(asked, 1e-3)),
                        "errors": fields[a:b, e, : len(FIELDS)].mean(axis=0),
                    }
                )
    return rows


def axis_split(z, due, fields, age, valid) -> dict:
    span = SKILL_LAYOUT.slice_of("foot_position")
    column = FIELDS.index("foot_position")
    squared = (z[..., span] - due[..., span]) ** 2
    shares, tolerances_err, r2 = [], [], []
    for a in np.unique(age[valid]):
        m = valid & (age == a)
        if m.sum() < 500:
            continue
        x, y = squared[m], fields[..., column][m]
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        r2.append(1.0 - np.sum((y - x @ w) ** 2) / np.sum((y - y.mean()) ** 2))
        part = x * w
        shares.append(part.sum(0) / part.sum())
        # The field averages its six components, so a component's own tolerance error is
        # sqrt(6 w_c e_c^2).
        tolerances_err.append(np.median(np.sqrt(np.clip(6 * part, 0, None)), axis=0))
    return {
        "fit_r2_min": float(min(r2)),
        "share": dict(zip(AXES, np.mean(shares, 0).round(4).tolist(), strict=True)),
        "median_error_in_tolerances": dict(
            zip(AXES, np.mean(tolerances_err, 0).round(3).tolist(), strict=True)
        ),
    }


def capture_report(path: Path) -> dict:
    data = np.load(path)
    probes = json.loads(str(data["probes"]))
    keep = np.isin(
        data["probe_index"],
        [i for i, p in enumerate(probes) if p.get("group") in ("grid", "corner")],
    )
    z = data["skill_state"][1:, keep].astype(float)
    due = data["contract_due"][:-1, keep].astype(float)
    fields = data["contract_fields"][:-1, keep].astype(float)
    age = data["contract_age"][:-1, keep]
    done = data["done"][:, keep].astype(bool)
    valid = ~(done[:-1] | done[1:])
    valid[:20] = False
    rows = swings(z, due, fields, valid, data["probe_index"][keep])
    air = np.array([r["air"] / r["length"] for r in rows])
    ratio = np.array([r["ratio"] for r in rows])
    full = air >= 1.0
    correlation = {name: [] for name in FIELDS}
    for probe in sorted({r["probe"] for r in rows}):
        members = [r for r in rows if r["probe"] == probe]
        if len(members) < 30:
            continue
        lift = [r["ratio"] for r in members]
        for k, name in enumerate(FIELDS):
            c = spearmanr(lift, [r["errors"][k] for r in members]).correlation
            if np.isfinite(c):
                correlation[name].append(float(c))
    return {
        "swings": len(rows),
        "air_over_asked_median": float(np.median(air)),
        "clearance_ratio_median": float(np.median(ratio)),
        "fully_airborne_share": float(full.mean()),
        "fully_airborne_clearance_ratio_median": float(np.median(ratio[full])),
        "foot_position_term": axis_split(z, due, fields, age, valid),
        "median_field_error": {
            name: float(np.median(fields[..., k][valid])) for k, name in enumerate(FIELDS)
        },
        "lift_vs_field_error_spearman": {
            name: float(np.mean(v)) for name, v in correlation.items() if v
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for path in args.capture:
        report[path.as_posix()] = r = capture_report(path)
        split = r["foot_position_term"]
        print(path.name)
        print(
            f"  air/asked {r['air_over_asked_median']:.2f}; clearance ratio "
            f"{r['clearance_ratio_median']:.2f}, fully airborne "
            f"{r['fully_airborne_clearance_ratio_median']:.2f} "
            f"({r['fully_airborne_share']:.2f} of swings)"
        )
        print(
            f"  foot-position term (fit R2 >= {split['fit_r2_min']:.4f}): z share "
            f"{split['share']['left_z'] + split['share']['right_z']:.2f}; median error in "
            "tolerances "
            + " ".join(f"{k} {v:.1f}" for k, v in split["median_error_in_tolerances"].items())
        )
        print(
            "  Spearman(lift, field error): "
            + " ".join(f"{k} {v:+.2f}" for k, v in r["lift_vs_field_error_spearman"].items())
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
