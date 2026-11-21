"""Read a contact curriculum's record (``latent/curriculum.py``): where it drew, and whether a
cell's score was still falling.

The record (``logs/pact/curriculum/<tag>_<body>.npz``, written by the tracking command at
every refresh) holds per cell its score (the tracker's contact disagreement), decayed counts
and share of the draw. This prints, at the last refresh, the score per period and duty bin
for each speed bin, the range of the draw's boost over the uniform one, and for one cell
(``--cell P:duty:forward``, default the probes' failing 1.8:0.57:0.05) its boost over the
run and whether its score fell over the run's last 30 % (the drop of a least-squares line
through those refreshes against that line's standard error). A bias curriculum's record also
holds each cell's signed mean contact error (its duty bias); the cell's is printed.

    python transfer/scripts/pact_curriculum_read.py logs/pact/curriculum/contactcur_h1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.curriculum import (  # noqa: E402
    FORWARD_EDGES,
    MIN_COUNT,
    PERIOD_BINS,
    segment_cells,
)

FAMILY = {"period": (0.6, 3.0), "duty": (0.55, 0.95)}
"""The walking family the teachers are fine-tuned on (``GaitRanges`` with duty 0.55-0.95)."""
TAIL = 0.3


def trend(values: np.ndarray) -> tuple[float, float]:
    """``(drop, se)``: how far a least-squares line falls over ``values`` and its SE."""
    x = np.arange(len(values), dtype=float)
    if len(values) < 3:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, values, 1)
    residual = values - (slope * x + intercept)
    se = np.sqrt(residual.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    span = x[-1] - x[0]
    return float(-slope * span), float(se * span)


def read(path: Path, cell_spec: str) -> dict:
    record = np.load(path)
    score, counts, share = record["score"], record["counts"], record["draw_share"]
    bank = record["bank_share"]
    duty_bins = score.shape[1] // (PERIOD_BINS * (len(FORWARD_EDGES) + 1))
    period, duty, forward = (float(v) for v in cell_spec.split(":"))
    cell = int(
        segment_cells(
            torch.tensor([period]), torch.tensor([[duty, duty]]), torch.tensor([forward]),
            FAMILY["period"], FAMILY["duty"],
        )[0]
    )  # fmt: skip
    kind = str(record["kind"]) if "kind" in record else "disagreement"
    boost = share / np.maximum(bank, 1e-12)
    tail = score[int(len(score) * (1.0 - TAIL)) :, cell]
    drop, se = trend(tail)
    return {
        "kind": kind,
        "refreshes": len(score),
        "last_step": int(record["step"][-1]),
        "duty_bins": duty_bins,
        "score_last": score[-1].tolist(),
        "seen_last": (counts[-1] >= MIN_COUNT).tolist(),
        "boost_range_last": [float(boost[-1][bank > 0].min()), float(boost[-1][bank > 0].max())],
        "cell": {
            "spec": cell_spec,
            "index": cell,
            "bank_share": float(bank[cell]),
            "boost_mean": float(boost[:, cell].mean()),
            "boost_last": float(boost[-1, cell]),
            "score_first": float(score[0, cell]),
            "score_last": float(score[-1, cell]),
            "tail_refreshes": len(tail),
            "tail_drop": drop,
            "tail_drop_se": se,
            "still_falling": bool(drop > se),
            "signed_last": float(record["value"][-1, cell]) if "value" in record else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("record", type=Path)
    parser.add_argument("--cell", default="1.8:0.57:0.05", help="period:duty:forward")
    parser.add_argument("--output", type=Path, default=None, help="also write the reading as JSON")
    args = parser.parse_args()
    out = read(args.record, args.cell)
    duty_bins = out["duty_bins"]
    score = np.array(out["score_last"]).reshape(PERIOD_BINS, duty_bins, len(FORWARD_EDGES) + 1)
    seen = np.array(out["seen_last"]).reshape(score.shape)
    low = FAMILY["duty"][0]
    print(f"{args.record.name} ({out['kind']}): {out['refreshes']} refreshes to step {out['last_step']}")
    for f in range(score.shape[2]):
        print(f"  forward bin {f}: rows period bins, columns mean duty from {low:.2f} by 0.05 (* unseen)")
        for p in range(PERIOD_BINS):
            cells = " ".join(
                f"{score[p, d, f]:.3f}{' ' if seen[p, d, f] else '*'}" for d in range(duty_bins)
            )
            print(f"    {cells}")
    lo, hi = out["boost_range_last"]
    c = out["cell"]
    print(f"  draw boost over uniform at the last refresh: {lo:.2f}-{hi:.2f}")
    print(
        f"  cell {c['spec']} (#{c['index']}, {c['bank_share']:.4f} of the bank): boost mean "
        f"{c['boost_mean']:.2f}, last {c['boost_last']:.2f}; score {c['score_first']:.3f} -> "
        f"{c['score_last']:.3f}; last {c['tail_refreshes']} refreshes drop {c['tail_drop']:+.4f} "
        f"(SE {c['tail_drop_se']:.4f}) -> {'still falling' if c['still_falling'] else 'not falling'}"
    )
    if c["signed_last"] is not None:
        print(f"  cell's signed contact error (duty bias) at the last refresh: {c['signed_last']:+.4f}")
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
