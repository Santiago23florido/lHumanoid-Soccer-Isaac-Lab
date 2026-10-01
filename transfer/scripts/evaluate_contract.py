"""Close the loop on a contract: does a body that follows the plans walk like the teacher?

Reads closed-loop captures (``g1/scripts/capture_rollouts.py``) and applies two
gates fixed before the tracker was trained:

**Use (gate B).** The state a body reaches is scored against the plan it was
holding and against the plan another environment was holding at the same
instant (``contract_error`` and ``contract_error_other``, recorded by the
capture). A body that follows its plan is much closer to its own: the median
own error must be below half the median other error. A body that ignored the
plan and walked anyway would score alike against both.

**Behaviour (gate C).** The same criteria every closed-loop run of the study
was held to, from ``evaluate_latent.py``: speed at least 0.90 of the
teacher's, fall rate at most 2 points above it, and stride length, stride
period and both duty factors within 10 %.

Usage::

    python transfer/scripts/evaluate_contract.py --run teacher=g1/rollouts/closed_loop_teacher.npz \
        --run tracker=g1/rollouts/closed_loop_track.npz --output transfer/results/contract_g1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_latent import CRITERIA, SETTLE_STEPS, summarise, verdict  # noqa: E402

USE_RATIO_MAX = 0.5


def contract_use(path: Path) -> dict | None:
    data = np.load(path)
    if "contract_error" not in data.files:
        return None
    own, other = data["contract_error"], data["contract_error_other"]
    done = data["done"]
    # The step after a reset carries no plan and scores zero; leave it out.
    first = np.zeros_like(done)
    first[1:] = done[:-1]
    valid = ~first & (own > 0)
    own, other = own[valid], other[valid]
    ratio = float(np.median(own) / np.median(other))
    return {
        "median_error_own_plan": float(np.median(own)),
        "median_error_other_plan": float(np.median(other)),
        "mean_error_own_plan": float(own.mean()),
        "ratio": ratio,
        "rule": f"median own / median other < {USE_RATIO_MAX}",
        "passes": ratio < USE_RATIO_MAX,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="name=path/to/capture.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        runs[name] = summarise(REPO_ROOT / path)
        use = contract_use(REPO_ROOT / path)
        if use is not None:
            runs[name]["contract_use"] = use
    if "teacher" not in runs:
        print("a run named 'teacher' is needed as the reference")
        return 2
    for name, run in runs.items():
        if name != "teacher":
            run["verdict"] = verdict(run, runs["teacher"])

    report = {
        "criteria": CRITERIA,
        "use_ratio_max": USE_RATIO_MAX,
        "settle_steps": SETTLE_STEPS,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, run in runs.items():
        sig = run["signature"]
        line = (
            f"{name:<22} falls {run['falls']}/{run['episodes']}  speed {run['forward_speed']}  "
            f"stride {sig['stride_length']:.3f}  period {sig['stride_period']:.3f}  "
            f"duty {np.round(sig['duty_factor'], 3)}"
        )
        if "verdict" in run:
            v = run["verdict"]
            line += (
                f"  ratio {v['speed_ratio']:.2f}  dev "
                f"{ {k: round(x, 3) for k, x in v['signature_deviation'].items()} }  "
                f"{'PASS' if v['passes'] else 'fail'}"
            )
        if "contract_use" in run:
            u = run["contract_use"]
            line += f"  use {u['ratio']:.3f} ({'PASS' if u['passes'] else 'fail'})"
        print(line)
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
