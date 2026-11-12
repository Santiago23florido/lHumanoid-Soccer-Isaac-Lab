"""W5's robustness table, read from the results ``w5_robust.sh`` wrote.

For each tracker and skill at L4's forward command, deployed path: the nominal capture
(L4's own) against each perturbation (pushes of up to 0.1 and 0.2 ``V`` every 20 ``T`` on
average; the deployed estimate read with 2 and 4 times the estimator's precision). Per row:
falls over episodes, the speed ratio, the largest gate C deviation, the duty deviation and
the clearance reached over asked (``common/contact.py``). ``flat_b4`` rows come from W5b
(``w5b_flat_b4.sh``) where they exist. Instrumentation: it decides
nothing.

    python transfer/scripts/w5_table.py --output transfer/results/w5_robust.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
RESULTS = REPO_ROOT / "transfer" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contact_fidelity import score  # noqa: E402

TRACKERS = (("g1", "warm_r3_logload_sym"), ("nao", "r1_3498_sym_v3"), ("nao", "r1_3498_sym_v3_gen"))
SKILLS = ("flat_dagger2", "rough_b4", "flat_b4")
CONDITIONS = ("nominal", "push10", "push20", "noise2", "noise4")
SIGNATURE = ("stride_length", "stride_period", "duty_factor_0", "duty_factor_1")


def row(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
    verdict = run["verdict"]
    dev = verdict["signature_deviation"]
    contact = score(path)["runs"]["encoder"]
    lift = [
        b / a for b, a in zip(contact["clearance_body"], contact["clearance_asked"], strict=True)
    ]
    return {
        "falls": run["falls"],
        "episodes": run["episodes"],
        "fall_rate": run["fall_rate"],
        "speed_ratio": verdict["speed_ratio"],
        "largest": max(dev[k] for k in SIGNATURE),
        "duty": (dev["duty_factor_0"] + dev["duty_factor_1"]) / 2,
        "lift": float(np.mean(lift)),
        "passes": verdict["passes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {}
    for body, tag in TRACKERS:
        for skill in SKILLS:
            rows = {}
            for condition in CONDITIONS:
                suffix = "l4_forward" if condition == "nominal" else f"robust_{condition}_forward"
                path = RESULTS / f"zero_shot_{body}_{tag}_{skill}_auto_kin_{suffix}.json"
                if path.exists():
                    rows[condition] = row(path)
            key = f"{body} {tag} {skill}"
            report[key] = rows
            print(key)
            for condition, r in rows.items():
                print(
                    f"  {condition:8} falls {r['falls']:3}/{r['episodes']:<4} speed "
                    f"{r['speed_ratio']:.3f}  largest {r['largest']:.3f}  duty {r['duty']:.3f}  "
                    f"lift {r['lift']:.2f}  passes {r['passes']}"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
