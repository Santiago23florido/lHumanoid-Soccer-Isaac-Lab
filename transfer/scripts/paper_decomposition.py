"""The transfer's error decomposed into the generator's, the tracker's and estimation's parts.

An L4 result (``evaluate_contract.py``) holds, for one command, the teacher's gait
signature, the body's on the deployed path (``encoder``: the kinematic estimator) and
on the true-state path (``truth``), and for each path the signature its plans asked
for (``asked``). With every deviation relative to the teacher and signed,

    body - teacher = (asked - teacher) + (body - asked)
                     generator           tracker

holds exactly on each path. Estimation's part is what the deployed path adds to the
true one, ``encoder - truth``: on the deployed path the generator reads an estimated
history too, so this term is the estimate's effect through the plans and the tracker
together, not the estimator's error alone. Reported for the stride's length and period,
each foot's duty factor and the speed, per body, skill and command, and averaged over
the commands in absolute value.

Usage::

    python transfer/scripts/paper_decomposition.py --output transfer/paper/data/decomposition.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[2] / "transfer" / "results"
MODEL = {"g1": "warm_r3_logload_sym", "nao": "r1_3498_sym_v3"}
"""The adopted model's tags (``incumbents.json``), deployed symmetrized."""
SKILLS = {"rough": "rough_b4", "flat": "flat_dagger2"}
COMMANDS = ("forward", "fast", "turning")
FEATURES = ("speed", "stride_length", "stride_period", "duty_left", "duty_right")


def features(signature: dict) -> dict[str, float]:
    duty = signature["duty_factor"]
    return {
        "speed": signature["speed"],
        "stride_length": signature["stride_length"],
        "stride_period": signature["stride_period"],
        "duty_left": duty[0],
        "duty_right": duty[1],
    }


def relative(a: dict, b: dict, reference: dict) -> dict[str, float]:
    return {k: (a[k] - b[k]) / reference[k] for k in FEATURES}


def decompose(path: Path) -> dict:
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    teacher = features(runs["teacher"]["signature"])
    out = {}
    for name in ("encoder", "truth"):
        body = features(runs[name]["signature"])
        asked = features(runs[name]["asked"]["signature"])
        out[name] = {
            "total": relative(body, teacher, teacher),
            "generator": relative(asked, teacher, teacher),
            "tracker": relative(body, asked, teacher),
        }
    encoder, truth = features(runs["encoder"]["signature"]), features(runs["truth"]["signature"])
    out["estimator"] = relative(encoder, truth, teacher)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"model": MODEL, "features": FEATURES, "cases": {}, "summary": {}}
    for body, tag in MODEL.items():
        for skill, generator in SKILLS.items():
            rows = []
            for command in COMMANDS:
                path = RESULTS / f"zero_shot_{body}_{tag}_{generator}_auto_kin_l4_{command}.json"
                if not path.exists():
                    continue
                case = decompose(path)
                report["cases"][f"{body}/{skill}/{command}"] = case
                rows.append(case)
            if not rows:
                continue
            terms = {
                "total": [r["encoder"]["total"] for r in rows],
                "generator": [r["encoder"]["generator"] for r in rows],
                "tracker": [r["encoder"]["tracker"] for r in rows],
                "estimator": [r["estimator"] for r in rows],
            }
            report["summary"][f"{body}/{skill}"] = {
                term: {k: float(np.mean([abs(v[k]) for v in values])) for k in FEATURES}
                for term, values in terms.items()
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, terms in report["summary"].items():
        print(key)
        for term, values in terms.items():
            cells = "  ".join(f"{k} {v:.3f}" for k, v in values.items())
            print(f"  {term:10s} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
