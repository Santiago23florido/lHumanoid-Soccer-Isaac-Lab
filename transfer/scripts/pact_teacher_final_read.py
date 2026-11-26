"""The final teacher audit: is each body's chosen teacher near-perfect?

The bar (registered before the teachers were chosen), per body:

* walking probes: at least 90 % of the gait probes that ask a walk (mean duty >= 0.55) pass,
  with no probe fall (``pact_teacher_audit.sh`` stage ``probes``, ``evaluate_competence.py``);
* training task: at most 1 % of env-episodes terminated, over three seeds of the training-task
  capture (``PACT_AUDIT_SEED`` 0, 1, 2: 639 env-episodes per body). A termination after a
  fallen procedural biped is the reference's, not the teacher's: it is reported apart and not
  counted (``pact_cap_test_read.classify``).

Each pair is ``body:variant`` (the audit's variant names: ``prec`` is the symmetrised precision
teacher). Missing seeds are reported and the bar is not read on fewer than three.

    python transfer/scripts/pact_teacher_final_read.py --pair g1:prec --pair h1:prec \\
        --pair nao:prec --pair cassie:sole --pair digit:prec \\
        --output transfer/results/icml_teacher_final.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from pact_cap_test_read import classify  # noqa: E402
from pact_teacher_audit_read import probes  # noqa: E402

TEACHERS = REPO / "logs/pact/teachers"
SEEDS = (0, 1, 2)
WALK_BAR = 0.90
TERMINATION_BAR = 0.01


def train_capture(body: str, variant: str, seed: int) -> Path:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return TEACHERS / f"{body}_{variant}{suffix}_train.npz"


def read_pair(body: str, variant: str) -> dict:
    score = probes(REPO / f"transfer/results/icml_teacher_{body}_{variant}_probes.json")
    seeds, missing = {}, []
    for seed in SEEDS:
        path = train_capture(body, variant, seed)
        if path.exists():
            c = classify(path)
            seeds[seed] = {k: c[k] for k in ("terminations", "task", "procedural", "biped_fallen", "envs")}
        else:
            missing.append(seed)
    episodes = sum(s["envs"] for s in seeds.values())
    counted = sum(s["task"] + s["procedural"] for s in seeds.values())
    walk_share = score["walk_passed"] / score["walk_probes"] if score else None
    entry = {
        "probes": score,
        "walk_share": walk_share,
        "seeds": seeds,
        "missing_seeds": missing,
        "episodes": episodes,
        "counted_terminations": counted,
        "biped_fallen": sum(s["biped_fallen"] for s in seeds.values()),
        "termination_share": counted / episodes if episodes else None,
    }
    entry["walks_ok"] = bool(score and walk_share >= WALK_BAR and score["walk_falls"] == 0)
    entry["terminations_ok"] = (
        None if missing else bool(entry["termination_share"] <= TERMINATION_BAR)
    )
    entry["near_perfect"] = None if missing else bool(entry["walks_ok"] and entry["terminations_ok"])
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pair", action="append", required=True, help="body:variant")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = {"walk_bar": WALK_BAR, "termination_bar": TERMINATION_BAR, "seeds": list(SEEDS), "teachers": {}}
    for pair in args.pair:
        body, variant = pair.split(":")
        e = read_pair(body, variant)
        report["teachers"][pair] = e
        p = e["probes"]
        walks = f"walks {p['walk_passed']}/{p['walk_probes']} falls {p['walk_falls']}" if p else "walks -"
        share = e["termination_share"]
        terms = (
            f"terminations {e['counted_terminations']}/{e['episodes']}"
            f" ({100 * share:.2f} %), {e['biped_fallen']} after a fallen biped"
            if share is not None
            else "terminations -"
        )
        seeds = f"seeds {sorted(e['seeds'])}" + (f", missing {e['missing_seeds']}" if e["missing_seeds"] else "")
        print(f"{pair:14s} {walks} | {terms} | {seeds} | near-perfect {e['near_perfect']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
