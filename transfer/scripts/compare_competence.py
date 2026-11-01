"""Two trackers' competence over the same probes, compared probe by probe.

``evaluate_competence.py`` scores each probe against fixed criteria; to tell one
tracker from another the same probes are paired instead, and each continuous
measure's mean difference (candidate minus reference) is reported with its
bootstrap standard error over probes -- no threshold decides it. A difference
larger than twice its standard error is the adoption rule of
``transfer/docs/skill_space.md`` §9. The probes each passes alone are counted too
(McNemar's discordant pairs).

Usage::

    python transfer/scripts/compare_competence.py \
        transfer/results/competence_g1_dec_v1_corners.json \
        transfer/results/competence_g1_dec_v2h_corners.json \
        --output transfer/results/compare_g1_dec_v2h_vs_v1_corners.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MEASURES = ("fall_rate", "speed", "stride_period", "duty", "use")
"""Each lower is better: falls per episode, and deviations from the gait asked for."""


def measures(probe: dict) -> dict[str, float]:
    deviation = probe.get("deviation", {})
    duty = [deviation.get("duty_left"), deviation.get("duty_right")]
    return {
        "fall_rate": probe["falls"] / max(probe["episodes"], 1),
        "speed": deviation.get("speed", np.nan),
        "stride_period": deviation.get("stride_period", np.nan),
        "duty": float(np.nanmean([np.nan if d is None else d for d in duty])),
        "use": probe.get("use", np.nan),
    }


def compare(reference: dict, candidate: dict, draws: int, seed: int) -> dict:
    ref = {p["name"]: p for p in reference["probes"] if p.get("representable", True)}
    cand = {p["name"]: p for p in candidate["probes"] if p.get("representable", True)}
    rng = np.random.default_rng(seed)
    report = {}
    for group in sorted({p["group"] for p in ref.values()}):
        names = [n for n, p in ref.items() if p["group"] == group and n in cand]
        if not names:
            continue
        rows = {}
        for measure in MEASURES:
            a = np.array([measures(ref[n])[measure] for n in names], dtype=float)
            b = np.array([measures(cand[n])[measure] for n in names], dtype=float)
            keep = np.isfinite(a) & np.isfinite(b)
            diff = (b - a)[keep]
            if len(diff) == 0:
                continue
            picks = rng.integers(0, len(diff), size=(draws, len(diff)))
            rows[measure] = {
                "reference": float(a[keep].mean()),
                "candidate": float(b[keep].mean()),
                "difference": float(diff.mean()),
                "se": float(diff[picks].mean(axis=1).std()),
                "probes": int(keep.sum()),
            }
        passed_a = np.array([bool(ref[n]["passes"]) for n in names])
        passed_b = np.array([bool(cand[n]["passes"]) for n in names])
        report[group] = {
            "probes": len(names),
            "passed": {"reference": int(passed_a.sum()), "candidate": int(passed_b.sum())},
            "discordant": {
                "reference_only": int((passed_a & ~passed_b).sum()),
                "candidate_only": int((~passed_a & passed_b).sum()),
            },
            "measures": rows,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare(reference, candidate, args.draws, args.seed)
    print(f"{args.candidate.name} against {args.reference.name} (candidate - reference ± se)")
    for group, row in report.items():
        passed, discordant = row["passed"], row["discordant"]
        print(
            f"{group:10s} passed {passed['reference']}/{row['probes']} -> "
            f"{passed['candidate']}/{row['probes']} "
            f"(only reference {discordant['reference_only']}, "
            f"only candidate {discordant['candidate_only']})"
        )
        for measure, m in row["measures"].items():
            verdict = "better" if m["difference"] < -2 * m["se"] else (
                "worse" if m["difference"] > 2 * m["se"] else "no difference")  # fmt: skip
            print(
                f"  {measure:14s} {m['reference']:.3f} -> {m['candidate']:.3f}  "
                f"{m['difference']:+.3f} ± {m['se']:.3f}  {verdict}"
            )
    if args.output is not None:
        result = {"reference": str(args.reference), "candidate": str(args.candidate), **report}
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
