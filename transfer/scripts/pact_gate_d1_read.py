"""Read gate G-D1: the shared PACT policy against each body's own tracker, probe by probe.

Each DAgger round's capture (``pact_dagger.sh``) is scored like the trackers' map stage
(``evaluate_competence.py``) on the same 71 gait probes, so the two scores pair by probe.
Per body and round:

- probes passed, PACT and tracker, for the grid, the manoeuvres, the twins and all;
- the paired difference PACT − tracker in probes passed, with its bootstrap SE over
  probes (the unit both scores share);
- falls (summed over the probes' environments), PACT − tracker, SE over probes.

Registered criterion (prereg, G-D1): on every body, competence within 2 SE of the
tracker's (PACT − tracker ≥ −2 SE, read on the grid and on all probes) and falls no more
than the tracker's + 2 SE. Only representable probes count.

Usage::

    python transfer/scripts/pact_gate_d1_read.py --tag main
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1].parent
BODIES = ("g1", "nao", "h1", "cassie")
GROUPS = ("grid", "manoeuvre", "twin")


def probe_table(path: Path) -> dict[str, dict]:
    probes = json.loads(path.read_text("utf-8"))["probes"]
    return {p["name"]: p for p in probes if p.get("representable", True)}


def paired(student: dict, teacher: dict, names: list[str], key, draws: int, seed: int):
    """``(student sum, teacher sum, difference, bootstrap SE)`` of ``key`` over ``names``."""
    s = np.array([float(key(student[n])) for n in names])
    t = np.array([float(key(teacher[n])) for n in names])
    d = s - t
    if len(d) == 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(draws, len(d)))].sum(1)
    return float(s.sum()), float(t.sum()), float(d.sum()), float(boots.std(ddof=1))


def read(tag: str, draws: int, seed: int) -> dict:
    report: dict = {"gate": "G-D1", "tag": tag, "bootstrap": draws, "bodies": {}}
    for body in BODIES:
        teacher_path = REPO / f"transfer/results/icml_competence_{body}.json"
        rounds = sorted(
            REPO.glob(f"transfer/results/icml_gate_d1_{tag}_r*_{body}.json"),
            key=lambda p: int(p.stem.split("_r")[-1].split("_")[0]),
        )
        if not teacher_path.exists() or not rounds:
            continue
        teacher = probe_table(teacher_path)
        entry: dict = {"tracker": str(teacher_path.relative_to(REPO)), "rounds": {}}
        for path in rounds:
            k = int(path.stem.split("_r")[-1].split("_")[0])
            student = probe_table(path)
            names = sorted(set(student) & set(teacher))
            row: dict = {"capture_score": str(path.relative_to(REPO))}
            for group in (*GROUPS, "all"):
                chosen = [n for n in names if group == "all" or student[n]["group"] == group]
                s, t, d, se = paired(student, teacher, chosen, lambda p: p["passes"], draws, seed)
                row[group] = {"pact": s, "tracker": t, "diff": d, "se": se, "probes": len(chosen)}
            s, t, d, se = paired(student, teacher, names, lambda p: p["falls"], draws, seed)
            row["falls"] = {"pact": s, "tracker": t, "diff": d, "se": se}
            competent = all(row[g]["diff"] >= -2 * row[g]["se"] for g in ("grid", "all"))
            safe = row["falls"]["diff"] <= 2 * row["falls"]["se"]
            row["passes"] = bool(competent and safe)
            entry["rounds"][k] = row
        report["bodies"][body] = entry
    last = {b: e["rounds"][max(e["rounds"])] for b, e in report["bodies"].items()}
    report["passes_last_round"] = bool(last) and all(r["passes"] for r in last.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="main")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = read(args.tag, args.bootstrap, args.seed)
    output = args.output or REPO / f"transfer/results/icml_gate_d1_{args.tag}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for body, entry in report["bodies"].items():
        for k, row in entry["rounds"].items():
            cells = "  ".join(
                f"{g} {row[g]['pact']:.0f}/{row[g]['tracker']:.0f} "
                f"({row[g]['diff']:+.0f} ± {row[g]['se']:.1f})"
                for g in (*GROUPS, "all")
            )
            falls = row["falls"]
            print(
                f"{body:7s} r{k}: {cells}  falls {falls['pact']:.0f}/{falls['tracker']:.0f} "
                f"({falls['diff']:+.0f} ± {falls['se']:.1f})  {'PASS' if row['passes'] else '-'}"
            )
    print(f"G-D1 ({args.tag}), last round, every body: {report['passes_last_round']}")
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
