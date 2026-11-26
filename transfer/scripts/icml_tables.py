"""The ICML paper's tables, rebuilt from ``transfer/results`` into LaTeX and a Markdown preview.

Every cell reads a result file written by a script named in ``icml_runs.py``. A cell whose files
do not exist yet prints ``--`` and is listed as pending, so the paper compiles from the first day
and fills as the manifest runs. A value over seeds is the mean ± its standard error (seeds as
the unit, ``n - 1``); a single seed prints its value and the number of seeds used.

    python transfer/scripts/icml_tables.py [--out docs/icml/generated]

Tables:
* ``table_distribution``: one policy for every body, walking probes passed and falls on the
  probe map (G-D1 criterion, walking probes), per arm, against each body's teacher.
* ``table_held_out_task``: the held-out task (sidestep, never trained on) on every body. Falls
  are summed over both generators and three commands; the largest gait deviation is a ratio to
  the teacher's.
* ``table_held_out_body``: each body left out of the shared model, offline R² (torque) on its
  test probes, against the same arm trained with it.
* ``table_teachers``: the teacher bar per body (walking probes; training-task terminations over
  three seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from body_conditions import TEACHERS, registry  # noqa: E402
from icml_runs import ARMS, SEEDS  # noqa: E402

RESULTS = REPO / "transfer/results"
LABEL = {
    "pact_d": "PACT", "pact_na": "PACT w/o anchor", "pact_nb": "PACT w/o $E_b$",
    "urma_d": "URMA", "hpt_d": "HPT",
}  # fmt: skip
BODY = {"g1": "G1", "nao": "NAO", "h1": "H1", "cassie": "Cassie", "digit": "Digit"}
PENDING: list[str] = []


def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        PENDING.append(name)
        return None
    return json.loads(path.read_text("utf-8"))


def over_seeds(values: list[float | None], digits: int = 1) -> str:
    got = [v for v in values if v is not None]
    if not got:
        return "--"
    mean = float(np.mean(got))
    if len(got) == 1:
        return f"{mean:.{digits}f}$^{{(1)}}$"
    se = float(np.std(got, ddof=1) / np.sqrt(len(got)))
    note = "" if len(got) == len(values) else f"$^{{({len(got)})}}$"
    return f"{mean:.{digits}f} $\\pm$ {se:.{digits}f}{note}"


def last_round(d1: dict | None, body: str) -> dict | None:
    if d1 is None or body not in d1.get("bodies", {}):
        return None
    rounds = d1["bodies"][body]["rounds"]
    return rounds[max(rounds, key=int)]


def table_distribution(bodies: list[str]) -> tuple[list[str], list[list[str]]]:
    header = ["Body", "Teacher"] + [LABEL[a] for a in ARMS]
    rows = []
    for b in bodies:
        variant = registry(TEACHERS)[b]["variant"]
        teacher = load(f"icml_teacher_{b}_{variant}_probes.json")
        walk = None
        if teacher is not None:
            from pact_teacher_audit_read import probes

            score = probes(RESULTS / f"icml_teacher_{b}_{variant}_probes.json")
            walk = f"{score['walk_passed']}/{score['walk_probes']}" if score else None
        row = [BODY.get(b, b), walk or "--"]
        for arm in ARMS:
            passed, falls = [], []
            for s in SEEDS:
                r = last_round(load(f"icml_final_{arm}_s{s}_d1.json"), b)
                passed.append(None if r is None else r["all"]["pact"])
                falls.append(None if r is None else r["falls"]["pact"])
            row.append(f"{over_seeds(passed)} ({over_seeds(falls, 0)})")
        rows.append(row)
    return header, rows


def table_held_out_task(bodies: list[str]) -> tuple[list[str], list[list[str]]]:
    header = ["Body", "Teacher falls"] + [f"{LABEL[a]} falls / dev." for a in ARMS]
    rows = []
    for b in bodies:
        teacher_falls = None
        cells = []
        for arm in ARMS:
            falls, ratio = [], []
            for s in SEEDS:
                r = load(f"icml_held_out_icml_{arm}_s{s}.json")
                c = None if r is None else r["bodies"].get(b, {}).get("criterion")
                falls.append(None if c is None else c["falls"]["model"])
                ratio.append(None if c is None else c["largest_ratio"])
                if c is not None and teacher_falls is None:
                    teacher_falls = c["falls"]["teacher"]
            cells.append(f"{over_seeds(falls, 0)} / {over_seeds(ratio, 2)}")
        rows.append([BODY.get(b, b), "--" if teacher_falls is None else f"{teacher_falls:.0f}"] + cells)
    return header, rows


def table_held_out_body(bodies: list[str]) -> tuple[list[str], list[list[str]]]:
    arms = [a for a in ARMS if a != "hpt_d"]  # HPT has no stem for an unseen body
    header = ["Held-out body"] + [f"{LABEL[a]} held out / trained" for a in arms]
    rows = []
    for b in bodies:
        row = [BODY.get(b, b)]
        for arm in arms:
            held, trained = [], []
            for s in SEEDS:
                loo = load(f"icml_loo_{arm}_{b}_s{s}.json")
                held.append(None if loo is None else loo["bodies"][b][arm]["r2"])
                full = load(f"icml_final_{arm}_s{s}_offline.json")
                trained.append(None if full is None or b not in full["bodies"] else full["bodies"][b][arm]["r2"])
            row.append(f"{over_seeds(held, 2)} / {over_seeds(trained, 2)}")
        rows.append(row)
    return header, rows


def table_teachers(bodies: list[str]) -> tuple[list[str], list[list[str]]]:
    header = ["Body", "Walking probes", "Probe falls", "Terminations (3 seeds)", "Near-perfect"]
    final = load("icml_teacher_final.json")
    rows = []
    for b in bodies:
        variant = registry(TEACHERS)[b]["variant"]
        e = None if final is None else final["teachers"].get(f"{b}:{variant}")
        if e is None or e["probes"] is None:
            rows.append([BODY.get(b, b), "--", "--", "--", "--"])
            continue
        p = e["probes"]
        terms = (
            f"{e['counted_terminations']}/{e['episodes']} ({100 * e['termination_share']:.2f} \\%)"
            if e["termination_share"] is not None
            else "--"
        )
        rows.append([
            BODY.get(b, b), f"{p['walk_passed']}/{p['walk_probes']}", str(p["walk_falls"]), terms,
            {True: "yes", False: "no", None: "--"}[e["near_perfect"]],
        ])  # fmt: skip
    return header, rows


def latex(header: list[str], rows: list[list[str]]) -> str:
    cols = "l" + "c" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown(name: str, header: list[str], rows: list[list[str]]) -> str:
    clean = lambda c: c.replace("$\\pm$", "±").replace("\\%", "%").replace("$", "")  # noqa: E731
    out = [f"### {name}", "", "| " + " | ".join(map(clean, header)) + " |", "|" + " --- |" * len(header)]
    out += ["| " + " | ".join(map(clean, r)) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "docs/icml/generated")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bodies = list(registry(TEACHERS))
    preview = ["# ICML tables (generated by transfer/scripts/icml_tables.py; -- = pending)", ""]
    for fn in (table_teachers, table_distribution, table_held_out_task, table_held_out_body):
        header, rows = fn(bodies)
        (args.out / f"{fn.__name__}.tex").write_text(latex(header, rows), encoding="utf-8")
        preview.append(markdown(fn.__name__, header, rows))
    pending = sorted(set(PENDING))
    preview += ["### pending result files", "", *[f"- {p}" for p in pending]]
    (args.out / "tables.md").write_text("\n".join(preview) + "\n", encoding="utf-8")
    print(f"written {args.out}; {len(pending)} result files pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
