"""A teacher's gait probes cell by cell: passes and duty bias per (period, asked duty).

Reads ``transfer/results/icml_teacher_<body>_<variant>_probes.json`` (``pact_teacher_audit.sh``,
stage ``probes``) and, for the grid probes, prints per period and asked duty how many probes
pass and the body's duty minus the asked one, averaged over the speeds. ``--cell P:duty``
(repeatable) also prints the passes in those cells summed, per variant.

    python transfer/scripts/pact_probe_cells.py --body h1 --variants contact,mid \\
        --cell 0.8:0.55 --cell 1.2:0.55 --cell 1.8:0.55 --cell 2.6:0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def cells(path: Path) -> dict[tuple[str, float], list[tuple[bool, float]]]:
    """``(period, asked duty) -> [(passes, body duty - asked duty)]`` over the grid probes."""
    table: dict[tuple[str, float], list[tuple[bool, float]]] = {}
    for probe in json.loads(path.read_text(encoding="utf-8"))["probes"]:
        if probe["group"] != "grid" or not probe["representable"]:
            continue
        period = probe["name"].split("_P")[1].split("_")[0]
        asked = float(np.mean(probe["asked"]["duty"]))
        bias = float(np.mean(probe["body"]["duty"])) - asked
        key = (period, round(asked * 20) / 20)
        table.setdefault(key, []).append((bool(probe["passes"]), bias))
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body", required=True)
    parser.add_argument("--variants", default="sym,contact")
    parser.add_argument("--cell", action="append", default=[], help="period:duty, e.g. 1.8:0.55")
    args = parser.parse_args()
    for variant in args.variants.split(","):
        path = REPO / f"transfer/results/icml_teacher_{args.body}_{variant}_probes.json"
        if not path.exists():
            print(f"{args.body} {variant}: missing {path.name}")
            continue
        table = cells(path)
        periods = sorted({p for p, _ in table}, key=float)
        duties = sorted({d for _, d in table})
        print(f"== {args.body} {variant}: passes/probes and duty bias, rows period, columns asked duty")
        print("   P     " + "  ".join(f"{d:>12.2f}" for d in duties))
        for period in periods:
            row = []
            for duty in duties:
                entries = table.get((period, duty))
                if not entries:
                    row.append(f"{'-':>12s}")
                    continue
                passed = sum(p for p, _ in entries)
                bias = np.mean([b for _, b in entries])
                row.append(f"{passed}/{len(entries)} {bias:+.2f}".rjust(12))
            print(f"   {period:5s} " + "  ".join(row))
        for cell in args.cell:
            period, duty = cell.split(":")
            entries = table.get((period, round(float(duty) * 20) / 20), [])
            passed = sum(p for p, _ in entries)
            bias = np.mean([b for _, b in entries]) if entries else float("nan")
            print(f"   cell P{period} d{duty}: {passed}/{len(entries)}, duty bias {bias:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
