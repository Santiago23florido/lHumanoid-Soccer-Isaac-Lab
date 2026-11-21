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
_LOAD = slice(22, 24)
CONTACT = 0.05
"""Load (in the body's weight) above which a foot is in contact: the contact reward's."""
WALK_DUTY = 0.55


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


def disagreement(path: Path) -> dict[str, float]:
    """The teacher's contact disagreement on the grid probes of a probe capture."""
    data = np.load(path)
    probes = json.loads(str(data["probes"]))
    settle = int(data["settle_steps"])
    # The body's state after a step against the plan row that step was due on.
    reached = data["skill_state"][1:, :, _LOAD] > CONTACT
    planned = data["contract_due"][:-1, :, _LOAD] > CONTACT
    share = (reached != planned).mean(-1)[settle:]
    per_probe: dict[str, list[float]] = {}
    for env, index in enumerate(data["probe_index"]):
        if index >= 0:
            per_probe.setdefault(probes[index]["name"], []).append(float(share[:, env].mean()))
    grid = {name: float(np.mean(v)) for name, v in per_probe.items() if name.startswith("v")}
    duty = {name: float(name.split("_d")[1]) for name in grid}
    return {
        "walking_mean": float(np.mean([v for n, v in grid.items() if duty[n] >= WALK_DUTY])),
        "p1.8_d0.55": float(np.mean([v for n, v in grid.items() if "_P1.8_d0.55" in n])),
        "best_grid": float(min(grid.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body", required=True)
    parser.add_argument("--variants", default="sym,contact")
    parser.add_argument("--cell", action="append", default=[], help="period:duty, e.g. 1.8:0.55")
    parser.add_argument("--disagreement", action="store_true", help="read the probe capture too")
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
        if args.disagreement:
            capture = REPO / f"logs/pact/teachers/{args.body}_{variant}_probes.npz"
            if not capture.exists():
                print(f"   contact disagreement: missing {capture.name}")
                continue
            d = disagreement(capture)
            print(
                f"   contact disagreement: walking mean {d['walking_mean']:.3f}, P1.8 d0.55 "
                f"{d['p1.8_d0.55']:.3f}, best grid cell {d['best_grid']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
