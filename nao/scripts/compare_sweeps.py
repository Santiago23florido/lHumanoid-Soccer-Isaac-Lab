"""Two trackers on the same robots: the difference, setting by setting, in units of its error.

Reads two summaries written by ``nao/scripts/sensitivity.py`` over the same settings
(``nao/scripts/evaluate_adaptation.sh`` or ``evaluate_sim2real.sh``) and reports, for
each setting both hold, the change from ``base`` to ``other`` in the contract error,
the falls per 1000 steps and the forward speed, each with the standard error of the
difference (the two sweeps are independent robots, so the errors add in quadrature)
and the difference in those units.

Usage::

    python nao/scripts/compare_sweeps.py nao/results/adaptation_<base>.json \
        nao/results/adaptation_<other>.json [--output <json>]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = {
    "contract_error": ("contract_error_env_median_mean", "contract_error_se"),
    "falls_per_1000": ("falls_per_1000_steps", "falls_per_1000_steps_se"),
    "forward_speed": ("forward_speed_m_s", "forward_speed_se"),
}


def compare(base: dict, other: dict) -> dict:
    rows = {}
    for name, row in base["rows"].items():
        if name not in other["rows"]:
            continue
        out = {}
        for metric, (value, se) in METRICS.items():
            delta = other["rows"][name][value] - row[value]
            error = math.hypot(row[se], other["rows"][name][se])
            if error > 0:
                z = delta / error
            else:  # neither moved over its robots: no fall in either, say
                z = 0.0 if delta == 0 else math.copysign(math.inf, delta)
            out[metric] = {
                "base": row[value],
                "other": other["rows"][name][value],
                "delta": delta,
                "se": error,
                "z": z,
            }
        rows[name] = out
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = compare(*(json.loads(p.read_text(encoding="utf-8")) for p in (args.base, args.other)))
    print(f"{'setting':18s}" + "".join(f"{m:>34s}" for m in METRICS))
    for name, out in rows.items():
        cells = "".join(
            f"{v['base']:9.3f} -> {v['other']:7.3f} ({v['z']:+6.1f} SE)" for v in out.values()
        )
        print(f"{name:18s}{cells}")
    if args.output is not None:
        result = {"base": str(args.base), "other": str(args.other), "rows": rows}
        args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
