"""Recompute gate C "at the body's rate" for existing results, loads resampled without bias.

``evaluate_contract.signature_at_rate`` resampled the teacher's loads linearly before
thresholding them, which lengthens every stance. This script
leaves every result file as it is and writes, for each run that carries a
``verdict_at_body_rate``, three verdicts side by side: against the teacher on its own
grid (the stored ``verdict``), at the body's rate as stored (linear loads), and at the
body's rate with the loads resampled in log units, the nearest-sample duty beside it.

    python transfer/scripts/body_rate_verdicts.py --glob "zero_shot_nao_*_l4_*.json" \
        --output transfer/results/body_rate_verdicts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_contract import signature_at_rate  # noqa: E402
from evaluate_latent import verdict  # noqa: E402


def in_v(run: dict) -> dict:
    speed = run["forward_speed"]
    return {**run, "forward_speed": None if speed is None else speed / run["velocity_unit"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--glob", action="append", required=True, help="pattern in transfer/results"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = []
    for pattern in args.glob:
        files += sorted((REPO_ROOT / "transfer" / "results").glob(pattern))
    cache: dict = {}
    entries = []
    for path in files:
        result = json.loads(path.read_text(encoding="utf-8"))
        teacher = result["runs"]["teacher"]
        for name, run in result["runs"].items():
            if "verdict_at_body_rate" not in run:
                continue
            rate = run["verdict_at_body_rate"]["body_dt_star"]
            key = (teacher["file"], round(rate, 9))
            if key not in cache:
                capture = REPO_ROOT / teacher["file"]
                cache[key] = {
                    mode: signature_at_rate(capture, rate, loads=mode)
                    for mode in ("log", "nearest")
                }
            unbiased = {**teacher, "signature": cache[key]["log"]}
            fixed = verdict(in_v(run), in_v(unbiased))
            entries.append(
                {
                    "result": path.name,
                    "run": name,
                    "own_grid": {
                        "passes": run["verdict"]["passes"],
                        "signature_deviation": run["verdict"]["signature_deviation"],
                    },
                    "at_rate_linear": {
                        "passes": run["verdict_at_body_rate"]["passes"],
                        "signature_deviation": run["verdict_at_body_rate"]["signature_deviation"],
                        "teacher_duty": run["verdict_at_body_rate"]["teacher_signature"][
                            "duty_factor"
                        ],
                    },
                    "at_rate_unbiased": {
                        "passes": fixed["passes"],
                        "signature_deviation": fixed["signature_deviation"],
                        "teacher_duty": list(cache[key]["log"]["duty_factor"]),
                        "teacher_duty_nearest": list(cache[key]["nearest"]["duty_factor"]),
                    },
                    "teacher_duty_own_grid": teacher["signature"]["duty_factor"],
                }
            )

    # Passes per result group (the file name without its command) and run.
    groups: dict = defaultdict(lambda: defaultdict(int))
    for e in entries:
        group = e["result"].rsplit("_", 1)[0] + f" [{e['run']}]"
        groups[group]["commands"] += 1
        for ref in ("own_grid", "at_rate_linear", "at_rate_unbiased"):
            groups[group][ref] += int(e[ref]["passes"])
    summary = {g: dict(v) for g, v in sorted(groups.items())}
    changed = [
        f"{e['result']} [{e['run']}]"
        for e in entries
        if e["at_rate_linear"]["passes"] != e["at_rate_unbiased"]["passes"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "changed": changed, "entries": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"{'group':<78} cmds own linear unbiased")
    for group, counts in summary.items():
        print(
            f"{group:<78} {counts['commands']:>4} {counts['own_grid']:>3} "
            f"{counts['at_rate_linear']:>6} {counts['at_rate_unbiased']:>8}"
        )
    print(f"verdicts that change: {len(changed)}")
    for c in changed:
        print("  ", c)
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
