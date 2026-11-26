"""Read R4, the registered unseen-body criterion (prereg, "Body breadth ...", 2026-11-19).

Each body is left out in turn, for the encoder (G-A3b's ``pact_kd``) and the translator (G-X's
``pact_x``):

    gap closure = (control - held out) / (control - in training)

The terms are:
* **control:** the physics-only estimate for the encoder, copying the source body for the
  translator;
* **in training:** the same arm trained with that body (``pact_gate_a3.py`` / ``pact_gate_x.py``
  on every body, the manifest's ``cross`` jobs);
* **held out:** G-N1's zero-shot arm with the body left out (the manifest's ``r4`` jobs).

All three are read on the final co-executions, as RMSE on the body's test probes or chains.
The claim "zero-shot to an unseen body" is kept if the mean gap closure over the bodies is at
least 0.5 for both parts. Otherwise the new-body claim is the cheap addition (G-N2, G-N3).

    python transfer/scripts/icml_r4_read.py [--seeds 0,1,2] --output transfer/results/icml_r4.json
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

RESULTS = REPO / "transfer/results"
BAR = 0.5


def load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def terms(part: str, body: str, seed: int) -> dict | None:
    held = load(f"icml_r4_{part}_{body}_s{seed}.json")
    if part == "encoder":
        full = load(f"icml_final_gate_a3_s{seed}.json")
        if held is None or full is None or body not in full["bodies"]:
            return None
        row = full["bodies"][body]
        control, trained = row["physics"]["all"], row["pact_kd"]["all"]
        zero = held["budgets"]["1"]["arms"]["zero"]["rmse"]["all"]
    else:
        full = load(f"icml_final_gate_x_s{seed}.json")
        if held is None or full is None or body not in full["bodies"]:
            return None
        row = full["bodies"][body]
        control, trained = row["copy"], row["pact_x"]
        zero = held["budgets"]["1"]["arms"]["zero"]["rmse"]
    gap = control - trained
    return {
        "control": control, "in_training": trained, "held_out": zero,
        "closure": (control - zero) / gap if gap > 0 else None,
    }  # fmt: skip


def read(bodies: list[str], seeds: list[int]) -> dict:
    report: dict = {"bar": BAR, "seeds": seeds, "parts": {}}
    for part in ("encoder", "translator"):
        rows = {b: {s: terms(part, b, s) for s in seeds} for b in bodies}
        per_seed = []
        for s in seeds:
            values = [rows[b][s]["closure"] for b in bodies if rows[b][s] and rows[b][s]["closure"] is not None]
            if len(values) == len(bodies):
                per_seed.append(float(np.mean(values)))
        mean = float(np.mean(per_seed)) if per_seed else None
        se = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else None
        report["parts"][part] = {
            "bodies": rows, "mean_closure_per_seed": per_seed, "mean_closure": mean, "se": se,
            "complete_seeds": len(per_seed), "passes": None if mean is None else bool(mean >= BAR),
        }  # fmt: skip
    verdicts = [p["passes"] for p in report["parts"].values()]
    report["zero_shot_body_claim"] = None if None in verdicts else bool(all(verdicts))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bodies", default=",".join(registry(TEACHERS)))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output", type=Path, default=RESULTS / "icml_r4.json")
    args = parser.parse_args()
    report = read([b for b in args.bodies.split(",") if b], [int(s) for s in args.seeds.split(",")])
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for part, p in report["parts"].items():
        cells = " ".join(
            f"{b}:{np.mean([r['closure'] for r in rows.values() if r and r['closure'] is not None]):.2f}"
            if any(r and r["closure"] is not None for r in rows.values())
            else f"{b}:-"
            for b, rows in p["bodies"].items()
        )
        print(f"{part:10s} mean closure {p['mean_closure']} (seeds {p['complete_seeds']}) | {cells}")
    print(f"zero-shot body claim: {report['zero_shot_body_claim']}; written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
