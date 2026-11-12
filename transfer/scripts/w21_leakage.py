"""W21's leakage reading: how much body each generator's history
summary carries beyond the ``z*`` history it summarises.

For every generator named, on one set of probe pairs (the bodies' own histories, as
``train_contract.py`` draws them, from the bodies' captures in the first generator's report):
``latent/body_data.leakage``, whose reference is the same probe on the same noisy ``z*``
windows. Every generator is read on the same device and pairs, so their figures compare.

The gate: a summary passes when its balanced accuracy exceeds the
window's by at most 3 SE of the difference (the two binomial SE combined).

    python transfer/scripts/w21_leakage.py --generators mt18_no_sidestep mt18_no_sidestep_m \\
        --output transfer/results/w21_leakage.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.body_data import (  # noqa: E402
    body_pairs,
    leakage,
    load_body_streams,
)
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402

RESULTS = REPO_ROOT / "transfer" / "results"
GATE_SE = 3.0
PAIRS = 50000


def gate(entry: dict) -> bool:
    combined = math.hypot(entry["se"], entry["window_se"])
    return bool(entry["beyond_window"] <= GATE_SE * combined)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generators", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    first = json.loads((RESULTS / f"contract_{args.generators[0]}.json").read_text("utf-8"))
    captures = [REPO_ROOT / c for c in first["bodies"]["captures"]]
    body, capture = load_body_streams(captures)
    body = body.to(args.device)
    groups = torch.as_tensor(first["bodies"]["groups"], device=args.device)[capture.to(args.device)]
    report = {"captures": first["bodies"]["captures"], "gate_se": GATE_SE, "generators": {}}
    for name in args.generators:
        contract = load_contract(REPO_ROOT / f"logs/contract/{name}/generator.pt", args.device)
        generator = contract.generator
        pairs = body_pairs(generator, body, groups, None, None, "own", PAIRS, seed=args.seed + 1)
        entry = leakage(generator, pairs, contract.sigma, seed=args.seed)
        entry["summary_beta"] = generator.summary_beta
        entry["passes"] = gate(entry)
        report["generators"][name] = entry
        verdict = "passes" if entry["passes"] else "FAILS"
        print(
            f"{name:24s} beta {generator.summary_beta:<6g}"
            f" summary {entry['balanced_accuracy']:.4f}"
            f"  window {entry['window_balanced_accuracy']:.4f}"
            f"  beyond {entry['beyond_window']:+.4f}  control {entry['control']:.4f}  {verdict}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
