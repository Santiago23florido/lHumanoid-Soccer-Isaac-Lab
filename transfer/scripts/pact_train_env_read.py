"""Read the DAgger rounds run in the trackers' training task (``pact_dagger.sh``, env train).

Round 0 is each body's tracker driving its ``<Body>Track-v0`` (the reference); rounds 1..k
are PACT driving it, the tracker labelling. Per body and round: terminations (falls and
plans left, ``contract_left``) over the capture, their bootstrap SE over environments,
the forward speed (``V``) and the median contract error, as the capture's summary line
prints them, recomputed from the capture.

Usage::

    python transfer/scripts/pact_train_env_read.py --tag train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402

BODIES = ("g1", "nao", "h1", "cassie")


def read(tag: str, draws: int, seed: int) -> dict:
    report: dict = {"tag": tag, "bootstrap": draws, "bodies": {}}
    rng = np.random.default_rng(seed)
    for body in BODIES:
        rounds = sorted(
            REPO.glob(f"logs/pact/dagger/{tag}/r*/{body}.npz"),
            key=lambda p: int(p.parent.name[1:]),
        )
        if not rounds:
            continue
        entry: dict = {}
        for path in rounds:
            data = np.load(path)
            per_env = data["terminated"].sum(0).astype(float)
            boots = per_env[rng.integers(0, len(per_env), (draws, len(per_env)))].sum(1)
            speed = SKILL_LAYOUT.unpack(data["skill_state"])["com_velocity"][..., 0].mean()
            entry[int(path.parent.name[1:])] = {
                "capture": str(path.relative_to(REPO)).replace("\\", "/"),
                "driver": "tracker" if float(data["teacher_driven"].mean()) == 1.0 else "pact",
                "terminations": float(per_env.sum()),
                "terminations_se": float(boots.std(ddof=1)),
                "forward_speed_v": float(speed),
                "median_contract_error": float(np.median(data["contract_error"])),
            }
        report["bodies"][body] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="train")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = read(args.tag, args.bootstrap, args.seed)
    output = args.output or REPO / f"transfer/results/icml_gate_d1_{args.tag}_env.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for body, entry in report["bodies"].items():
        cells = "  ".join(
            f"r{k} {r['driver'][0]}: {r['terminations']:.0f} ± {r['terminations_se']:.0f}, "
            f"{r['forward_speed_v']:.3f} V, err {r['median_contract_error']:.2f}"
            for k, r in entry.items()
        )
        print(f"{body:7s} {cells}")
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
