"""How precisely a body's deployed estimator reads each foot's load, in the generator's log space.

The tracking reward scores the reached state against the plan in units of the encoder's
precision. For the loads it did so in linear units, where a foot left on the ground with
5 % of the weight while the plan has it in the air costs under one tolerance -- and that
much is already contact for the duty factor, the check that fails 21-25 of every tracker's
~24 failing probes of the family (``transfer/docs/skill_space.md`` §9.1). Scored as the
generator models loads, ``log(load + 0.01)``, the same miss costs several tolerances.

This measures that space's precision on a capture of the deployed path (``estimate``, the
kinematic estimator with the learned encoder's loads, against the true ``z*``), so the
reward's scale is measured, not chosen.

Usage::

    python transfer/scripts/load_precision.py g1/rollouts/zs_r3_6194_sym_v3_walk_kin_encoder.npz \
        --output transfer/results/load_log_precision_g1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402

EPSILON = 0.01
"""The generator's load warp, ``log(load + EPSILON)`` (``ContractGenerator.warp``)."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("--settle", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.capture, allow_pickle=True)
    load = SKILL_LAYOUT.slice_of("foot_load")
    alive = ~data["done"][args.settle :].astype(bool)
    estimate = np.clip(data["estimate"][args.settle :, :, load][alive], 0.0, None)
    true = np.clip(data["skill_state"][args.settle :, :, load][alive], 0.0, None)
    log_error = np.log(estimate + EPSILON) - np.log(true + EPSILON)
    report = {
        "capture": args.capture.as_posix(),
        "epsilon": EPSILON,
        "samples": int(true.size),
        "linear_rms": float(np.sqrt(np.mean((estimate - true) ** 2))),
        "log_rms": float(np.sqrt(np.mean(log_error**2))),
        "log_bias": float(np.mean(log_error)),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
