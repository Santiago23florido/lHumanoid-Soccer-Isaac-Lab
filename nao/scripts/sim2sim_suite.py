"""The MuJoCo sim2sim suite of a deployment bundle: every plan, friction and latency.

Runs :func:`sim2sim.run` (the robot's runtime on the MuJoCo NAO of
``nao/scripts/build_mjcf.py``, started from rest) for each plan of the bundle,
at sole friction 0.64 and 1.0 and latency 0 and 36 ms, ``--runs`` times each,
and writes falls, distance, speed and turn per setting.

Usage (the environment with MuJoCo)::

    python nao/scripts/sim2sim_suite.py nao/bundles/<name> --output nao/results/sim2sim_<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from humanoid_transfer.nao.deploy.controller import Bundle  # noqa: E402
from sim2sim import MujocoNao, run  # noqa: E402

FRICTIONS = (0.64, 1.0)
LATENCIES_MS = (0.0, 36.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--mjcf", type=Path, default=ROOT / "nao" / "assets" / "generated" / "nao_mujoco.xml"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--plans", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="A student: measured robot parameters (JSON, own units) instead of its estimate.",
    )
    parser.add_argument(
        "--tell-latency",
        action="store_true",
        help="A student: given each setting's latency as delay_ms, the rest as --params says.",
    )
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    given = {} if args.params is None else json.loads(args.params.read_text(encoding="utf-8"))
    plans = args.plans or bundle.plans()
    rows = []
    for plan in plans:
        for friction in FRICTIONS:
            for latency in LATENCIES_MS:
                nao = MujocoNao(args.mjcf, bundle, latency / 1000.0, friction)
                params = {**given, "delay_ms": latency} if args.tell_latency else given
                runs = [
                    run(nao, bundle, plan, None, 0.02, seed, params or None)
                    for seed in range(args.runs)
                ]
                row = {
                    "plan": plan,
                    "friction": friction,
                    "latency_ms": latency,
                    "runs": len(runs),
                    "falls": sum(r["fell_at_s"] is not None for r in runs),
                    "forward_m": float(np.mean([r["forward_m"] for r in runs])),
                    "distance_m": float(np.mean([r["distance_m"] for r in runs])),
                    "turned_rad": float(np.mean([r["turned_rad"] for r in runs])),
                    "commanded_turn_rad": float(runs[0]["commanded_turn_rad"]),
                    "results": runs,
                }
                if params:
                    row["given"] = params
                if "robot_params" in runs[0]:
                    row["robot_params"] = {
                        name: float(np.median([r["robot_params"][name]["median"] for r in runs]))
                        for name in runs[0]["robot_params"]
                    }
                rows.append(row)
                acted = row.get("robot_params", {})
                print(
                    f"{plan:8s} friction {friction:4.2f} latency {latency:4.0f} ms: falls "
                    f"{row['falls']}/{row['runs']}  forward {row['forward_m']:+.2f} m  turned "
                    f"{row['turned_rad']:+.2f} of {row['commanded_turn_rad']:+.2f} rad"
                    + "".join(f"  {k.split('_')[0]} {v:.2f}" for k, v in acted.items())
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"bundle": str(args.bundle), "rows": rows}, indent=1), encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
