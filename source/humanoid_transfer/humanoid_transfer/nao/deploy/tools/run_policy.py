"""Run the tracker on the robot: ramp up, crouch, follow a plan, stop, and log it all.

**First runs: robot in its harness.** The plans are a bundle's
(``nao/scripts/export_runtime.py``): ``stand`` (zero command), ``forward`` and
``turning`` (2 s standing, 10 s walking, 3 s standing). ``--max-seconds``
cuts a run short; the chest button, a fall or a hot motor stop it::

    python3 -m deploy.tools.run_policy --lola --bundle bundle --plan stand --max-seconds 5
"""

from __future__ import annotations

import argparse
import json

from ..controller import Bundle, Controller
from ..session import Phases, SafetyConfig, Session
from .common import add_backend_args, backend_from, stamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_backend_args(parser)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--plan", default="stand")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--stiffness", type=float, default=1.0)
    parser.add_argument("--fall-tilt", type=float, default=0.6)
    parser.add_argument("--temperature-limit", type=float, default=75.0)
    parser.add_argument("--pose-seconds", type=float, default=3.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    controller = Controller(bundle, args.plan)
    safety = SafetyConfig(
        fall_tilt=args.fall_tilt,
        temperature_limit=args.temperature_limit,
        run_stiffness=args.stiffness,
    )
    backend = backend_from(args)
    session = Session(controller, backend, safety, Phases(pose_s=args.pose_seconds))
    steps = None if args.max_seconds is None else int(args.max_seconds / controller.dt)
    try:
        reason = session.run(max_steps=steps)
    finally:
        output = args.output or f"logs/run_{args.plan}_{stamp()}.npz"
        session.log.save(output, plan=args.plan, bundle=str(args.bundle))
        backend.close()
    compute = session.log.rows.get("compute_s", [])
    summary = {
        "plan": args.plan,
        "stopped": reason,
        "policy_steps": len(compute),
        "compute_ms_max": 1000 * float(max(compute)) if compute else None,
        "log": str(output),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
