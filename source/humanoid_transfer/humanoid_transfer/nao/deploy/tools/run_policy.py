"""Run the tracker on the robot: ramp up, crouch, follow a plan, stop, and log it all.

**First runs: robot in its harness.** The plans are a bundle's
(``nao/scripts/export_runtime.py``): ``stand`` (zero command), ``forward`` and
``turning`` (2 s standing, 10 s walking, 3 s standing). The robot holds its
crouch until the chest button is pressed (set it on its feet first), then
follows the plan; ``--max-seconds`` cuts a run short; the chest button, a fall
or a hot motor stop it::

    python3 -m deploy.tools.run_policy --lola --bundle bundle --plan stand --max-seconds 5

A student's bundle estimates the robot's parameters as it runs (logged as
``robot_params``); ``--params`` gives it measured ones instead, from
``nao/scripts/fit_actuators.py --params`` (a parameter the file leaves out keeps the
estimate)::

    python3 -m deploy.tools.run_policy --lola --bundle bundle --plan stand --params params.json
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
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the policy right after the crouch, without waiting for the chest button.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="A student's robot parameters, measured (JSON: name to value in its own units).",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    given = None
    if args.params is not None:
        with open(args.params, encoding="utf-8") as handle:
            given = json.load(handle)
    controller = Controller(bundle, args.plan, robot_params=given)
    safety = SafetyConfig(
        fall_tilt=args.fall_tilt,
        temperature_limit=args.temperature_limit,
        run_stiffness=args.stiffness,
    )
    backend = backend_from(args)
    phases = Phases(pose_s=args.pose_seconds, wait_button=not args.no_wait)
    session = Session(controller, backend, safety, phases)
    steps = None if args.max_seconds is None else int(args.max_seconds / controller.dt)
    try:
        reason = session.run(max_steps=steps)
    finally:
        output = args.output or f"logs/run_{args.plan}_{stamp()}.npz"
        session.log.save(
            output, plan=args.plan, bundle=str(args.bundle), params=json.dumps(given or {})
        )
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
