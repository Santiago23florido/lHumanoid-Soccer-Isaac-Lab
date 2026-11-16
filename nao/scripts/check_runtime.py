"""Replay a bundle's parity records through the robot's NumPy runtime.

For every step of every schedule, the sensor readings the simulator's estimator
took go into :class:`~humanoid_transfer.nao.deploy.controller.Controller`; its
observation and action must be the simulator's. Reports the largest
differences, and fails above ``--tolerance``.

The runtime's next input carries its own last actions. Replayed free, a policy whose
actions feed back into what it estimates (a student's adaptation module reads them)
can amplify float32-against-float64 rounding step after step: that measures the
policy's loop, not the runtime. Each schedule is therefore also replayed with the
simulator's action fed back instead (``action_max_abs_forced``), and the check fails on
that one; the free difference is reported beside it.

Usage::

    python nao/scripts/check_runtime.py nao/bundles/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.deploy.controller import Bundle, Controller  # noqa: E402


def replay(bundle: Bundle, name: str, forced: bool = False) -> dict:
    record = np.load(bundle.root / "parity" / f"{name}.npz")
    env = int(record["env"])
    controller = Controller(bundle, name)
    sim_action = record["action"][:, env].astype(np.float64)
    observations, actions = [], []
    for k in range(len(record["q"])):
        observation = controller.observe(
            record["q"][k, env], record["gravity"][k, env], record["gyro"][k, env],
            record["loads"][k, env],
        )  # fmt: skip
        observations.append(observation)
        controller.act(observation)
        actions.append(controller.last_action)
        if forced:
            controller.last_action = sim_action[k].copy()
    observation = np.stack(observations)
    action = np.stack(actions)
    sim_obs = record["observation"][:, env].astype(np.float64)
    terms = bundle.config["policy_terms"]
    per_term, start = {}, 0
    sensed = np.ones(observation.shape[-1], dtype=bool)
    for term, width in terms:
        part = slice(start, start + width)
        if term == "robot_params":
            # Not sensed: the simulator's are the truth, the robot's a placeholder the actor
            # overwrites with its estimate. The actions compare what it does with them.
            sensed[part] = False
        else:
            per_term[term] = float(np.abs(observation[:, part] - sim_obs[:, part]).max())
        start += width
    return {
        "steps": len(observation),
        "observation_max_abs": float(np.abs(observation - sim_obs)[:, sensed].max()),
        "observation_per_term": per_term,
        "action_max_abs": float(np.abs(action - sim_action).max()),
        "action_scale": float(np.abs(sim_action).max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    results = {}
    for name in bundle.plans():
        results[name] = replay(bundle, name)
        forced = replay(bundle, name, forced=True)
        results[name]["action_max_abs_forced"] = forced["action_max_abs"]
        results[name]["observation_max_abs_forced"] = forced["observation_max_abs"]
    print(json.dumps(results, indent=1))
    worst = max(r["action_max_abs_forced"] for r in results.values())
    free = max(r["action_max_abs"] for r in results.values())
    (args.bundle / "parity.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if worst > args.tolerance:
        print(f"FAILED: the runtime's actions differ from the simulator's by {worst:.2e}")
        return 1
    print(f"passed: largest action difference {worst:.2e} (replayed free: {free:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
