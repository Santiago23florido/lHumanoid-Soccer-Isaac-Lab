"""Check the reconstructed ground reactions against Newton and the contact sensor, in simulation.

:mod:`~humanoid_transfer.transfer.tasks.tracking.balance` rebuilds each foot's
ground force and centre of pressure from the ankle's joint wrench. Before a
critic or a physics head reads them, two closures must hold on the body:

* **Newton.** The feet's forces sum to ``m (c̈ + g)``, the whole body's mass
  times its CoM's acceleration plus gravity (the only other external force is
  gravity) -- in all three axes, the horizontal ones included, which the
  contact sensor never reports.
* **The sensor.** Each foot's vertical force agrees with the contact sensor's
  normal force.

A tracker drives the body (``--driver-policy``, a TorchScript export); the
residuals are reported in units of the body's weight, over the steps where the
episode runs on.

Usage::

    python transfer/scripts/check_balance.py --headless --task-id G1Track-Play-v0 \
        --driver-policy logs/rsl_rl/g1_track/<run>/exported/model_6194.pt \
        --output transfer/results/check_balance_g1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task-id", required=True)
parser.add_argument("--driver-policy", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--settle-steps", type=int, default=50)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from humanoid_transfer.transfer.tasks.tracking.balance import GRAVITY, foot_reactions  # noqa: E402

COMMAND = "base_velocity"


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args_cli.task_id, num_envs=args_cli.num_envs)
    cfg.commands.base_velocity.use_encoder = False
    env = gym.make(args_cli.task_id, cfg=cfg)
    inner = env.unwrapped
    term = inner.command_manager.get_term(COMMAND)
    robot = inner.scene["robot"]
    sensor = inner.scene["contact_forces"]
    driver = torch.jit.load(str(args_cli.driver_policy), map_location=inner.device).eval()
    masses = term.body.masses
    weight = float(masses.sum()) * GRAVITY

    obs, _ = env.reset()
    rows: dict[str, list[np.ndarray]] = {"newton": [], "sensor": [], "valid": [], "loaded": []}
    for step in range(args_cli.settle_steps + args_cli.steps):
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(driver(obs["policy"]))
        if step < args_cli.settle_steps:
            continue
        reaction = foot_reactions(term)
        acceleration = torch.einsum("b,nbi->ni", masses, robot.data.body_com_acc_w[..., :3])
        expected = acceleration.clone()
        expected[:, 2] += float(masses.sum()) * GRAVITY
        total = reaction["force"].sum(dim=1)
        normal = sensor.data.net_forces_w[:, term.sensor_ids, 2]
        rows["newton"].append(((total - expected) / weight).cpu().numpy())
        rows["sensor"].append(((reaction["force"][..., 2] - normal) / weight).cpu().numpy())
        rows["valid"].append((~(terminated | truncated)).cpu().numpy())
        rows["loaded"].append(reaction["loaded"].cpu().numpy())
    newton = np.stack(rows["newton"])
    sensor_gap = np.stack(rows["sensor"])
    valid = np.stack(rows["valid"])
    loaded = np.stack(rows["loaded"])
    report = {
        "task": args_cli.task_id,
        "driver": str(args_cli.driver_policy),
        "steps": int(valid.sum()),
        "unit": "body weight",
        "newton_rms": {
            axis: float(np.sqrt(np.mean(newton[valid][:, i] ** 2))) for i, axis in enumerate("xyz")
        },  # fmt: skip
        "newton_bias": {axis: float(np.mean(newton[valid][:, i])) for i, axis in enumerate("xyz")},
        "sensor_rms": float(np.sqrt(np.mean(sensor_gap[valid] ** 2))),
        "sensor_rms_loaded": float(np.sqrt(np.mean(sensor_gap[valid][loaded[valid]] ** 2))),
        "loaded_share": float(loaded[valid].mean()),
    }
    print(json.dumps(report, indent=2))
    if args_cli.output is not None:
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
