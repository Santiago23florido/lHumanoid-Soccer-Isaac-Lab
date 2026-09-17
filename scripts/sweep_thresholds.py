"""Find where each controller fails, per direction, against the theory.

The zero-step capturability bound says how fast the centre of mass can be
moving before the robot must take a step. It is direction dependent, because
the support polygon is: for this NAO, 0.548 m/s forward, 0.443 m/s backward,
0.620 m/s lateral. And it is derived from the linear inverted pendulum, which
assumes centroidal angular momentum stays constant.

A learned policy does not know that assumption exists. This script measures
whether it breaks it, in which directions, and by how much -- which is a
question with a number for an answer, unlike "does reinforcement learning
work".

Every trial runs on the same environment instance with a **fixed** push
direction and an **exact** magnitude, so a threshold means what it says.

Usage::

    python scripts/sweep_thresholds.py --headless --num-envs 64 \
        --policy logs/rsl_rl/nao_stand/<run>/exported/policy.pt
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_soccer_lab"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep push thresholds per direction.")
parser.add_argument("--num-envs", type=int, default=64, help="Environments per trial.")
parser.add_argument("--steps", type=int, default=250, help="Control steps per trial.")
parser.add_argument("--policy", type=Path, default=None, help="TorchScript policy to include.")
parser.add_argument(
    "--magnitudes",
    type=float,
    nargs="+",
    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    help="Push magnitudes in m/s to test.",
)
parser.add_argument(
    "--results-file", type=Path, default=None, help="Write the sweep as JSON."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from humanoid_soccer_lab.assets.nao_paths import describe_missing_meshes, meshes_are_available

if not meshes_are_available():
    print("\n" + describe_missing_meshes() + "\n", file=sys.stderr)
    raise SystemExit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_soccer_lab.tasks  # noqa: F401  (registers the task ids)
from humanoid_soccer_lab.assets import nao_kinematics as nk
from humanoid_soccer_lab.tasks.direct.nao_stand import TASK_ID
from humanoid_soccer_lab.tasks.direct.nao_stand.nao_stand_env_cfg import NaoStandEnvCfg

DIRECTIONS = {
    "forward": (0.0, "forward"),
    "backward": (math.pi, "backward"),
    "lateral": (math.pi / 2.0, "lateral"),
}
"""Push headings relative to the robot's facing, and the bound each is tested against."""


def fall_free_rate(env, actions_fn, steps: int) -> float:
    """Fraction of environments that never terminated during the trial."""
    obs, _ = env.reset()
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    ever_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for _ in range(steps):
        actions = actions_fn(obs, num_envs, device)
        obs, _, terminated, _, _ = env.step(actions)
        ever_fell |= terminated

    return float((~ever_fell).float().mean())


def main() -> int:
    cfg = NaoStandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.push_exact_magnitude = True
    cfg.push_curriculum_steps = 1
    # A push every second, so a 250-step trial delivers two or three and each
    # has a full second to be recovered from. The interval is deterministic
    # whenever push_exact_magnitude is set, so every environment gets the same
    # treatment and the threshold means what it says.
    cfg.push_interval_s = 1.0

    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    bounds = nk.capturable_com_velocity()

    controllers: dict[str, object] = {
        "joint PD": lambda obs, n, d: torch.zeros(n, cfg.action_space, device=d)
    }
    if args_cli.policy is not None:
        module = torch.jit.load(str(args_cli.policy), map_location=inner.device)
        module.eval()

        def learned(obs, n, d):
            with torch.inference_mode():
                return module(obs["policy"]).clone()

        controllers["PPO policy"] = learned

    print("\n" + "=" * 78)
    print("PUSH THRESHOLD SWEEP -- measured against the capturability bound")
    print("=" * 78)
    print(f"  environments per trial : {inner.num_envs}")
    print(f"  steps per trial        : {args_cli.steps}")
    print("  theoretical bounds (zero-step, ankle strategy only):")
    for name, (_, key) in DIRECTIONS.items():
        print(f"    {name:9s} {bounds[key]:.3f} m/s")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for controller, policy in controllers.items():
        results[controller] = {}
        print(f"\n  --- {controller} ---")
        header = "  magnitude  " + "".join(f"{d:>12s}" for d in DIRECTIONS)
        print(header)
        for magnitude in args_cli.magnitudes:
            row = f"  {magnitude:8.2f}   "
            for name, (angle, _) in DIRECTIONS.items():
                cfg.push_direction_rad = angle
                cfg.push_velocity_initial = magnitude
                cfg.push_velocity_final = magnitude
                rate = fall_free_rate(env, policy, args_cli.steps)
                results[controller].setdefault(name, {})[f"{magnitude:.2f}"] = rate
                row += f"{rate:11.1%} "
            print(row)

    print("\n" + "=" * 78)
    print("WHERE EACH CONTROLLER CROSSES 50% FALL-FREE")
    print("=" * 78)
    print(f"  {'direction':10s} {'bound':>8s} {'joint PD':>10s} {'policy':>10s} {'ratio':>8s}")
    print("-" * 60)
    summary: dict[str, dict[str, float]] = {}
    for name, (_, key) in DIRECTIONS.items():
        bound = bounds[key]
        row = {"bound": bound}
        for controller in results:
            crossing = float("nan")
            previous = None
            for magnitude in args_cli.magnitudes:
                rate = results[controller][name][f"{magnitude:.2f}"]
                if rate < 0.5:
                    crossing = magnitude if previous is None else previous
                    break
                previous = magnitude
            else:
                crossing = args_cli.magnitudes[-1]
            row[controller] = crossing
        summary[name] = row
        pd_value = row.get("joint PD", float("nan"))
        policy_value = row.get("PPO policy", float("nan"))
        ratio = policy_value / bound if bound else float("nan")
        print(
            f"  {name:10s} {bound:8.3f} {pd_value:10.2f} {policy_value:10.2f} {ratio:7.2f}x"
        )
    print("-" * 60)
    print("  ratio = last magnitude the policy survives, over the LIPM bound.")
    print("  Above 1.0 means it beat a limit derived assuming constant angular")
    print("  momentum, which is only possible by not keeping it constant.")
    print("=" * 78 + "\n")

    if args_cli.results_file is not None:
        args_cli.results_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.results_file.write_text(
            json.dumps(
                {
                    "num_envs": inner.num_envs,
                    "steps": args_cli.steps,
                    "bounds": {k: bounds[v[1]] for k, v in DIRECTIONS.items()},
                    "sweep": results,
                    "crossing": summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  results written to {args_cli.results_file}\n")

    env.close()
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close(skip_cleanup=True)
    raise SystemExit(exit_code)
