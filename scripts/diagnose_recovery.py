"""Check what a controller is actually doing when it survives a push.

The threshold sweep compares the push magnitude against the zero-step
capturability bound and reports a ratio. That comparison is only meaningful if
four assumptions hold, and none of them were checked. This script checks them.

**1. The push is not the centre-of-mass velocity.** The push writes a velocity
onto the articulation root, which carries about 20% of the mass. The bound is
about the velocity of the whole-body centre of mass. Reporting a ratio of one
against the other compares different quantities, so the actual jump is measured
here.

**2. The bound assumes a constant centre-of-mass height.** Since
``omega_0 = sqrt(g/z_c)``, crouching *raises* the bound. A policy that drops its
centre of mass beats ``omega_0 d`` without touching angular momentum, so the
height has to be tracked to rule that out.

**3. The bound assumes a fixed support polygon.** Widening the stance grows
``d`` and moves the bound for the same reason.

**4. The bound is a zero-step bound.** Nothing in the task forbids stepping. If
a foot moves, the robot is doing one-step recovery and the zero-step bound is
simply the wrong comparison -- it would be exceeded trivially and for an
uninteresting reason.

Only with all four checked does "exceeds the bound" support "changes angular
momentum".

Usage::

    python scripts/diagnose_recovery.py --headless --push 0.7 --direction backward \
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

parser = argparse.ArgumentParser(description="Diagnose how a controller survives a push.")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--push", type=float, default=0.7, help="Push magnitude in m/s.")
parser.add_argument(
    "--direction", choices=("forward", "backward", "lateral"), default="backward"
)
parser.add_argument("--policy", type=Path, default=None, help="TorchScript policy, or PD.")
parser.add_argument("--steps", type=int, default=200, help="Steps to watch after the push.")
parser.add_argument("--results-file", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from humanoid_soccer_lab.nao.assets.nao_paths import describe_missing_meshes, meshes_are_available

if not meshes_are_available():
    print("\n" + describe_missing_meshes() + "\n", file=sys.stderr)
    raise SystemExit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import humanoid_soccer_lab.tasks  # noqa: F401  (registers the task ids)
from humanoid_soccer_lab.nao.assets import nao_kinematics as nk
from humanoid_soccer_lab.nao.tasks.nao_stand import TASK_ID
from humanoid_soccer_lab.nao.tasks.nao_stand.nao_stand_env_cfg import NaoStandEnvCfg

ANGLES = {"forward": 0.0, "backward": math.pi, "lateral": math.pi / 2.0}

FOOT_MOVED_M = 0.02
"""Horizontal foot displacement counted as a step, in metres.

Two centimetres is well past solver jitter and contact creep, and well under
the length of any deliberate step.
"""


def whole_body_com(robot, device):
    """Whole-body centre of mass position and velocity, in world frame."""
    masses = robot.data.default_mass.to(device)
    weights = (masses / masses.sum(dim=1, keepdim=True)).unsqueeze(-1)
    return (
        (robot.data.body_com_pos_w * weights).sum(dim=1),
        (robot.data.body_com_lin_vel_w * weights).sum(dim=1),
    )


def main() -> int:
    cfg = NaoStandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.push_exact_magnitude = True
    cfg.push_curriculum_steps = 1
    cfg.push_velocity_initial = args_cli.push
    cfg.push_velocity_final = args_cli.push
    cfg.push_direction_rad = ANGLES[args_cli.direction]
    # One push, far enough in that the robot has settled, and nothing after it.
    cfg.push_interval_s = 0.5

    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    robot = inner._robot

    if args_cli.policy is not None:
        module = torch.jit.load(str(args_cli.policy), map_location=device)
        module.eval()

        def act(obs):
            with torch.inference_mode():
                return module(obs["policy"]).clone()

        label = "PPO policy"
    else:

        def act(obs):
            return torch.zeros(inner.num_envs, cfg.action_space, device=device)

        label = "joint PD"

    obs, _ = env.reset()
    feet = inner._foot_ids

    # Settle, then catch the push as it lands.
    push_step = int(cfg.push_interval_s / inner.step_dt)
    com_before = com_after = None
    foot_reference = None
    alive = torch.ones(inner.num_envs, dtype=torch.bool, device=device)

    com_height_min = torch.full((inner.num_envs,), 1e3, device=device)
    stance_max = torch.zeros(inner.num_envs, device=device)
    foot_shift_max = torch.zeros(inner.num_envs, device=device)
    dcm_max = torch.zeros(inner.num_envs, device=device)

    total = push_step + args_cli.steps
    for step in range(total):
        if step == push_step - 1:
            com_before = whole_body_com(robot, device)[1].clone()
            foot_reference = robot.data.body_pos_w[:, feet, :2].clone()

        obs, _, terminated, truncated, _ = env.step(act(obs))
        alive &= ~(terminated | truncated)

        if step == push_step - 1:
            com_after = whole_body_com(robot, device)[1].clone()

        if step < push_step:
            continue

        position, _ = whole_body_com(robot, device)
        ground = inner._terrain.env_origins[:, 2]
        com_height_min = torch.minimum(com_height_min, position[:, 2] - ground)

        foot_xy = robot.data.body_pos_w[:, feet, :2]
        stance = (foot_xy[:, 0, :] - foot_xy[:, 1, :]).norm(dim=-1)
        stance_max = torch.maximum(stance_max, stance)
        shift = (foot_xy - foot_reference).norm(dim=-1).max(dim=1).values
        foot_shift_max = torch.maximum(foot_shift_max, shift)
        dcm_max = torch.maximum(dcm_max, inner._dcm_b.norm(dim=-1))

    survived = alive
    n = int(survived.sum())
    bounds = nk.capturable_com_velocity()
    nominal_stance = 2.0 * nk.STANCE_HALF_WIDTH

    def among_survivors(values):
        return float(values[survived].mean()) if n else float("nan")

    jump = (com_after - com_before)[:, :2].norm(dim=-1)

    print("\n" + "=" * 78)
    print(f"RECOVERY DIAGNOSTICS -- {label}, {args_cli.push:.2f} m/s {args_cli.direction}")
    print("=" * 78)
    print(f"  environments             : {inner.num_envs}")
    print(f"  survived                 : {n} ({n / inner.num_envs:.1%})")
    if not n:
        print("\n  Nothing survived; the checks below need survivors.")
        env.close()
        return 0

    print("\n  1. IS THE PUSH THE CENTRE-OF-MASS VELOCITY?")
    print(f"     commanded push        : {args_cli.push:.3f} m/s (onto the root)")
    print(f"     actual CoM jump       : {among_survivors(jump):.3f} m/s")
    print(f"     ratio                 : {among_survivors(jump) / args_cli.push:.2f}")
    print("     The bound is about the whole-body CoM, so this is the number that")
    print("     should be compared against it -- not the commanded push.")

    print("\n  2. DOES IT CROUCH? (a lower CoM raises the bound)")
    nominal_height = nk.com_height_above_soles(nk.NOMINAL_STAND_JOINT_POS)
    print(f"     nominal CoM height    : {nominal_height:.4f} m")
    print(f"     minimum during recovery: {among_survivors(com_height_min):.4f} m")
    drop = nominal_height - among_survivors(com_height_min)
    if drop > 0.001:
        raised = nk.lipm_omega(max(among_survivors(com_height_min), 1e-6))
        nominal_omega = nk.lipm_omega(nominal_height)
        print(f"     drop                  : {drop * 1e3:+.1f} mm")
        print(
            f"     omega_0 would rise to : {raised:.3f} rad/s "
            f"(nominal {nominal_omega:.3f}), raising the bound by "
            f"{raised / nominal_omega - 1.0:+.1%}"
        )

    print("\n  3. DOES IT WIDEN THE STANCE? (a bigger polygon raises the bound)")
    print(f"     nominal stance width  : {nominal_stance:.4f} m")
    print(f"     maximum during recovery: {among_survivors(stance_max):.4f} m")

    print("\n  4. DOES IT STEP? (a zero-step bound would then be the wrong one)")
    stepped = (foot_shift_max > FOOT_MOVED_M) & survived
    worst = float(foot_shift_max[survived].max()) if n else float("nan")
    # Mean and worst, because they answer different questions and the mean on
    # its own misleads: one environment travelling 300 mm among thirty that
    # moved 4 mm averages to 14 mm and reads as "nobody stepped". This label
    # said "max" while reporting the mean, which is how that went unnoticed.
    print(f"     foot travel, mean     : {among_survivors(foot_shift_max) * 1e3:.1f} mm")
    print(f"     foot travel, worst    : {worst * 1e3:.1f} mm")
    print(f"     survivors that stepped: {int(stepped.sum())} of {n}")
    print(f"     threshold used        : {FOOT_MOVED_M * 1e3:.0f} mm")

    print("\n  5. DID THE DIVERGENT COMPONENT LEAVE THE POLYGON?")
    print(f"     peak |DCM|            : {among_survivors(dcm_max) * 1e3:.1f} mm")
    print(f"     forward polygon edge  : {nk.SUPPORT_POLYGON_X[1] * 1e3:.1f} mm")
    print(f"     zero-step bound here  : {bounds[args_cli.direction]:.3f} m/s")
    print("=" * 78 + "\n")

    if args_cli.results_file is not None:
        args_cli.results_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.results_file.write_text(
            json.dumps(
                {
                    "controller": label,
                    "push": args_cli.push,
                    "direction": args_cli.direction,
                    "survived": n,
                    "num_envs": inner.num_envs,
                    "com_velocity_jump": among_survivors(jump),
                    "com_height_min": among_survivors(com_height_min),
                    "stance_max": among_survivors(stance_max),
                    "foot_displacement_mean": among_survivors(foot_shift_max),
                    "foot_displacement_worst": worst,
                    "stepped": int(stepped.sum()),
                    "dcm_peak": among_survivors(dcm_max),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  written to {args_cli.results_file}\n")

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
