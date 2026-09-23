"""Score every controller for the standing task in one run, on one protocol.

The joint PD, the capture-point controller and a trained policy all drive the
same 19 joint offsets, so they can be scored against the same environment
instance: same reset distribution, same friction and mass draws, same push
schedule, same fall accounting.

Building the scene once instead of three times is not only faster. It removes
the possibility that a difference between controllers is really a difference
between two randomisation draws, which is exactly the kind of error that makes
a policy look better than the baseline it never beat.

Usage::

    python scripts/compare_controllers.py --headless --num-envs 128 --steps 600 \
        --policy logs/rsl_rl/nao_stand/<run>/exported/policy.pt
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare balance controllers on one protocol.")
parser.add_argument("--num-envs", type=int, default=128, help="Environments. Default 128.")
parser.add_argument("--steps", type=int, default=600, help="Control steps per trial. Default 600.")
parser.add_argument(
    "--policy",
    type=Path,
    default=None,
    help="TorchScript policy from rsl-rl, e.g. logs/.../exported/policy.pt. Optional.",
)
parser.add_argument(
    "--ablate-arms",
    action="store_true",
    help=(
        "Also score the policy with its eight arm joints frozen at nominal. "
        "The drop is how much of the policy's advantage comes from changing "
        "centroidal angular momentum rather than from the ankles and hips."
    ),
)
parser.add_argument(
    "--no-pushes",
    action="store_true",
    help="Measure a quiet stance instead of the perturbation protocol.",
)
parser.add_argument(
    "--protocol",
    choices=("directional", "absolute"),
    default="directional",
    help=(
        "directional pushes at a fraction of what is recoverable along each "
        "heading, which is the distribution the policy trains on. absolute "
        "pushes at a fixed speed in every direction, which includes headings "
        "no zero-step controller can survive. Default directional."
    ),
)
parser.add_argument(
    "--push-speed",
    type=float,
    default=None,
    help="Speed for --protocol absolute, in m/s. Defaults to the forward bound.",
)
parser.add_argument(
    "--results-file",
    type=Path,
    default=None,
    help="Write the table as JSON for the report.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from humanoid_transfer.nao.assets.nao_paths import describe_missing_meshes, meshes_are_available

if not meshes_are_available():
    print("\n" + describe_missing_meshes() + "\n", file=sys.stderr)
    raise SystemExit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

import humanoid_transfer.tasks  # noqa: F401  (registers the task ids)
from humanoid_transfer.nao.assets import nao_kinematics as nk
from humanoid_transfer.nao.controllers import DcmBalanceController
from humanoid_transfer.nao.tasks.nao_stand import TASK_ID
from humanoid_transfer.nao.tasks.nao_stand.nao_stand_env_cfg import NaoStandEnvCfg


def run_trial(env, actions_fn, steps: int) -> dict[str, float]:
    """Step the environment under one controller and summarise the outcome.

    ``actions_fn`` receives the observation the environment just returned, so
    a learned policy reads exactly what it would read in deployment and the
    environment is stepped once per control step, not queried twice.

    Falls and timeouts are counted separately. The episode length buffer is
    randomised at reset so the batch does not terminate in lockstep, which means
    a fixed number of steps always contains truncations; charging those to the
    controller would understate every one of them equally but by an amount that
    depends on the run.
    """
    obs, _ = env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    ever_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    falls = torch.zeros(num_envs, device=device)
    timeouts = torch.zeros(num_envs, device=device)
    reward_total = torch.zeros(num_envs, device=device)
    torque_sum = torch.zeros(num_envs, device=device)
    saturated_steps = torch.zeros(num_envs, device=device)
    action_rate_sum = torch.zeros(num_envs, device=device)
    previous = torch.zeros(num_envs, env.unwrapped.cfg.action_space, device=device)

    robot = env.unwrapped._robot
    ankles, _ = robot.find_joints(["LAnklePitch", "RAnklePitch"], preserve_order=True)
    effort = next(j.effort for j in nk.load_model().joints if j.name == "LAnklePitch")

    for _ in range(steps):
        actions = actions_fn(obs, num_envs, device)
        obs, reward, terminated, truncated, _ = env.step(actions)
        reward_total += reward
        ever_fell |= terminated
        falls += terminated.float()
        timeouts += truncated.float()
        used = robot.data.applied_torque[:, ankles].abs().max(dim=1).values / effort
        torque_sum += used
        saturated_steps += (used > 0.99).float()
        action_rate_sum += (actions - previous).abs().mean(dim=1)
        previous = actions

    return {
        "fall_free_rate": float((~ever_fell).float().mean()),
        "falls": int(falls.sum()),
        "timeouts": int(timeouts.sum()),
        "mean_return": float(reward_total.mean()),
        "mean_ankle_torque": float((torque_sum / steps).mean()),
        "saturated_fraction": float((saturated_steps / steps).mean()),
        "mean_action_rate": float((action_rate_sum / steps).mean()),
    }


def joint_pd_policy(cfg):
    """A zero action already is the nominal-posture hold."""
    return lambda obs, n, d: torch.zeros(n, cfg.action_space, device=d)


def dcm_policy(inner, cfg):
    """Capture-point feedback, expressed in the environment's action space."""
    frames = nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)
    sole_mid = 0.5 * (frames["l_sole"][:3, 3] + frames["r_sole"][:3, 3])
    nominal_com = nk.center_of_mass(nk.NOMINAL_STAND_JOINT_POS) - sole_mid

    controller = DcmBalanceController(
        omega=cfg.lipm_omega,
        nominal_joint_pos=inner._robot.data.default_joint_pos.clone(),
        joint_index={name: i for i, name in enumerate(inner._robot.joint_names)},
        polygon_x=cfg.support_polygon_x,
        polygon_y=cfg.support_polygon_y,
        foot_polygon_y=nk.SUPPORT_POLYGON_Y_SINGLE,
        dcm_reference=(float(nominal_com[0]), float(nominal_com[1])),
    )

    def policy(obs, num_envs: int, device: str) -> torch.Tensor:
        robot = inner._robot
        masses = robot.data.default_mass.to(device)
        weights = (masses / masses.sum(dim=1, keepdim=True)).unsqueeze(-1)
        com_pos = (robot.data.body_com_pos_w * weights).sum(dim=1)
        com_vel = (robot.data.body_com_lin_vel_w * weights).sum(dim=1)

        wrench = robot.data.body_incoming_joint_wrench_b[:, inner._foot_ids, :]
        quats = robot.data.body_quat_w[:, inner._foot_ids, :]
        forces = math_utils.quat_apply(quats, wrench[..., :3])
        foot_mass = robot.data.default_mass[:, inner._foot_ids].to(device)
        load = -forces[..., 2] + foot_mass * nk.GRAVITY

        targets = controller.compute(
            com_pos_w=com_pos,
            com_vel_w=com_vel,
            foot_pos_w=robot.data.body_pos_w[:, inner._foot_ids, :],
            foot_normal_force=load,
            joint_pos=robot.data.joint_pos,
            joint_vel=robot.data.joint_vel,
            stiffness=robot.data.joint_stiffness,
            damping=robot.data.joint_damping,
            base_quat_w=robot.data.root_quat_w,
        )
        offset = targets[:, inner._actuated_ids] - inner._nominal_joint_pos
        return offset / cfg.action_scale

    return policy


def learned_policy(inner, path: Path, freeze_arms: bool = False):
    """The TorchScript actor rsl-rl exports, normaliser included.

    With ``freeze_arms`` the eight shoulder and elbow components of the action
    are zeroed, which holds those joints at the nominal posture while the legs
    still follow the policy. That is the ablation: the arms are the only way
    to change centroidal angular momentum once the pressure centre saturates
    at the edge of the foot, so whatever the policy loses here is what it was
    winning from that strategy.

    Note this scores the *unmodified* policy in a crippled body rather than a
    policy trained without arms. It measures how much the learned behaviour
    depends on the arms, not how well the task could be solved without them.
    """
    module = torch.jit.load(str(path), map_location=inner.device)
    module.eval()

    mask = torch.ones(len(nk.ACTUATED_JOINTS), device=inner.device)
    if freeze_arms:
        for name in nk.ARM_ACTUATED_JOINTS:
            mask[nk.ACTUATED_JOINTS.index(name)] = 0.0

    def policy(obs, num_envs: int, device: str) -> torch.Tensor:
        # The observation the environment just returned, not a fresh one.
        # Re-deriving it would call _get_observations a second time per step,
        # which recomputes the balance quantities and touches the previous-
        # action buffer the action-rate penalty reads. It happens to come out
        # the same today; relying on that is not worth the fragility.
        with torch.inference_mode():
            return module(obs["policy"]).clone() * mask

    return policy


def main() -> int:
    cfg = NaoStandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    # Skip the ramp either way: the question is what each controller can
    # already do, not how it got there.
    cfg.push_curriculum_steps = 1
    if args_cli.no_pushes:
        cfg.push_fraction_initial = 0.0
        cfg.push_fraction_final = 0.0
        cfg.push_exact_magnitude = False
        protocol = "off"
    elif args_cli.protocol == "absolute":
        # One speed in every direction. Backward that is 124% of the bound, so
        # part of this protocol is unwinnable for everyone -- fair between
        # controllers, but it compresses the difference between them.
        cfg.push_exact_magnitude = True
        speed = args_cli.push_speed or nk.capturable_com_velocity()["forward"]
        cfg.push_velocity_initial = speed
        cfg.push_velocity_final = speed
        protocol = f"absolute, {speed:.3f} m/s in every direction"
    else:
        # A fraction of what is recoverable along each heading. This is the
        # distribution the policy is trained on, and every push in it is
        # survivable by something.
        cfg.push_exact_magnitude = False
        cfg.push_fraction_initial = cfg.push_fraction_final
        protocol = f"directional, {cfg.push_fraction_final:.2f} of the bound per heading"

    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped

    controllers = {
        "joint PD": joint_pd_policy(cfg),
        "capture-point PD": dcm_policy(inner, cfg),
    }
    if args_cli.policy is not None:
        controllers["PPO policy"] = learned_policy(inner, args_cli.policy)
        if args_cli.ablate_arms:
            controllers["PPO, arms frozen"] = learned_policy(
                inner, args_cli.policy, freeze_arms=True
            )

    print("\n" + "=" * 78)
    print("BALANCE CONTROLLER COMPARISON")
    print("=" * 78)
    print(f"  environments           : {inner.num_envs}")
    print(f"  control steps          : {args_cli.steps}")
    print(f"  control rate           : {1.0 / inner.step_dt:.0f} Hz")
    print(f"  pushes                 : {protocol}")
    print("  Every controller is scored on this same environment instance.")

    results: dict[str, dict[str, float]] = {}
    for name, policy in controllers.items():
        print(f"\n  running {name}...")
        results[name] = run_trial(env, policy, args_cli.steps)

    print("\n" + "=" * 78)
    print(
        f"{'controller':20s} {'fall-free':>10s} {'falls':>7s} {'return':>8s} "
        f"{'ankle':>7s} {'sat':>6s} {'d(a)':>7s}"
    )
    print("-" * 78)
    for name, row in results.items():
        print(
            f"{name:20s} {row['fall_free_rate']:9.1%} {row['falls']:7d} "
            f"{row['mean_return']:8.2f} {row['mean_ankle_torque']:6.1%} "
            f"{row['saturated_fraction']:5.1%} {row['mean_action_rate']:7.4f}"
        )

    best = max(results, key=lambda k: results[k]["fall_free_rate"])
    print("-" * 78)
    print("  ankle = mean torque used, sat = fraction of steps saturated,")
    print("  d(a) = mean action change per step, which is what chattering shows up as.")
    print(f"  best: {best} at {results[best]['fall_free_rate']:.1%} fall-free")
    if "PPO, arms frozen" in results:
        cost = (
            results["PPO policy"]["fall_free_rate"]
            - results["PPO, arms frozen"]["fall_free_rate"]
        )
        print(f"  freezing the arms costs the policy {cost:+.1%} fall-free")
    if "PPO policy" in results:
        margin = results["PPO policy"]["fall_free_rate"] - results["joint PD"]["fall_free_rate"]
        verdict = "beats" if margin > 0 else "does NOT beat"
        print(f"  the policy {verdict} the joint PD by {margin:+.1%}")
    print("=" * 78 + "\n")

    if args_cli.results_file is not None:
        payload = {
            "num_envs": inner.num_envs,
            "steps": args_cli.steps,
            "protocol": protocol,
            "results": results,
        }
        args_cli.results_file.parent.mkdir(parents=True, exist_ok=True)
        args_cli.results_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
