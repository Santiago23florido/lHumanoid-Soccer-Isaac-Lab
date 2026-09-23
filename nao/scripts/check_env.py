"""Smoke-test the NAO standing environment without training anything.

Instantiates ``NaoStand-Direct-v0``, steps it, and checks the things that are
easy to get silently wrong in a Direct RL environment:

* the observation and state tensors really have the widths the config declares;
* the joints the policy commands are the ones it thinks it commands;
* a **zero action holds the posture**, which is the load-bearing claim of the
  whole design -- the action is an offset on the nominal pose, so an untrained
  policy starts from the stabilising baseline rather than from a rag doll;
* a random-action policy does noticeably worse, so the reward can tell them
  apart and there is something for learning to actually do;
* the reward terms and termination causes are finite and sane.

This is a correctness check, not a benchmark. It exits non-zero on failure so
it can be used as a regression gate before spending GPU hours on training.

Usage::

    python scripts/check_nao_stand_env.py --headless --device cuda:0
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
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

parser = argparse.ArgumentParser(description="Smoke-test the NAO standing environment.")
parser.add_argument("--num-envs", type=int, default=16, help="Environments to create. Default 16.")
parser.add_argument("--steps", type=int, default=300, help="Control steps per trial. Default 300.")
parser.add_argument(
    "--baseline",
    choices=("joint_pd", "dcm"),
    default="joint_pd",
    help=(
        "Which model-based controller to measure. joint_pd is a zero action, "
        "which is the nominal-posture hold. dcm drives the same action space "
        "with capture-point feedback. Default joint_pd."
    ),
)
parser.add_argument(
    "--policy",
    type=Path,
    default=None,
    help=(
        "TorchScript policy exported by rsl-rl, usually "
        "logs/rsl_rl/nao_stand/<run>/exported/policy.pt. Scores a trained "
        "policy on the same protocol as the model-based baselines; overrides "
        "--baseline."
    ),
)
parser.add_argument(
    "--pushes",
    action="store_true",
    help=(
        "Enable perturbations at the final curriculum magnitude. This measures the "
        "fixed PD against the full task rather than against a quiet stance."
    ),
)
parser.add_argument(
    "--no-reset-randomization",
    action="store_true",
    help=(
        "Reset to the exact nominal state instead of the randomised distribution. "
        "Use this to test whether the posture itself is stable, separately from "
        "whether a fixed PD survives the reset distribution."
    ),
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

PASS = "  OK  "
FAIL = " FAIL "


class Checks:
    """Collects pass/fail lines so every check runs before the verdict."""

    def __init__(self) -> None:
        self.failures = 0

    def record(self, ok: bool, label: str, detail: str = "") -> None:
        marker = PASS if ok else FAIL
        self.failures += int(not ok)
        print(f"  [{marker}] {label}" + (f"   {detail}" if detail else ""))


def run_trial(env, actions_fn, steps: int) -> dict[str, float]:
    """Step the environment with a given action policy and summarise the outcome.

    Falling and running out of time are counted separately on purpose. The
    episode length buffer is randomised at the first reset so the batch does not
    terminate in lockstep, which means a fixed number of steps always contains
    some truncations. Charging those to the controller would understate it by
    roughly ``steps / max_episode_length`` -- about 15% here, enough to hide
    whether the posture is stable at all.
    """
    env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    ever_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
    falls = torch.zeros(num_envs, device=device)
    timeouts = torch.zeros(num_envs, device=device)
    reward_total = torch.zeros(num_envs, device=device)
    steps_since_reset = torch.zeros(num_envs, device=device)
    alive_steps: list[float] = []

    for _ in range(steps):
        actions = actions_fn(num_envs, device)
        _, reward, terminated, truncated, _ = env.step(actions)
        reward_total += reward
        steps_since_reset += 1.0

        if bool(terminated.any()):
            alive_steps.extend(steps_since_reset[terminated].tolist())
        ever_fell |= terminated
        falls += terminated.float()
        timeouts += truncated.float()
        steps_since_reset[terminated | truncated] = 0.0

    return {
        "fall_free_rate": float((~ever_fell).float().mean()),
        "falls": float(falls.sum()),
        "timeouts": float(timeouts.sum()),
        "mean_steps_to_fall": (
            sum(alive_steps) / len(alive_steps) if alive_steps else float("nan")
        ),
        "mean_return": float(reward_total.mean()),
    }


def make_baseline_policy(inner, cfg):
    """Return an action function implementing the requested model-based controller.

    The environment's action is a displacement on the nominal posture,
    ``q_des = q_nominal + action_scale * a``, so any controller that produces
    joint targets can be expressed as an action by inverting that:

        a = (q_des - q_nominal) / action_scale

    restricted to the joints the policy owns. That is what lets the joint PD,
    the capture-point controller and a learned policy all be scored on exactly
    the same protocol, which is the only way the comparison means anything.
    """
    if args_cli.policy is not None:
        # A TorchScript policy exported by rsl-rl. The export carries the
        # observation normaliser inside it, so it consumes the raw 65-value
        # actor observation and returns the 19 actions directly -- the same
        # tensor a zero-action or capture-point baseline would produce, which
        # is what makes the three comparable at all.
        module = torch.jit.load(str(args_cli.policy), map_location=inner.device)
        module.eval()

        def learned(num_envs: int, device: str) -> torch.Tensor:
            with torch.inference_mode():
                return module(inner._get_observations()["policy"]).clone()

        return learned

    if args_cli.baseline == "joint_pd":
        # A zero action already is the nominal-posture hold.
        return lambda n, d: torch.zeros(n, cfg.action_space, device=d)

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
    actuated = inner._actuated_ids
    nominal = inner._nominal_joint_pos

    def policy(num_envs: int, device: str) -> torch.Tensor:
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
        return (targets[:, actuated] - nominal) / cfg.action_scale

    return policy


def main() -> int:
    checks = Checks()
    cfg = NaoStandEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    # Pushes are disabled for the zero-action trial: the claim under test is
    # that the nominal posture is stable, not that it survives perturbation.
    if args_cli.pushes:
        # Skip the ramp: the question is what the controller can already do.
        cfg.push_velocity_initial = cfg.push_velocity_final
        cfg.push_curriculum_steps = 1
    else:
        cfg.push_velocity_initial = 0.0
        cfg.push_velocity_final = 0.0
    if args_cli.no_reset_randomization:
        cfg.events.reset_joints = None
        cfg.events.reset_base = None
        cfg.events.torso_mass = None

    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped

    print("\n" + "=" * 78)
    print("NAO STANDING ENVIRONMENT -- SMOKE TEST")
    print("=" * 78)
    print(f"  task                   : {TASK_ID}")
    print(f"  environments           : {inner.num_envs}")
    print(f"  control rate           : {1.0 / inner.step_dt:.0f} Hz")
    print(f"  episode length         : {inner.max_episode_length} steps")
    print(f"  device                 : {inner.device}")
    randomised = not args_cli.no_reset_randomization
    print(f"  reset randomisation    : {'on' if randomised else 'off'}")
    label = "policy" if args_cli.policy is not None else args_cli.baseline
    print(f"  controller             : {label}")
    if args_cli.policy is not None:
        print(f"  checkpoint             : {args_cli.policy}")
    print(
        f"  pushes                 : "
        f"{f'{cfg.push_velocity_final:.3f} m/s' if args_cli.pushes else 'off'}"
    )

    print("\n  STRUCTURE")
    obs, _ = env.reset()
    policy_dim = obs["policy"].shape[1]
    critic_dim = obs["critic"].shape[1]
    checks.record(
        policy_dim == cfg.observation_space,
        "actor observation width matches the config",
        f"{policy_dim} vs {cfg.observation_space}",
    )
    checks.record(
        critic_dim == cfg.state_space,
        "critic observation width matches the config",
        f"{critic_dim} vs {cfg.state_space}",
    )
    checks.record(
        len(inner._actuated_ids) == cfg.action_space,
        "actuated joint count matches the action space",
        f"{len(inner._actuated_ids)} joints",
    )
    commanded = [inner._robot.joint_names[i] for i in inner._actuated_ids]
    checks.record(
        commanded == list(nk.ACTUATED_JOINTS),
        "commanded joints resolved in the declared order",
    )
    checks.record(
        "RHipYawPitch" not in commanded,
        "the mimic-driven hip joint is never commanded",
    )
    checks.record(
        len(inner._foot_ids) == 2,
        "both feet resolved for contact and slip terms",
        f"{[inner._robot.body_names[i] for i in inner._foot_ids]}",
    )
    checks.record(
        len(inner._undesired_contact_ids) > 0,
        "undesired-contact bodies resolved",
        f"{len(inner._undesired_contact_ids)} bodies",
    )
    checks.record(
        bool(torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()),
        "observations are finite at reset",
    )

    print(f"\n  BASELINE ({args_cli.baseline}) -- model-based, no learning")
    zero = run_trial(env, make_baseline_policy(inner, cfg), args_cli.steps)
    print(f"      fall-free rate     : {zero['fall_free_rate']:.1%}")
    print(f"      falls / timeouts   : {zero['falls']:.0f} / {zero['timeouts']:.0f}")
    print(f"      mean steps to fall : {zero['mean_steps_to_fall']:.1f}")
    print(f"      mean return        : {zero['mean_return']:+.3f}")
    if randomised:
        # With randomisation on this is a measurement of the fixed PD, not a
        # correctness check: the reset distribution is deliberately harder than
        # anything a single linear feedback law is expected to cover, and the
        # gap is exactly what the policy has to close.
        checks.record(
            zero["fall_free_rate"] > 0.5,
            f"the {label} controller survives most perturbations",
            f"{zero['fall_free_rate']:.1%} never fell -- headroom for learning: "
            f"{1.0 - zero['fall_free_rate']:.1%}",
        )
    else:
        checks.record(
            zero["fall_free_rate"] > 0.98,
            f"the {label} controller holds a quiet stance",
            f"{zero['fall_free_rate']:.1%} never fell",
        )

    print("\n  RANDOM ACTION -- must be clearly worse, or there is nothing to learn")
    noisy = run_trial(
        env,
        lambda n, d: torch.randn(n, cfg.action_space, device=d).clamp(-1.0, 1.0),
        args_cli.steps,
    )
    print(f"      fall-free rate     : {noisy['fall_free_rate']:.1%}")
    print(f"      falls / timeouts   : {noisy['falls']:.0f} / {noisy['timeouts']:.0f}")
    print(f"      mean steps to fall : {noisy['mean_steps_to_fall']:.1f}")
    print(f"      mean return        : {noisy['mean_return']:+.3f}")
    checks.record(
        noisy["mean_return"] < zero["mean_return"],
        "the reward separates a good policy from a bad one",
        f"{noisy['mean_return']:+.3f} < {zero['mean_return']:+.3f}",
    )

    print("\n  REWARD TERMS (per-episode averages, from the last reset)")
    log = inner.extras.get("log", {})
    for key in sorted(k for k in log if k.startswith("Episode_Reward/")):
        value = log[key]
        value = float(value) if not isinstance(value, float) else value
        print(f"      {key.split('/', 1)[1]:20s} {value:+.5f}")
    finite = all(
        torch.isfinite(torch.as_tensor(float(v))).item()
        for k, v in log.items()
        if k.startswith("Episode_Reward/")
    )
    checks.record(finite and len(log) > 0, "every reward term logged a finite value")

    print("\n" + "=" * 78)
    if checks.failures == 0:
        print("ALL CHECKS PASSED -- the environment is ready to train.")
    else:
        print(f"{checks.failures} CHECK(S) FAILED -- do not start training yet.")
    print("=" * 78 + "\n")

    env.close()
    return 1 if checks.failures else 0


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
