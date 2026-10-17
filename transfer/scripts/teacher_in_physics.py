"""How well does the incumbent walk through the physics task's action, and its distilled student?

The distilled physics tracker exits the task several times as often as the incumbent it
was fitted to, though its joint targets are within 0.01 rad of the labels
(``transfer/docs/skill_space.md`` §9.1). This measures, in the physics task and on the
same states, each controller's terminations per 1000 env steps:

* ``teacher``: the incumbent's joint targets, sent through the residual as they are;
* ``teacher_clamped``: the same, clamped to the soft joint limits (the distillation's
  labels): a PD target beyond a limit still pushes, so the clamp is not a no-op;
* ``teacher_saturated``: clamped where the drive saturates instead, a limit plus the
  joint's effort over its stiffness: beyond that a target changes nothing;
* ``teacher_compensated``: saturated, less the prior's feed-forward over the joint's
  stiffness: the target whose drive torque plus that feed-forward is the teacher's;
* ``student``: a physics-task checkpoint (``--checkpoint``).

``--stochastic`` adds the incumbent's exploration noise, as PPO's rollouts do. For the
teacher it also reports, per joint, how often its target lies beyond the soft limit.

The teacher is shown its own last action (:class:`TeacherView`); ``--residual-view``
shows it the physics task's residual instead, as the first two distillations did. With
``--task G1Track-v0`` the same teacher modes run in its own task, for reference.

Usage::

    python transfer/scripts/teacher_in_physics.py --headless --num_envs 4096 \
        --teacher logs/rsl_rl/g1_track/<run>/exported/model_6194_sym.pt \
        --teacher-checkpoint logs/rsl_rl/g1_track/<run>/model_6194.pt \
        --checkpoint logs/rsl_rl/g1_track_physics/<run>/model_0.pt --output <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="G1TrackPhysics-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--teacher-checkpoint", type=Path, required=True)
parser.add_argument("--teacher-scale", type=float, default=0.5)
parser.add_argument("--checkpoint", type=Path, default=None, help="A physics-task checkpoint.")
parser.add_argument(
    "--modes", nargs="+", default=["teacher", "teacher_clamped", "teacher_saturated", "student"]
)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--residual-view", action="store_true", help="The distillations' old view.")
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402, F401
import torch  # noqa: E402
from humanoid_transfer.transfer.tasks.tracking.teacher_view import TeacherView  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def main() -> int:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = agent_cfg.device
    inner = env.unwrapped
    term = inner.action_manager.get_term("joint_pos")
    robot = inner.scene["robot"]
    joints = getattr(term, "_action_joints", None)  # the physics term's; the plain one's ids
    if joints is None:
        ids = term._joint_ids
        joints = torch.arange(robot.num_joints, device=device) if isinstance(ids, slice) else ids
    names = [robot.joint_names[int(j)] for j in joints]
    default = robot.data.default_joint_pos[:, joints]
    limits = robot.data.soft_joint_pos_limits[:, joints]
    # Where a drive saturates: past its limit by its effort over its stiffness.
    reach = robot.data.joint_effort_limits[:, joints] / robot.data.joint_stiffness[:, joints]
    saturated = torch.stack([limits[..., 0] - reach, limits[..., 1] + reach], dim=-1)
    stiffness = robot.data.joint_stiffness[:, joints]

    teacher = torch.jit.load(str(args_cli.teacher), map_location=device).eval()
    saved = torch.load(args_cli.teacher_checkpoint, map_location=device, weights_only=False)
    policy_width = saved["actor_state_dict"]["obs_normalizer._mean"].shape[-1]
    teacher_std = saved["actor_state_dict"]["distribution.std_param"]
    view = TeacherView(
        inner.observation_manager, {"policy": policy_width}, env.num_envs, len(names), device
    )
    student = None
    if "student" in args_cli.modes:
        if args_cli.checkpoint is None:
            raise SystemExit("--checkpoint is needed for the student")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(str(args_cli.checkpoint))
        student = runner.get_inference_policy(device=device)
        student_std = runner.alg.actor.distribution.std_param.detach()

    report = {"task": args_cli.task, "steps": args_cli.steps, "envs": args_cli.num_envs,
              "stochastic": args_cli.stochastic, "residual_view": args_cli.residual_view,
              "modes": {}}  # fmt: skip
    for mode in args_cli.modes:
        torch.manual_seed(args_cli.seed)
        env.reset()
        view.reset()
        obs = env.get_observations()
        ended = {"terminated": 0, "timed_out": 0}
        beyond = torch.zeros(len(names), device=device)
        excess = torch.zeros(len(names), device=device)
        for _ in range(args_cli.steps):
            with torch.no_grad():
                if mode == "student":
                    action = student(obs)
                    if args_cli.stochastic:
                        action = action + student_std * torch.randn_like(action)
                else:
                    if args_cli.residual_view:
                        a = teacher(obs["policy"][:, :policy_width])
                    else:
                        a = teacher(view.observation(obs, "policy"))
                    if args_cli.stochastic:
                        a = a + teacher_std * torch.randn_like(a)
                    q = default + args_cli.teacher_scale * a
                    low, high = limits[..., 0], limits[..., 1]
                    out = (q < low) | (q > high)
                    beyond += out.float().mean(0)
                    excess += torch.where(out, torch.maximum(low - q, q - high), 0.0).mean(0)
                    if mode == "teacher_clamped":
                        q = torch.maximum(torch.minimum(q, high), low)
                    elif mode in ("teacher_saturated", "teacher_compensated"):
                        q = torch.maximum(torch.minimum(q, saturated[..., 1]), saturated[..., 0])
                    if mode == "teacher_compensated" and hasattr(term, "_feedforward"):
                        shift = term._feedforward / stiffness.clamp(min=1e-6)
                        q = q - torch.where(stiffness > 0, shift, 0.0)
                    if hasattr(term, "_mask"):
                        on_prior = torch.where(term._mask, term._targets, 0.0)
                        action = (q - term._offset - on_prior) / term._scale
                    else:
                        action = (q - term._offset) / term._scale
            obs, _, dones, extras = env.step(action)
            if mode != "student":
                view.record(a, dones)
            timed_out = extras.get("time_outs", torch.zeros_like(dones)).bool()
            ended["terminated"] += int((dones.bool() & ~timed_out).sum())
            ended["timed_out"] += int((dones.bool() & timed_out).sum())
        row = {
            "terminations_per_1000_steps": 1000.0
            * ended["terminated"]
            / (args_cli.steps * args_cli.num_envs),
            **ended,
        }
        if mode != "student":
            share = (beyond / args_cli.steps).cpu().tolist()
            depth = (excess / args_cli.steps).cpu().tolist()
            row["beyond_soft_limit"] = {
                n: {"share": round(s, 4), "mean_excess_rad": round(d, 4)}
                for n, s, d in zip(names, share, depth, strict=True)
                if s > 1e-3
            }
        report["modes"][mode] = row
        print(mode, json.dumps({k: v for k, v in row.items() if k != "beyond_soft_limit"}),
              flush=True)  # fmt: skip
    if args_cli.output is not None:
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
