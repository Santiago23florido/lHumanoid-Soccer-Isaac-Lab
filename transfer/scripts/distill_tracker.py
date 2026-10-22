"""Start a physics tracker from a competent one: distil the incumbent into decoder v2 (DAgger).

Decoder v2's from-scratch rounds left open whether its physics prior helps a tracker
that already walks (``transfer/docs/skill_space.md`` §9.1): both from-scratch trackers
stayed below the incumbent, which had three times the training. Here the question is
asked at equal competence: the physics task's actor (a residual on the prior) is fitted
to the incumbent's joint targets on the states it visits itself (DAgger, Ross et al.
2011), and its critic to the incumbent's values, before PPO goes on from there.

* **Teacher.** The incumbent's deployed (symmetrized) actor, which reads the plain
  tracker's observations: the physics task's policy group extends them, so they are
  its first columns -- all but its last action, which there is the physics task's
  residual; the teacher is shown instead the action the step executed, noise and all,
  in its own units (:class:`TeacherView`) -- as in its own PPO training, whose last
  action was its sampled one. The first two distillations showed it the residual, and
  their labels were the teacher's answers to an input it never saw; the fourth showed
  it its own noiseless output, which the student cannot see, and the labels kept an
  error the student could not fit (0.11 from the 19th round on). Its joint target
  ``q = q0 + 0.5 a``, limited by ``--label-limits``, less the prior's feed-forward over
  the joint's stiffness (the
  target whose drive torque plus that feed-forward is the teacher's own), becomes the
  student's residual target ``(q - offset - prior) / scale``. Run so through the physics
  task the teacher ends 0.016 episodes per 1000 steps, as in its own (0.014); with its
  targets as they are, 0.12; shown the residual, 16
  (``transfer/scripts/teacher_view_study.sh``, ``teacher_in_physics.py``).
* **Rollouts.** Each round runs the student, and a share of the envs, falling from all
  to none over the rounds, runs the teacher's target instead (DAgger's mixture).
* **Exploration.** The student's action noise is set to the teacher's in joint space,
  so PPO afterwards explores as the incumbent did, and every rollout is drawn with that
  noise (``--explore``): the states PPO visits, and the noisy last actions it observes,
  are then the ones the student was fitted on. The incumbent's noise is large (median
  1.42 action units, 0.7 rad), and a student fitted on noiseless rollouts left its first
  PPO episodes 27 times as often as the incumbent did (§9.1). The labels stay the
  teacher's noiseless targets.

* **Mirror.** PPO goes on with the mirror loss ``(π(M o) − P π(o))²``, which the
  incumbent was trained under and so meets (0.05), but a student fitted only on the states
  it visits does not: at their first PPO update the distilled core and physics trackers
  stood at 0.21 and 0.67, and both collapsed within two iterations. Each minibatch is
  therefore also fitted mirrored (``--mirror-labels``): the mirrored observation, with
  the mirrored label -- what that loss asks for -- and the same value; each round reports
  the student's own mirror loss.

The result is a checkpoint the physics task's PPO resumes from (``--resume``).

Usage::

    python transfer/scripts/distill_tracker.py --headless --num_envs 4096 \
        --teacher logs/rsl_rl/g1_track/<run>/exported/model_6194_sym.pt \
        --teacher-checkpoint logs/rsl_rl/g1_track/<run>/model_6194.pt --run_name distill_v2t
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime
from importlib import metadata
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="G1TrackPhysics-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--teacher", type=Path, required=True, help="The incumbent's exported actor.")
parser.add_argument("--teacher-checkpoint", type=Path, required=True, help="Its rsl-rl checkpoint.")
parser.add_argument(
    "--teacher-scale", type=float, default=0.5, help="The plain tracker's action scale."
)
parser.add_argument("--rounds", type=int, default=100)
parser.add_argument("--steps", type=int, default=24, help="Control steps per round.")
parser.add_argument("--keep", type=float, default=0.25, help="Share of each round's samples kept.")
parser.add_argument("--replay", type=int, default=12, help="Rounds of data trained on together.")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--batch", type=int, default=8192)
parser.add_argument("--learning-rate", type=float, default=5e-4)
parser.add_argument(
    "--explore",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Roll out with the student's exploration noise, as PPO will.",
)
parser.add_argument(
    "--compensate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Take the prior's feed-forward out of the teacher's targets (see above).",
)
parser.add_argument(
    "--label-limits",
    choices=("saturation", "soft", "none"),
    default="none",
    help="Where the teacher's targets are cut: where its drives saturate (a limit plus effort "
    "over stiffness, past which a target changes no torque), at the soft limits, or nowhere. "
    "Cut where they saturate the teacher acts the same (0.120 terminations per 1000 steps "
    "either way), but the last action it is shown in the envs it drives is no longer its own.",
)
parser.add_argument(
    "--mirror-labels",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fit each minibatch mirrored too, as PPO's mirror loss will ask (see above).",
)
parser.add_argument("--mirror", type=Path, default=None, help="The body's mirror.json.")
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--run_name", default="distill_v2t")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402, F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from humanoid_transfer.transfer.tasks.tracking.symmetry import TrackerMirror  # noqa: E402
from humanoid_transfer.transfer.tasks.tracking.teacher_view import TeacherView  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402


def main() -> int:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    # As Isaac Lab's train.py does: drop the keys this rsl-rl no longer takes.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(
        root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.run_name}"
    )
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    device = agent_cfg.device
    actor, critic = runner.alg.actor, runner.alg.critic
    inner = env.unwrapped
    term = inner.action_manager.get_term("joint_pos")
    robot = inner.scene["robot"]
    default = robot.data.default_joint_pos[:, term._action_joints]

    teacher = torch.jit.load(str(args_cli.teacher), map_location=device).eval()
    saved = torch.load(args_cli.teacher_checkpoint, map_location=device, weights_only=False)
    obs = env.get_observations()
    policy_width = saved["actor_state_dict"]["obs_normalizer._mean"].shape[-1]
    critic_width = saved["critic_state_dict"]["obs_normalizer._mean"].shape[-1]
    plain_obs = TensorDict({"critic": obs["critic"][:, :critic_width]}, batch_size=obs.batch_size)
    teacher_critic = MLPModel(
        plain_obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=[512, 256, 128],
        activation="elu", obs_normalization=True,
    ).to(device)  # fmt: skip
    teacher_critic.load_state_dict(saved["critic_state_dict"])
    teacher_critic.eval()
    # The teacher's exploration noise, in the student's action units.
    teacher_std = saved["actor_state_dict"]["distribution.std_param"]
    with torch.no_grad():
        actor.distribution.std_param.copy_(
            teacher_std * args_cli.teacher_scale / float(term._scale)
        )

    parameters = [p for p in actor.parameters() if p is not actor.distribution.std_param]
    # PPO's own optimizer, so its moments are not empty when PPO takes over from here (a fresh
    # Adam's first steps are as large as its rate, and broke the first distilled tracker).
    optimiser = runner.alg.optimizer
    ppo_rate = [group["lr"] for group in optimiser.param_groups]
    for group in optimiser.param_groups:
        group["lr"] = args_cli.learning_rate
    heads = getattr(actor, "physics_group", None)
    limits = robot.data.soft_joint_pos_limits[:, term._action_joints]
    if args_cli.label_limits == "saturation":
        joints = term._action_joints
        reach = robot.data.joint_effort_limits[:, joints] / robot.data.joint_stiffness[:, joints]
        limits = torch.stack([limits[..., 0] - reach, limits[..., 1] + reach], dim=-1)
    stiffness = robot.data.joint_stiffness[:, term._action_joints]
    view = TeacherView(
        inner.observation_manager,
        {"policy": policy_width, "critic": critic_width},
        env.num_envs,
        term.action_dim,
        device,
    )
    groups = list(actor.obs_groups) + [g for g in critic.obs_groups if g not in actor.obs_groups]
    if heads is not None:
        groups.append(heads)
    body = "nao" if args_cli.task.startswith("Nao") else "g1"
    mirror = TrackerMirror(inner, args_cli.mirror or REPO_ROOT / body / "body" / "mirror.json")

    def reflect(part: TensorDict) -> TensorDict:
        mirrored = part.clone()
        for group in part.keys():
            mirrored[group] = mirror.observation(group, part[group])
        return mirrored

    replay: deque = deque(maxlen=args_cli.replay)
    history = []
    for round_index in range(args_cli.rounds):
        share = max(0.0, 1.0 - round_index / (0.6 * args_cli.rounds))
        batch = {"obs": [], "target": [], "value": []}
        falls = 0
        for _ in range(args_cli.steps):
            with torch.no_grad():
                taught = teacher(view.observation(obs, "policy"))
                q_teacher = default + args_cli.teacher_scale * taught
                if args_cli.label_limits != "none":
                    q_teacher = torch.maximum(
                        torch.minimum(q_teacher, limits[..., 1]), limits[..., 0]
                    )
                shift = torch.zeros_like(q_teacher)
                if args_cli.compensate:
                    shift = term._feedforward / stiffness.clamp(min=1e-6)
                    shift = torch.where(stiffness > 0, shift, 0.0)
                    q_teacher = q_teacher - shift
                on_prior = torch.where(term._mask, term._targets, 0.0)
                target = (q_teacher - term._offset - on_prior) / term._scale
                value = teacher_critic(
                    TensorDict(
                        {"critic": view.observation(obs, "critic")}, batch_size=obs.batch_size
                    )
                )
                actor.update_normalization(obs)
                critic.update_normalization(obs)
                student = actor(obs)
                follow = torch.rand(len(target), device=device) < share
                action = torch.where(follow[:, None], target, student)
                if args_cli.explore:
                    action = action + actor.distribution.std_param * torch.randn_like(action)
                kept = torch.rand(len(target), device=device) < args_cli.keep
                # What the teacher would call this step's action: the joint targets it sends,
                # the compensation given back, in the teacher's units.
                sent = term._offset + on_prior + term._scale * action
                executed = (sent + shift - default) / args_cli.teacher_scale
            batch["obs"].append(obs.select(*groups)[kept].to("cpu"))
            batch["target"].append(target[kept].cpu())
            batch["value"].append(value[kept].cpu())
            obs, _, dones, extras = env.step(action)
            # The teacher's last action is the one executed, whoever acted, as the student's
            # is: the label is then a function of what the student observes.
            view.record(executed, dones)
            timed_out = extras.get("time_outs", torch.zeros_like(dones))
            falls += int(timed_out.logical_not().logical_and(dones).sum())
        replay.append({k: torch.cat(v) for k, v in batch.items()})
        # DAgger's aggregate: the recent rounds' states, each with the teacher's label.
        observations = torch.cat([r["obs"] for r in replay])
        targets = torch.cat([r["target"] for r in replay])
        values = torch.cat([r["value"] for r in replay])
        count = len(targets)
        losses = []
        for _ in range(args_cli.epochs):
            order = torch.randperm(count)
            for chunk in order.split(args_cli.batch):
                part = observations[chunk].to(device)
                label, value = targets[chunk].to(device), values[chunk].to(device)
                if args_cli.mirror_labels:
                    # The mirrored sample: the label PPO's mirror loss asks for, the same value.
                    part = torch.cat([part, reflect(part)])
                    label = torch.cat([label, mirror.action(label)])
                    value = torch.cat([value, value])
                action_loss = F.mse_loss(actor(part), label)
                value_loss = F.mse_loss(critic(part), value)
                total = action_loss + value_loss
                if heads is not None:
                    # The physics heads fitted on the trunk's features as they are, so PPO does
                    # not start from heads whose loss pulls the trunk.
                    total = total + actor.physics_loss(actor.trunk(part).detach(), part[heads])
                optimiser.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_([*parameters, *critic.parameters()], 1.0)
                optimiser.step()
                losses.append((float(action_loss), float(value_loss)))
        action_loss = sum(a for a, _ in losses) / len(losses)
        value_loss = sum(v for _, v in losses) / len(losses)
        with torch.no_grad():
            # The student's mirror loss as PPO computes it, on a slice of this round's states.
            probe = observations[: args_cli.batch].to(device)
            mirror_loss = float(F.mse_loss(actor(reflect(probe)), mirror.action(actor(probe))))
        row = {"round": round_index, "teacher_share": share, "action_mse": action_loss,
               "value_mse": value_loss, "mirror_loss": mirror_loss, "falls": falls,
               "samples": count}  # fmt: skip
        history.append(row)
        cells = (f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())
        print(" ".join(cells), flush=True)

    for group, rate in zip(optimiser.param_groups, ppo_rate, strict=True):
        group["lr"] = rate
    # The runner's own save, without its logger (set up only by learn()).
    path = os.path.join(log_dir, "model_0.pt")
    saved_dict = runner.alg.save()
    saved_dict["iter"] = 0
    saved_dict["infos"] = {
        "distilled_from": str(args_cli.teacher_checkpoint),
        "label_limits": args_cli.label_limits,
        "compensate": args_cli.compensate,
        "explore": args_cli.explore,
        "mirror_labels": args_cli.mirror_labels,
        "teacher_view": "own last action",
    }
    torch.save(saved_dict, path)
    Path(log_dir, "distill.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"distilled into {path}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
