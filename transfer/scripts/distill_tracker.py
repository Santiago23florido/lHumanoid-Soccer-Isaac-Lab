"""Start a physics tracker from a competent one: distil the incumbent into decoder v2 (DAgger).

Decoder v2's from-scratch rounds left open whether its physics prior helps a tracker
that already walks (``transfer/docs/skill_space.md`` §9.1): both from-scratch trackers
stayed below the incumbent, which had three times the training. Here the question is
asked at equal competence: the physics task's actor (a residual on the prior) is fitted
to the incumbent's joint targets on the states it visits itself (DAgger, Ross et al.
2011), and its critic to the incumbent's values, before PPO goes on from there.

* **Teacher.** The incumbent's deployed (symmetrized) actor, which reads the plain
  tracker's observations: the physics task's policy group extends them, so they are
  its first columns. Its joint target ``q = q0 + 0.5 a`` becomes the student's residual
  target ``(q - offset - prior) / scale``.
* **Rollouts.** Each round runs the student, and a share of the envs, falling from all
  to none over the rounds, runs the teacher's target instead (DAgger's mixture).
* **Exploration.** The student's action noise is set to the teacher's in joint space,
  so PPO afterwards explores as the incumbent did.

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
from datetime import datetime
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
parser.add_argument("--rounds", type=int, default=60)
parser.add_argument("--steps", type=int, default=24, help="Control steps per round.")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatches", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=5e-4)
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
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
    optimiser = torch.optim.Adam([*parameters, *critic.parameters()], lr=args_cli.learning_rate)
    history = []
    for round_index in range(args_cli.rounds):
        share = max(0.0, 1.0 - round_index / (0.6 * args_cli.rounds))
        batch = {"obs": [], "target": [], "value": []}
        falls = 0
        for _ in range(args_cli.steps):
            with torch.no_grad():
                q_teacher = default + args_cli.teacher_scale * teacher(
                    obs["policy"][:, :policy_width]
                )
                on_prior = torch.where(term._mask, term._targets, 0.0)
                target = (q_teacher - term._offset - on_prior) / term._scale
                value = teacher_critic(
                    TensorDict(
                        {"critic": obs["critic"][:, :critic_width]}, batch_size=obs.batch_size
                    )
                )
                actor.update_normalization(obs)
                critic.update_normalization(obs)
                student = actor(obs)
                follow = torch.rand(len(target), device=device) < share
                action = torch.where(follow[:, None], target, student)
            batch["obs"].append(obs.clone())
            batch["target"].append(target)
            batch["value"].append(value)
            obs, _, dones, extras = env.step(action)
            falls += int(
                extras.get("time_outs", torch.zeros_like(dones))
                .logical_not()
                .logical_and(dones)
                .sum()
            )
        observations = torch.cat(batch["obs"])
        targets, values = torch.cat(batch["target"]), torch.cat(batch["value"])
        count = len(targets)
        losses = []
        for _ in range(args_cli.epochs):
            order = torch.randperm(count, device=device)
            for chunk in order.chunk(args_cli.minibatches):
                part = observations[chunk]
                action_loss = F.mse_loss(actor(part), targets[chunk])
                value_loss = F.mse_loss(critic(part), values[chunk])
                optimiser.zero_grad(set_to_none=True)
                (action_loss + value_loss).backward()
                torch.nn.utils.clip_grad_norm_([*parameters, *critic.parameters()], 1.0)
                optimiser.step()
                losses.append((float(action_loss), float(value_loss)))
        action_loss = sum(a for a, _ in losses) / len(losses)
        value_loss = sum(v for _, v in losses) / len(losses)
        row = {"round": round_index, "teacher_share": share, "action_mse": action_loss,
               "value_mse": value_loss, "falls": falls}  # fmt: skip
        history.append(row)
        cells = (f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())
        print(" ".join(cells), flush=True)

    runner.current_learning_iteration = 0
    path = os.path.join(log_dir, "model_0.pt")
    runner.save(path)
    Path(log_dir, "distill.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"distilled into {path}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
