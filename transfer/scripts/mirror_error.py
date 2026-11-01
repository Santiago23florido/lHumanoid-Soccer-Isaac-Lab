"""How far is a tracker from its mirror image, joint by joint, on the states it visits?

PPO's mirror loss (rsl-rl, ``use_mirror_loss``) is the mean of ``(π(M o) − P π(o))²``
over a minibatch: the action means at the mirrored observation against the mirrored
action means. The warm study's distilled physics trackers began PPO with it far above
the incumbent's (0.67 against 0.05 in round 6, §9.1), and collapsed. This measures it
where PPO does -- the tracker run with its exploration noise, the env's own
:class:`~humanoid_transfer.transfer.tasks.tracking.symmetry.TrackerMirror` -- and splits
it by joint, and by observation term: for each term, the error left when only that
term is mirrored and the rest left as it is says how much of the asymmetry reads it.

Usage::

    python transfer/scripts/mirror_error.py --headless --task G1TrackPhysics-v0 \
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
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument(
    "--mirror", type=Path, default=REPO_ROOT / "g1" / "body" / "mirror.json", help="The body's."
)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402, F401
import torch  # noqa: E402
from humanoid_transfer.transfer.tasks.tracking.symmetry import TrackerMirror  # noqa: E402
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
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(args_cli.checkpoint))
    actor = runner.alg.actor
    actor.eval()
    std = actor.distribution.std_param.detach()
    mirror = TrackerMirror(inner, args_cli.mirror)
    (term,) = inner.action_manager._terms.values()
    names = list(term._joint_names)
    groups = list(actor.obs_groups)
    manager = inner.observation_manager
    layout = {}
    for group in groups:
        start = 0
        for name, shape in zip(
            manager.active_terms[group], manager.group_obs_term_dim[group], strict=True
        ):
            width = int(torch.tensor(shape).prod())
            layout[(group, name)] = slice(start, start + width)
            start += width

    per_joint = torch.zeros(len(names), device=device)
    per_term = {f"{g}/{n}": 0.0 for g, n in layout}
    count = 0
    obs = env.get_observations()
    for _ in range(args_cli.steps):
        with torch.no_grad():
            mean = actor(obs)
            reflected = obs.clone()
            for group in groups:
                reflected[group] = mirror.observation(group, obs[group])
            target = mirror.action(mean)
            error = (actor(reflected) - target).pow(2)
            per_joint += error.mean(0)
            # Each term alone mirrored, the rest of the observation as it is.
            for (group, name), span in layout.items():
                partial = obs.clone()
                partial[group][:, span] = reflected[group][:, span]
                per_term[f"{group}/{name}"] += float((actor(partial) - mean).pow(2).mean())
            count += 1
            action = mean + std * torch.randn_like(mean)
        obs, _, _, _ = env.step(action)

    per_joint = (per_joint / count).cpu()
    joints = {n: round(float(v), 5) for n, v in zip(names, per_joint, strict=True)}
    terms = {k: round(v / count, 5) for k, v in per_term.items()}
    report = {
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint),
        "mirror_loss": float(per_joint.mean()),
        "per_joint": dict(sorted(joints.items(), key=lambda kv: -kv[1])),
        "change_when_only_this_term_is_mirrored": dict(
            sorted(terms.items(), key=lambda kv: -kv[1])
        ),
    }
    print(json.dumps({k: report[k] for k in ("task", "mirror_loss")}))
    print("worst joints:", list(report["per_joint"].items())[:6])
    print("terms:", list(report["change_when_only_this_term_is_mirrored"].items())[:6])
    if args_cli.output is not None:
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
