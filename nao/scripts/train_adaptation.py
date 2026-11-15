"""The student: the teacher's actor, frozen, told the robot's parameters by an adaptation module.

The teacher (``nao/scripts/train_teacher.sh``) reads each robot's five parameters
(``robot_params``: latency and the leg drives' stiffness, damping, torque ceiling and
rotor inertia, :mod:`humanoid_transfer.nao.sim2real.params`). A real robot does not
know them; the student estimates them from the last 0.5 s of what it sensed and did
(the ``*_adapt_history`` terms of ``NaoTrack-Sim2RealStudent-v0``) with a small
network ``φ``, and the teacher's actor acts on that estimate (Kumar et al., RMA,
2021). Only ``φ`` is trained, by regression on the true parameters:

* **On the student's own states.** Every rollout is driven by the deployed student
  -- the teacher's symmetrised actor fed ``φ``'s estimate, its mean action -- except
  a share of the robots, falling from all to none over the first rounds, driven with
  the true parameters (DAgger's mixture). The teacher is never asked about anything
  the student does not observe: its other inputs, the last action included, are the
  student's own.
* **Symmetric.** The parameters are the same robot in the mirror, so the estimate is
  ``½ (φ(h) + φ(M h))``, trained as such.
* **Measured as it goes.** Each round reports, on its own fresh rollouts before they are
  trained on, the error and ``R²`` of each parameter, the error by time since the
  episode began, and the falls per 1000 steps of the robots the student drove.

Writes ``adaptation.pt`` (``φ``, its input normaliser, the layout) and ``rounds.json``
in ``logs/rsl_rl/nao_track/<date>_<name>/``; ``nao/scripts/student_bundle.py`` makes a
deployment bundle of it.

Usage::

    python nao/scripts/train_adaptation.py --headless \
        --teacher logs/rsl_rl/nao_track/<teacher run>/model_<n>.pt --run_name s2r_student
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="NaoTrack-Sim2RealStudent-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--teacher", type=Path, required=True, help="The teacher's rsl-rl checkpoint.")
parser.add_argument("--rounds", type=int, default=300)
parser.add_argument("--steps", type=int, default=24, help="Control steps per round.")
parser.add_argument("--mixture", type=float, default=0.3, help="Share of rounds the mixture lasts.")
parser.add_argument("--keep", type=float, default=0.125, help="Share of each round's samples kept.")
parser.add_argument("--replay", type=int, default=8, help="Rounds of data trained on together.")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch", type=int, default=8192)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--run_name", default="s2r_student")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402, F401
import torch  # noqa: E402
from humanoid_transfer.nao.sim2real.params import PARAMS  # noqa: E402
from humanoid_transfer.nao.tasks.nao_track.nao_track_sim2real_env_cfg import (  # noqa: E402
    ADAPT_HISTORY,
    PARAM_RANGES,
)
from humanoid_transfer.transfer.tasks.tracking.symmetry import TrackerMirror  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from torch import nn  # noqa: E402

EPSILON = 1e-2
"""rsl-rl's empirical normaliser divides by its standard deviation plus this."""
AGE_BINS = (0, 25, 50, 100, 200, 10**9)
"""Control steps since the episode began, for the estimate's error over time."""


class Teacher(nn.Module):
    """The teacher's actor from its checkpoint: normaliser, MLP, ELU; the mean action."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        self.register_buffer("mean", state["obs_normalizer._mean"][0].float())
        self.register_buffer("scale", state["obs_normalizer._std"][0].float() + EPSILON)
        indices = sorted(
            int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")
        )
        layers: list[nn.Module] = []
        for i in indices:
            weight = state[f"mlp.{i}.weight"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(state[f"mlp.{i}.bias"])
            layers += [linear, nn.ELU()]
        self.mlp = nn.Sequential(*layers[:-1])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp((observation - self.mean) / self.scale)


class Adaptation(nn.Module):
    """``φ``: the history, normalised, through an ELU MLP to the five parameters."""

    def __init__(self, inputs: int, hidden: list[int], outputs: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(inputs))
        self.register_buffer("scale", torch.ones(inputs))
        layers: list[nn.Module] = []
        width = inputs
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ELU()]
            width = size
        layers.append(nn.Linear(width, outputs))
        self.mlp = nn.Sequential(*layers)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.mlp((history - self.mean) / self.scale)


def main() -> int:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    inner = env.unwrapped
    device = inner.device
    log_dir = (
        ROOT
        / "logs"
        / "rsl_rl"
        / "nao_track"
        / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.run_name}")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    manager = inner.observation_manager
    names = list(manager.active_terms["policy"])
    widths = [int(torch.tensor(d).prod()) for d in manager.group_obs_term_dim["policy"]]
    start = sum(widths[: names.index("robot_params")])
    width = start + len(PARAMS)
    terms = list(zip(names, widths, strict=True))
    if any(not n.endswith("_adapt_history") for n, _ in terms[names.index("robot_params") + 1 :]):
        raise ValueError(f"{args_cli.task}: the policy group does not end in the adapt history")

    saved = torch.load(args_cli.teacher, map_location=device, weights_only=False)
    teacher = Teacher(saved["actor_state_dict"]).to(device).eval()
    if teacher.mean.shape[0] != width:
        raise ValueError(f"the teacher reads {teacher.mean.shape[0]} inputs, the task {width}")
    for p in teacher.parameters():
        p.requires_grad_(False)
    mirror = TrackerMirror(env, ROOT / "nao" / "body" / "mirror.json")
    linear, offset = mirror.groups["policy"]
    base_linear, base_offset = linear[:width, :width], offset[:width]
    history_linear, history_offset = linear[width:, width:], offset[width:]
    if float(linear[:width, width:].abs().max()) or float(linear[width:, :width].abs().max()):
        raise ValueError("the mirror mixes the teacher's inputs with the history")

    def act(base: torch.Tensor) -> torch.Tensor:
        reflected = base @ base_linear.T + base_offset
        return 0.5 * (teacher(base) + mirror.action(teacher(reflected)))

    history_width = linear.shape[0] - width
    phi = Adaptation(history_width, args_cli.hidden, len(PARAMS)).to(device)

    def estimate(history: torch.Tensor) -> torch.Tensor:
        reflected = history @ history_linear.T + history_offset
        return 0.5 * (phi(history) + phi(reflected))

    optimiser = torch.optim.Adam(phi.parameters(), lr=args_cli.learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args_cli.rounds, 1e-5)
    replay: list[tuple[torch.Tensor, torch.Tensor]] = []
    rounds = []
    obs = env.get_observations()
    n = env.num_envs
    mixture_rounds = max(1, int(args_cli.mixture * args_cli.rounds))
    for round_index in range(args_cli.rounds):
        share = max(0.0, 1.0 - round_index / mixture_rounds)
        told = torch.rand(n, device=device) < share
        histories, targets, ages = [], [], []
        falls = torch.zeros((), device=device)
        student_steps = torch.zeros((), device=device)
        with torch.no_grad():
            for _ in range(args_cli.steps):
                policy = obs["policy"]
                base = policy[:, :width].clone()
                history = policy[:, width:]
                truth = base[:, start:width].clone()
                guess = estimate(history).clamp(-1.0, 1.0)
                base[:, start:width] = torch.where(told[:, None], truth, guess)
                obs, _, dones, extras = env.step(act(base))
                keep = torch.rand(n, device=device) < args_cli.keep
                histories.append(history[keep].half())
                targets.append(truth[keep])
                ages.append(inner.episode_length_buf[keep].clone())
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
                falls += (dones.bool() & ~timeouts & ~told).sum()
                student_steps += (~told).sum()
        history = torch.cat(histories)
        target = torch.cat(targets)
        age = torch.cat(ages)
        # The round's rollouts, before training on them: the estimate's own error.
        with torch.no_grad():
            if round_index == 0:
                phi.mean.copy_(history.float().mean(0))
                phi.scale.copy_(history.float().std(0) + EPSILON)
            error = (estimate(history.float()).clamp(-1.0, 1.0) - target) ** 2
        variance = target.var(0).clamp_min(1e-9)
        by_age = {}
        for low, high in zip(AGE_BINS[:-1], AGE_BINS[1:], strict=True):
            mask = (age >= low) & (age < high)
            if mask.any():
                by_age[f"{low}-{high if high < 10**9 else 'end'}"] = float(error[mask].mean())
        record = {
            "round": round_index,
            "told_share": share,
            "mse": error.mean(0).tolist(),
            "r2": (1.0 - error.mean(0) / variance).tolist(),
            "mse_by_age": by_age,
            "student_falls_per_1000": float(1000.0 * falls / student_steps.clamp_min(1)),
            "samples": int(history.shape[0]),
        }
        replay.append((history, target))
        replay = replay[-args_cli.replay :]
        inputs = torch.cat([h for h, _ in replay])
        labels = torch.cat([t for _, t in replay])
        losses = []
        for _ in range(args_cli.epochs):
            order = torch.randperm(inputs.shape[0], device=device)
            for first in range(0, inputs.shape[0], args_cli.batch):
                batch = order[first : first + args_cli.batch]
                loss = ((estimate(inputs[batch].float()) - labels[batch]) ** 2).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                losses.append(float(loss))
        schedule.step()
        record["train_loss"] = sum(losses) / len(losses)
        rounds.append(record)
        pairs = zip(PARAMS, record["r2"], strict=True)
        r2 = " ".join(f"{k.split('_')[0]} {v:+.2f}" for k, v in pairs)
        print(
            f"round {round_index:3d} told {share:.2f} loss {record['train_loss']:.4f} R2 {r2} "
            f"falls {record['student_falls_per_1000']:.2f}/1000",
            flush=True,
        )
        if (round_index + 1) % 50 == 0 or round_index + 1 == args_cli.rounds:
            torch.save(
                {
                    "state_dict": phi.state_dict(),
                    "hidden": list(args_cli.hidden),
                    "params": list(PARAMS),
                    "param_ranges": [list(r) for r in PARAM_RANGES],
                    "adapt_history": ADAPT_HISTORY,
                    "policy_terms": [list(t) for t in terms],
                    "teacher": str(args_cli.teacher),
                    "round": round_index,
                },
                log_dir / "adaptation.pt",
            )
            (log_dir / "rounds.json").write_text(json.dumps(rounds, indent=1), encoding="utf-8")
    print(f"wrote {log_dir / 'adaptation.pt'}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    os._exit(code)
