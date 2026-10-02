"""Check the G1 tracking task against things that are already known to be right.

Three checks, with the pinned teacher driving the robot through the
``teacher`` observation group so the body walks the way the contract's
generator was trained on:

1. **The skill state.** The task computes ``z*`` on the GPU; the rollout
   capture computes it in NumPy from the same simulator state. They must
   agree, step by step.
2. **The plan's clock.** Plans are drawn at the first trustworthy step after a
   reset and then every ``commit`` steps; ``age`` counts 0 .. commit - 1.
3. **The scale of the reward.** A teacher-driven body realises one plausible
   future and the plan is another draw from the same distribution, so the
   tracking error, in units of the plan's tolerance, should be of order one
   -- not ten, which a wrong frame or index would give.

Usage::

    python g1/scripts/check_contract_env.py --headless
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "source" / "humanoid_transfer"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the G1 tracking task.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--encoder", action="store_true", help="Feed the contract from the encoder.")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--robot",
    choices=("g1", "nao"),
    default="g1",
    help="g1: the pinned teacher drives. nao: zero actions -- the default pose held "
    "by the PD drives, since the NAO has no teacher.",
)
parser.add_argument(
    "--start-fraction",
    type=float,
    default=0.0,
    help="Share of resets from demonstrated states (RSI).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_transfer.tasks  # noqa: F401
from humanoid_transfer.common.frames import quat_to_matrix
from humanoid_transfer.common.skill_state import MultibodySnapshot, encode
from humanoid_transfer.g1.tasks.g1_track import PLAY_TASK_ID as G1_PLAY
from humanoid_transfer.nao.tasks.nao_track import PLAY_TASK_ID as NAO_PLAY

TEACHER = _REPO_ROOT / "g1" / "policy" / "teacher_policy.pt"
BODY_CARD = _REPO_ROOT / "g1" / "body" / "body_card.json"


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    task = G1_PLAY if args_cli.robot == "g1" else NAO_PLAY
    cfg = parse_env_cfg(task, num_envs=args_cli.num_envs)
    cfg.commands.base_velocity.use_encoder = args_cli.encoder
    cfg.commands.base_velocity.start_fraction = args_cli.start_fraction
    env = gym.make(task, cfg=cfg)
    inner = env.unwrapped
    term = inner.command_manager.get_term("base_velocity")
    robot = inner.scene["robot"]
    sensor = inner.scene["contact_forces"]
    teacher = (
        torch.jit.load(str(TEACHER), map_location=inner.device).eval()
        if args_cli.robot == "g1"
        else None
    )

    # The NumPy encoder fed with the constants the task built: this checks the GPU
    # transcription in place; the G1's constants are checked against its body card
    # by the rollout capture, the NAO's sole points against its URDF.
    body = term.body
    sensor_ids = term.sensor_ids
    masses = body.masses.cpu().numpy().astype(float)
    inertia = body.inertia_local.cpu().numpy().astype(float)
    sole = body.sole_from_com.cpu().numpy().astype(float)
    base_ids, foot_ids = [body.base_index], list(body.foot_index)
    units = body.units
    n = inner.num_envs

    def numpy_skill() -> np.ndarray:
        data = robot.data
        origins = inner.scene.env_origins.cpu().numpy()[:, None, :]
        snapshot = MultibodySnapshot(
            masses=np.broadcast_to(masses, (n, len(masses))),
            inertia_local=np.broadcast_to(inertia, (n, len(masses), 3, 3)),
            link_rot=quat_to_matrix(data.body_link_quat_w.cpu().numpy().astype(float)),
            com_pos=data.body_com_pos_w.cpu().numpy().astype(float) - origins,
            com_lin_vel=data.body_com_lin_vel_w.cpu().numpy().astype(float),
            ang_vel=data.body_com_ang_vel_w.cpu().numpy().astype(float),
            foot_force=sensor.data.net_forces_w[:, sensor_ids, :].cpu().numpy(),
            base_index=base_ids[0],
            foot_index=(foot_ids[0], foot_ids[1]),
            sole_from_com=sole,
        )
        return encode(snapshot, units)

    obs, _ = env.reset()
    speeds, zs = [], []
    worst, ages, draws, errors, falls = 0.0, [], 0, [], 0
    previous_plan = term.plan.clone()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            if teacher is None:
                action = torch.zeros(
                    inner.num_envs, inner.action_manager.total_action_dim, device=inner.device
                )
            else:
                action = teacher(obs["teacher"])
        obs, _, terminated, truncated, _ = env.step(action)
        falls += int(terminated.sum())
        speeds.append(float(robot.data.root_lin_vel_b[:, 0].mean()))
        if step >= 2:
            gpu = term.generator.skill_norm.inverse(term.true_skill()).cpu().numpy()
            zs.append(gpu)
            worst = max(worst, float(np.abs(gpu - numpy_skill()).max()))
            errors.append(term.last_error.cpu().numpy().copy())
        ages.append(term.age.cpu().numpy().copy())
        changed = (term.plan - previous_plan).abs().amax(dim=(1, 2)) > 0
        draws += int(changed.sum())
        previous_plan = term.plan.clone()

    ages = np.stack(ages)
    errors = np.stack(errors)
    zs = np.stack(zs)
    settled = errors[50:]
    result = {
        "steps": args_cli.steps,
        "envs": n,
        "encoder": args_cli.encoder,
        "max_abs_difference_gpu_vs_numpy": worst,
        "age_values": sorted(int(a) for a in np.unique(ages)),
        "commit": term.commit,
        "view": term.view,
        "plans_drawn_per_env_per_commit": draws / n / (args_cli.steps / term.commit),
        "tracking_error_median": float(np.median(settled)),
        "tracking_error_mean": float(settled.mean()),
        "tracking_error_by_age": [
            # The error at step s was scored against the age held after step s - 1.
            float(np.median(settled[ages[51:][: len(settled)] == a]))
            for a in range(term.commit)
        ],
        "falls": falls,
        "robot": args_cli.robot,
        "mean_skill_state_last_100_steps": dict(
            zip(
                ("height", "load_left", "load_right"),
                [float(v) for v in zs[-100:].mean(axis=(0, 1))[[0, 22, 23]]],
                strict=True,
            )
        ),
        "start_fraction": args_cli.start_fraction,
        "forward_speed_first_25_steps": float(np.mean(speeds[:25])),
        "forward_speed_all": float(np.mean(speeds)),
    }
    print(json.dumps(result, indent=2))
    if args_cli.output is not None:
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close(skip_cleanup=True)
    raise SystemExit(code)
