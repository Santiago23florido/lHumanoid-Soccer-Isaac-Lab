"""What the teacher does from the states a closed loop actually visits (DAgger on the task side).

A contract generator learns from the teacher's own trajectories, and a closed
loop drifts away from them: the G1 tracker, following its plans precisely, slowed
down to standing even when started mid-walk, and the generator had never seen
the teacher in a slow or standing state under a walking command -- so it never
learned the teacher's way back. The remedy for compounding error in imitation is
to ask the expert in the states the learner reaches (Ross, Gordon and Bagnell,
2011). Here the learner is the closed loop of generator and tracker, and the
expert is still the teacher:

1. take states from a closed-loop capture (``capture_rollouts.py`` records the base
   pose, joints and velocities);
2. write each into the G1 with the command it was under, and let the pinned teacher
   act from there for ``--horizon`` steps;
3. save, per start, the closed loop's own ``z*`` history up to that state followed by
   the teacher's response, with ``target_mask`` marking the steps whose future is the
   teacher's -- the only ones the generator may learn from.

Another task's teacher answers the same way: ``--checkpoint`` and ``--task-id``
name it and the task it was trained in (Isaac Lab's published G1 policies, for
the zero-shot study's task-side DAgger), and ``--fixed-command`` keeps each
start's command as captured instead of letting the task's heading control
recompute the turn.

Usage::

    python g1/scripts/teacher_from_states.py --headless \
        --states g1/rollouts/diag_r5_500_rsi.npz --states g1/rollouts/diag_r5_500_stand.npz \
        --output g1/rollouts/dagger_contract_1.npz
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "source" / "humanoid_transfer"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="The teacher's response from closed-loop states.")
parser.add_argument("--states", type=Path, action="append", required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--rounds", type=int, default=8, help="Starts per environment.")
parser.add_argument("--horizon", type=int, default=50, help="Teacher steps from each start.")
parser.add_argument("--history", type=int, default=26, help="Closed-loop steps kept before.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--checkpoint", type=Path, default=None, help="TorchScript teacher (default: pinned)."
)
parser.add_argument("--task-id", default=None, help="The teacher's task (default: the walk's).")
parser.add_argument(
    "--fixed-command", action="store_true", help="No heading control: hold the command."
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import quat_apply

import humanoid_transfer.tasks  # noqa: F401
from humanoid_transfer.common.similarity import UnitSystem
from humanoid_transfer.g1.tasks.g1_walk import TASK_ID
from humanoid_transfer.transfer.latent.skill_torch import BodyConstants, skill_state

TEACHER = _REPO_ROOT / "g1" / "policy" / "teacher_policy.pt"
BODY_CARD = _REPO_ROOT / "g1" / "body" / "body_card.json"
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
KEYS = ("joint_pos", "joint_vel", "base_pos", "base_quat", "base_lin_vel", "base_ang_vel",
        "command", "skill_state", "obs")  # fmt: skip


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    torch.manual_seed(args_cli.seed)
    rng = np.random.default_rng(args_cli.seed)
    task_id = args_cli.task_id or TASK_ID
    checkpoint = args_cli.checkpoint or TEACHER
    cfg = parse_env_cfg(task_id, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.episode_length_s = 1.0e6  # a start is never cut by a time-out
    env = gym.make(task_id, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    robot = inner.scene["robot"]
    sensor = inner.scene["contact_forces"]
    command_term = inner.command_manager.get_term("base_velocity")
    teacher = torch.jit.load(str(checkpoint), map_location=device).eval()

    card = json.loads(BODY_CARD.read_text(encoding="utf-8"))
    foot_ids, _ = robot.find_bodies(FEET, preserve_order=True)
    sensor_ids, _ = sensor.find_bodies(FEET, preserve_order=True)
    base_ids, _ = robot.find_bodies("pelvis")
    view = robot.root_physx_view
    masses = view.get_masses()[0].to(device, torch.float32)
    com_offset = robot.data.body_com_pos_b[0].to(torch.float32)
    sole = torch.tensor([card["foot"][n]["sole_point_link"] for n in FEET], device=device)
    units = UnitSystem(card["leg_length"], float(masses.sum()))
    body = BodyConstants(
        masses=masses,
        inertia_local=view.get_inertias()[0].to(device, torch.float32).reshape(-1, 3, 3),
        base_index=int(base_ids[0]),
        foot_index=(int(foot_ids[0]), int(foot_ids[1])),
        sole_from_com=sole - com_offset[foot_ids],
        units=units,
    ).to(device)

    def z_now() -> torch.Tensor:
        data = robot.data
        origins = inner.scene.env_origins[:, None, :]
        return skill_state(
            body,
            data.body_link_quat_w,
            data.body_com_pos_w - origins,
            data.body_com_lin_vel_w,
            data.body_com_ang_vel_w,
            sensor.data.net_forces_w[:, sensor_ids],
        )

    # Candidate starts: a full history inside one episode, away from any fall.
    pools = []
    for path in args_cli.states:
        data = np.load(path)
        names = [str(n) for n in data["joint_names"]]
        if names != list(robot.data.joint_names):
            raise ValueError(f"{path}: joints are not this robot's")
        done = data["done"]
        history = args_cli.history
        clean = ~np.lib.stride_tricks.sliding_window_view(done, history + 1, axis=0).any(-1)
        t, n = np.nonzero(clean)
        pools.append(({k: data[k] for k in KEYS}, np.stack([t + history - 1, n], -1)))

    hist, future, commands, obs_hist, obs_future, alive = [], [], [], [], [], []
    env.reset()
    for _ in range(args_cli.rounds):
        source, index = pools[rng.integers(len(pools))]
        pick = index[rng.integers(len(index), size=args_cli.num_envs)]
        t, n = pick[:, 0], pick[:, 1]

        def take(key: str, src=source, tt=t, nn=n) -> torch.Tensor:
            return torch.as_tensor(src[key][tt, nn], device=device, dtype=torch.float32)

        ids = torch.arange(args_cli.num_envs, device=device)
        quat = take("base_quat")
        pose = torch.cat([take("base_pos") + inner.scene.env_origins, quat], -1)
        velocity = torch.cat(
            [quat_apply(quat, take("base_lin_vel")), quat_apply(quat, take("base_ang_vel"))], -1
        )
        robot.write_root_pose_to_sim(pose, env_ids=ids)
        robot.write_root_velocity_to_sim(velocity, env_ids=ids)
        robot.write_joint_state_to_sim(take("joint_pos"), take("joint_vel"), env_ids=ids)
        command = take("command")
        command_term.vel_command_b[:] = command
        command_term.is_standing_env[:] = command.norm(dim=-1) == 0
        if args_cli.fixed_command and hasattr(command_term, "is_heading_env"):
            command_term.is_heading_env[:] = False
        command_term.time_left[:] = 1.0e9
        inner.action_manager.reset()
        observation = inner.observation_manager.compute()["policy"]

        window = np.stack([source["skill_state"][t - k, n] for k in range(args_cli.history)][::-1])
        window_obs = np.stack([source["obs"][t - k, n] for k in range(args_cli.history)][::-1])
        if window_obs.shape[-1] != observation.shape[-1]:
            # A teacher that reads more than the capture kept (a height scan): the history's
            # observations are not the generator's input, so they are left empty.
            window_obs = np.zeros((*window_obs.shape[:-1], observation.shape[-1]), np.float32)
        steps, steps_obs = [], []
        living = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
        lived = []
        for _ in range(args_cli.horizon):
            steps_obs.append(observation.cpu().numpy())
            with torch.inference_mode():
                action = teacher(observation)
            out, _, terminated, truncated, _ = env.step(action)
            observation = out["policy"]
            # A reset environment no longer continues the start: stop counting it.
            living &= ~(terminated | truncated)
            lived.append(living.cpu().numpy().copy())
            steps.append(z_now().cpu().numpy())
        hist.append(window)
        future.append(np.stack(steps))
        commands.append(source["command"][t, n])
        obs_hist.append(window_obs)
        obs_future.append(np.stack(steps_obs))
        alive.append(np.stack(lived))

    skill = np.concatenate([np.concatenate(hist, 1), np.concatenate(future, 1)], 0)
    obs = np.concatenate([np.concatenate(obs_hist, 1), np.concatenate(obs_future, 1)], 0)
    living = np.concatenate(alive, 1)
    total, columns = skill.shape[:2]
    lives = np.concatenate([np.ones((args_cli.history, columns), bool), living], 0)
    done = np.zeros((total, columns), bool)
    done[-1] = True
    first_dead = np.argmin(lives, axis=0)
    died = ~lives.all(axis=0)
    done[first_dead[died], np.nonzero(died)[0]] = True
    terminated = done & ~lives
    target = np.zeros((total, columns), bool)
    target[args_cli.history - 1 :] = True
    target &= np.cumprod(lives, axis=0).astype(bool)
    command = np.broadcast_to(np.concatenate(commands, 0)[None], (total, columns, 3))

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        skill_state=skill.astype(np.float32),
        obs=obs.astype(np.float32),
        action_teacher=np.zeros((total, columns, 37), np.float32),
        command=command.astype(np.float32),
        done=done,
        terminated=terminated,
        target_mask=target,
        leg_length=np.float64(units.length),
        total_mass=np.float64(units.mass),
        control_dt=np.float32(inner.step_dt),
        joint_names=np.array(robot.data.joint_names),
        driver=np.array("teacher"),
        teacher_sha256=np.array(hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()),
        teacher=np.array(str(checkpoint)),
        task=np.array(task_id),
        states=np.array([str(p) for p in args_cli.states]),
        history=np.int64(args_cli.history),
    )
    speed = skill[args_cli.history :, :, 1] * units.velocity
    print(
        f"{columns} starts, teacher speed after 0.2 s {speed[10].mean():.3f}, after 1 s "
        f"{speed[-1].mean():.3f} m/s; falls {int(died.sum())}; written {args_cli.output}"
    )
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
