"""Capture a velocity policy on any body, in the form ``evaluate_contract.py`` reads.

``capture_contract.py`` needs a contract task, whose command term computes ``z*``. A
velocity task (Isaac Lab's, or the NAO oracle's ``NaoVelocity*``) has none, so here
``z*`` comes from the same GPU transcription
(:func:`~humanoid_transfer.transfer.latent.skill_torch.skill_state`) with the body
described as the contract tasks describe it: its masses and inertias from PhysX, its
sole points from the body's own record (``g1/body/body_card.json``; the NAO's URDF).

The archive has what the L4 verdict reads -- ``skill_state``, ``done``, ``terminated``,
``command``, the body's units and control step -- so an oracle is scored against the
G1's teacher exactly as a zero-shot tracker is.

Usage::

    python transfer/scripts/capture_velocity.py --headless --body nao \
        --task-id NaoVelocityFlat-Play-v0 --driver-policy <run>/exported/model_1499.pt \
        --command 0.154 0 0 --output nao/rollouts/oracle_flat_forward.npz
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "source" / "humanoid_transfer"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture a velocity policy on any body.")
parser.add_argument(
    "--task-id", required=True, help="A velocity task, e.g. NaoVelocityFlat-Play-v0."
)
parser.add_argument("--body", choices=("g1", "nao"), required=True)
parser.add_argument("--driver-policy", type=Path, required=True, help="TorchScript policy.")
parser.add_argument("--command", type=float, nargs=3, required=True, metavar=("VX", "VY", "WZ"))
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--settle-steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_transfer.tasks  # noqa: F401
from humanoid_transfer.common.similarity import UnitSystem
from humanoid_transfer.common.skill_state import SKILL_LAYOUT
from humanoid_transfer.transfer.latent.skill_torch import BodyConstants, skill_state

COMMAND = "base_velocity"


def body_record(body: str) -> tuple[str, list[str], list[list[float]], float]:
    """The base body, the feet, each sole point in its foot's link frame, the leg length."""
    if body == "g1":
        card = json.loads((_REPO_ROOT / "g1" / "body" / "body_card.json").read_text("utf-8"))
        feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
        return (
            "pelvis",
            feet,
            [card["foot"][f]["sole_point_link"] for f in feet],
            card["leg_length"],
        )
    from humanoid_transfer.nao.assets import nao_kinematics as nk
    from humanoid_transfer.nao.tasks.nao_track.nao_track_env_cfg import sole_points
    from humanoid_transfer.transfer.embodiment import NAO

    return "base_link", list(nk.FOOT_BODIES), sole_points(), NAO.leg_length


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args_cli.task_id, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    command_cfg = getattr(cfg.commands, COMMAND)
    vx, vy, wz = args_cli.command
    command_cfg.ranges.lin_vel_x = (vx, vx)
    command_cfg.ranges.lin_vel_y = (vy, vy)
    command_cfg.ranges.ang_vel_z = (wz, wz)
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    env = gym.make(args_cli.task_id, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    robot = inner.scene["robot"]
    sensor = inner.scene["contact_forces"]
    driver = torch.jit.load(str(args_cli.driver_policy), map_location=device).eval()

    base, feet, soles, leg_length = body_record(args_cli.body)
    foot_ids, foot_names = robot.find_bodies(feet, preserve_order=True)
    sensor_ids, sensor_names = sensor.find_bodies(feet, preserve_order=True)
    if list(foot_names) != feet or list(sensor_names) != feet:
        raise RuntimeError(f"foot bodies disagree: {foot_names} / {sensor_names}")
    base_ids, _ = robot.find_bodies(base)
    view = robot.root_physx_view
    masses = view.get_masses()[0].to(device, torch.float32)
    inertia = view.get_inertias()[0].to(device, torch.float32).reshape(-1, 3, 3)
    com_offset = robot.data.body_com_pos_b[0].to(torch.float32)
    sole = torch.tensor(soles, device=device, dtype=torch.float32)
    units = UnitSystem(float(leg_length), float(masses.sum()))
    constants = BodyConstants(
        masses=masses,
        inertia_local=inertia,
        base_index=int(base_ids[0]),
        foot_index=(int(foot_ids[0]), int(foot_ids[1])),
        sole_from_com=sole - com_offset[foot_ids],
        units=units,
    ).to(device)

    def skill_now() -> np.ndarray:
        data = robot.data
        origins = inner.scene.env_origins[:, None, :]
        z = skill_state(
            constants,
            data.body_link_quat_w,
            data.body_com_pos_w - origins,
            data.body_com_lin_vel_w,
            data.body_com_ang_vel_w,
            sensor.data.net_forces_w[:, sensor_ids],
        )
        return z.cpu().numpy()

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(driver(obs["policy"]))

    keys = (
        "skill_state", "command", "action_executed", "base_lin_vel", "base_ang_vel",
        "base_quat", "base_pos", "done", "terminated",
    )  # fmt: skip
    record: dict[str, list] = {k: [] for k in keys}
    origins = inner.scene.env_origins
    for _ in range(args_cli.steps):
        record["skill_state"].append(skill_now())
        record["command"].append(inner.command_manager.get_command(COMMAND).cpu().numpy())
        record["base_lin_vel"].append(robot.data.root_lin_vel_b.cpu().numpy())
        record["base_ang_vel"].append(robot.data.root_ang_vel_b.cpu().numpy())
        record["base_quat"].append(robot.data.root_quat_w.cpu().numpy())
        record["base_pos"].append((robot.data.root_pos_w - origins).cpu().numpy())
        with torch.inference_mode():
            action = driver(obs["policy"])
        record["action_executed"].append(action.cpu().numpy())
        obs, _, terminated, truncated, _ = env.step(action)
        record["done"].append((terminated | truncated).cpu().numpy())
        record["terminated"].append(terminated.cpu().numpy())

    arrays = {
        k: np.stack(v).astype(np.bool_ if k in ("done", "terminated") else np.float32)
        for k, v in record.items()
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        # No teacher labels this body: the label slot repeats the executed action.
        action_teacher=arrays["action_executed"],
        joint_names=np.array(robot.data.joint_names),
        skill_components=np.array(SKILL_LAYOUT.component_names()),
        control_dt=np.float32(inner.step_dt),
        leg_length=np.float64(units.length),
        total_mass=np.float64(units.mass),
        task=np.array(args_cli.task_id),
        driver=np.array(str(args_cli.driver_policy)),
        seed=np.int64(args_cli.seed),
        settle_steps=np.int64(args_cli.settle_steps),
        **arrays,
    )
    speed = SKILL_LAYOUT.unpack(arrays["skill_state"])["com_velocity"][..., 0].mean()
    print(
        f"{args_cli.task_id}: {args_cli.steps} x {inner.num_envs}, falls "
        f"{int(arrays['terminated'].sum())}, forward speed {speed:.4f} V "
        f"({speed * units.velocity:.3f} m/s); written {args_cli.output}"
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
