"""Build a deployment bundle for the real NAO: the tracker, its body model and its plans.

One Isaac Lab run with the robot's own sensors and estimator in the loop
(``humanoid_transfer.nao.sim2real``), for each command schedule:

* the plans the generator draws from the robot's standing state, as the
  tracker sees them at every control step (``plans/<name>.npz``: ``view``,
  ``has_plan``, ``path`` relative to where they start) -- the generator never
  runs on the robot;
* a parity record (``parity/<name>.npz``): every step's sensor readings as the
  estimator took them, the policy's observation and its action, for
  ``nao/scripts/check_runtime.py`` to replay through the NumPy runtime.

Plus ``body_model.json`` (from the URDF: never committed), ``actor.npz`` (the
TorchScript actor's weights, its normaliser and, symmetrised, its mirror) and
``runtime.json``. The bundle goes to ``nao/bundles/<name>/`` (ignored by git).

Usage (with the environment of ``nao/scripts/sensitivity.sh``)::

    python nao/scripts/export_runtime.py --headless --policy <run>/exported/model_3498_sym.pt \
        --bundle nao/bundles/r1_3498_sym
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from isaaclab.app import AppLauncher

SPEED = 0.15388
"""0.25 m/s on the G1 in the NAO's units (``nao/scripts/sensitivity.sh``)."""
TURN = 0.3 * 1.6246
SCHEDULES = {
    "stand": [(12.0, 0.0, 0.0, 0.0)],
    "forward": [(2.0, 0.0, 0.0, 0.0), (10.0, SPEED, 0.0, 0.0), (3.0, 0.0, 0.0, 0.0)],
    "turning": [(2.0, 0.0, 0.0, 0.0), (10.0, 1.2 * SPEED, 0.8 * SPEED, TURN), (3.0, 0.0, 0.0, 0.0)],
}
"""Each plan: segments of (seconds, vx m/s, vy m/s, wz rad/s); the paper's commands."""
ROBOT = {
    "sensors": True, "estimator": True, "odometry": True, "loads": "fsr",
    "fsr_noise": 0.0, "gyro_noise": 0.0,
}  # fmt: skip
"""The robot's pipeline, noiseless: the runtime replays the same readings."""

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--policy", type=Path, required=True, help="TorchScript actor (symmetrised).")
parser.add_argument(
    "--actor-npz",
    type=Path,
    default=None,
    help="The bundle's actor.npz as written elsewhere (a student's: nao/scripts/"
    "student_bundle.py), instead of one read from --policy, which still drives the simulator.",
)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--task-id", default="NaoTrack-Play-v0")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--schedules", nargs="+", default=list(SCHEDULES))
parser.add_argument("--robot-model", default=None, help="Settings JSON (default: ROBOT).")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import humanoid_transfer.tasks  # noqa: F401
import numpy as np
import torch
from humanoid_transfer.nao.sim2real.robot_model import apply_robot_model

COMMAND = "base_velocity"


def export_actor(path: Path, out: Path) -> dict:
    module = torch.jit.load(str(path), map_location="cpu").eval()
    state = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in module.state_dict().items()}
    prefix = "actor." if "actor.mean" in state else ""
    actor = getattr(module, "actor", module)
    for name, child in actor.mlp.named_children():
        if int(name) % 2 == 1 and child.original_name != "ELU":
            raise ValueError(f"{path}: layer {name} is {child.original_name}, not ELU")
    arrays = {"mean": state[f"{prefix}mean"], "scale": state[f"{prefix}scale"]}
    layers = sorted(int(k.split(".")[-2]) for k in state if k.endswith(".weight"))
    for i, layer in enumerate(layers):
        arrays[f"weight_{i}"] = state[f"{prefix}mlp.{layer}.weight"]
        arrays[f"bias_{i}"] = state[f"{prefix}mlp.{layer}.bias"]
    if prefix and "mirror_linear" in state:
        # nao/scripts/checkpoint_bundle.py --torchscript names its buffers as the bundle does.
        arrays.update({k: state[k] for k in ("mirror_linear", "mirror_offset", "action_joints")})
        arrays["action_offset"] = state["action_offset"]
    elif prefix:
        arrays.update(
            mirror_linear=state["linear"], mirror_offset=state["offset"],
            action_joints=state["joints"], action_offset=state["action_offset"],
        )  # fmt: skip
    np.savez(out, **arrays)
    return arrays


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    bundle = args_cli.bundle
    (bundle / "plans").mkdir(parents=True, exist_ok=True)
    (bundle / "parity").mkdir(parents=True, exist_ok=True)
    if args_cli.actor_npz is None:
        export_actor(args_cli.policy, bundle / "actor.npz")
    elif args_cli.actor_npz.resolve() != (bundle / "actor.npz").resolve():
        shutil.copy(args_cli.actor_npz, bundle / "actor.npz")
    driver = None

    cfg = parse_env_cfg(args_cli.task_id, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    command_cfg = getattr(cfg.commands, COMMAND)
    command_cfg.autonomous = True
    command_cfg.use_encoder = True
    command_cfg.estimator = "kinematic"
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    settings = ROBOT if args_cli.robot_model is None else json.loads(args_cli.robot_model)
    apply_robot_model(cfg, settings)
    env = gym.make(args_cli.task_id, cfg=cfg)
    inner = env.unwrapped
    term = inner.command_manager.get_term(COMMAND)
    robot = inner.scene["robot"]
    action_term = inner.action_manager.get_term("joint_pos")
    driver = torch.jit.load(str(args_cli.policy), map_location=inner.device).eval()
    dt = float(inner.step_dt)
    manager = inner.observation_manager
    names = manager.active_terms["policy"]
    widths = [int(np.prod(s)) for s in manager.group_obs_term_dim["policy"]]

    (bundle / "body_model.json").write_text(json.dumps(term.body_model_spec), encoding="utf-8")
    ids = action_term._joint_ids
    limits = robot.data.joint_pos_limits[0, ids].cpu().numpy()
    default = robot.data.default_joint_pos[0].cpu().numpy()
    joint_names = list(robot.data.joint_names)
    actuated = list(action_term._joint_names)
    held = {
        n: float(default[joint_names.index(n)])
        for n in term.robot_model.joint_names
        if n not in actuated
    }
    units = term.observed_units
    stiffness = robot.data.joint_stiffness[0].cpu().numpy()
    damping = robot.data.joint_damping[0].cpu().numpy()
    effort = robot.data.joint_effort_limits[0].cpu().numpy()
    lower_all = robot.data.joint_pos_limits[0, :, 0].cpu().numpy()
    upper_all = robot.data.joint_pos_limits[0, :, 1].cpu().numpy()
    drives = {
        n: {
            "stiffness": float(stiffness[i]),
            "damping": float(damping[i]),
            "effort": float(effort[i]),
            "lower": float(lower_all[i]),
            "upper": float(upper_all[i]),
            "default": float(default[i]),
        }
        for i, n in enumerate(joint_names)
    }
    runtime = {
        "control_dt": dt,
        "leg_length": float(term.units.length),
        "total_mass": float(term.units.mass),
        "actuated": actuated,
        "default_pos": [float(default[joint_names.index(n)]) for n in actuated],
        "held_pos": held,
        "action_scale": float(
            action_term._scale
            if not torch.is_tensor(action_term._scale)
            else action_term._scale[0, 0]
        ),  # fmt: skip
        "joint_lower": limits[:, 0].tolist(),
        "joint_upper": limits[:, 1].tolist(),
        "skill_mean": term.generator.skill_norm.mean.cpu().tolist(),
        "skill_std": term.generator.skill_norm.std.cpu().tolist(),
        "units": None if units is None else [u.cpu().tolist() for u in units],
        "plan_limit": 50.0,
        "policy_terms": [[n, w] for n, w in zip(names, widths, strict=True)],
        "history": {
            n: int(c.history_length)
            for n, c in zip(names, manager._group_obs_term_cfgs["policy"], strict=True)
            if c.history_length > 0
        },
        "pivot": term.estimator.pivot,
        "policy": str(args_cli.policy),
        "contract": str(command_cfg.contract),
        "robot_model": settings,
        "drives": drives,
    }
    if "robot_params" in names:
        from humanoid_transfer.nao.sim2real.params import PARAMS
        from humanoid_transfer.nao.tasks.nao_track.nao_track_sim2real_env_cfg import PARAM_RANGES

        runtime["robot_params"] = {"names": list(PARAMS), "ranges": [list(r) for r in PARAM_RANGES]}
    (bundle / "runtime.json").write_text(json.dumps(runtime, indent=1), encoding="utf-8")
    view = term.view
    length = term.units.length

    for name in args_cli.schedules:
        segments = SCHEDULES[name]
        steps = int(round(sum(s for s, *_ in segments) / dt))
        switches = []
        t = 0
        for seconds, vx, vy, wz in segments:
            switches.append((t, (vx, vy, wz)))
            t += int(round(seconds / dt))
        env.reset()
        # A reset inside a step updates the command before observing; env.reset() does
        # not. Do it here, so the first step is every later episode's first step.
        inner.command_manager.compute(dt=dt)
        obs = inner.observation_manager.compute()
        record = {k: [] for k in ("view", "has_plan", "path", "command", "q", "gravity", "gyro",
                                  "loads", "observation", "action", "fell")}  # fmt: skip
        # The path is stored relative to where the first plan starts: drawn on the step after
        # the reset (a command does not draw on its reset step), from the body's frame then.
        anchor_com = anchor_yaw = None
        fell = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
        for k in range(steps):
            for at, (vx, vy, wz) in switches:
                if k == at and k > 0:
                    term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=inner.device)
                    term.replan[:] = True
            index = term.age[:, None] + torch.arange(view, device=inner.device)
            planned = term.integrals[term._rows[:, None], index]
            c, s = torch.cos(term.start_yaw)[:, None], torch.sin(term.start_yaw)[:, None]
            ref_x = term.start_com[:, None, 0] + length * (
                c * planned[..., 0] - s * planned[..., 1]
            )
            ref_y = term.start_com[:, None, 1] + length * (
                s * planned[..., 0] + c * planned[..., 1]
            )
            ref_yaw = term.start_yaw[:, None] + planned[..., 2]
            if anchor_com is None and bool(term.has_plan.all()):
                anchor_com, anchor_yaw = term.start_com.clone(), term.start_yaw.clone()
            # Before the first plan the path is not observed (zero); any origin will do.
            origin = term.start_com if anchor_com is None else anchor_com
            origin_yaw = term.start_yaw if anchor_yaw is None else anchor_yaw
            ca, sa = torch.cos(origin_yaw)[:, None], torch.sin(origin_yaw)[:, None]
            dx, dy = ref_x - origin[:, None, 0], ref_y - origin[:, None, 1]
            path = torch.stack(
                [ca * dx + sa * dy, -sa * dx + ca * dy, ref_yaw - origin_yaw[:, None]], -1
            )
            q, gravity, gyro, loads = (x.cpu().double().numpy() for x in term.robot_inputs)
            record["view"].append(term.upcoming().cpu().numpy())
            record["has_plan"].append(term.has_plan.cpu().numpy())
            record["path"].append(path.cpu().numpy())
            record["command"].append(term.vel_command_b.cpu().numpy())
            record["q"].append(q)
            record["gravity"].append(gravity)
            record["gyro"].append(gyro)
            record["loads"].append(loads)
            record["observation"].append(obs["policy"].cpu().numpy())
            with torch.inference_mode():
                action = driver(obs["policy"])
            record["action"].append(action.cpu().numpy())
            obs, _, terminated, truncated, _ = env.step(action)
            fell |= terminated | truncated
            record["fell"].append(fell.cpu().numpy().copy())
        arrays = {k: np.stack(v) for k, v in record.items()}
        standing = np.flatnonzero(~arrays["fell"][-1])
        if len(standing) == 0:
            print(f"{name}: every environment fell; no plans written")
            continue
        chosen = int(standing[0])
        np.savez_compressed(
            bundle / "plans" / f"{name}.npz",
            view=arrays["view"][:, chosen].astype(np.float64),
            has_plan=arrays["has_plan"][:, chosen],
            path=arrays["path"][:, chosen].astype(np.float64),
            command=arrays["command"][:, chosen].astype(np.float64),
            dt=np.float64(dt),
            env=np.int64(chosen),
        )
        np.savez_compressed(bundle / "parity" / f"{name}.npz", **arrays, env=np.int64(chosen))
        print(
            f"{name}: {steps} steps x {inner.num_envs}, fell {int(arrays['fell'][-1].sum())}, "
            f"plans from env {chosen}"
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
