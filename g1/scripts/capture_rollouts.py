"""Record what the teacher does, in the form the student can be shown.

A trained teacher is only useful to the transfer study once its behaviour is
written down. What gets written down is the decision, because it fixes which of
the candidate signals can be compared at all:

``joint``
    Positions and velocities of every actuated joint, plus the names, so the
    correspondence map can be applied afterwards rather than baked in here.
``centroidal``
    Centre-of-mass position, velocity and acceleration, and centroidal angular
    momentum with its orbital and spin parts kept apart.
``contact``
    The 3D net contact force on each foot.
``skill_state``
    The 27-number embodiment-agnostic state of
    :mod:`humanoid_transfer.common.skill_state`, in the G1's own units -- the
    anchor the shared latent space is trained against.
``policy``
    The observation the teacher saw, the action it chose and the action that
    was executed. With ``--action-noise`` the two differ (DART: the executed
    action is perturbed so the data visits states a learner will drift into,
    and the label stays the teacher's clean choice).

Everything except the skill state is stored in the **teacher's own units**.
The skill state is the one quantity that is not, because being in body units is
what it is for.

Two earlier versions of this script recorded quantities that were not what
they said. Both are fixed here and worth knowing about when reading old
archives:

* **Foot force was all zeros**, in all 25 600 samples. The foot indices came
  from the articulation and were used to index the contact sensor, whose body
  order is its own (USD traversal of the prims carrying the contact API). The
  sensor's indices now come from the sensor, and the two name lists are
  checked against each other.
* **Angular momentum was the orbital term only**, although the docstring
  listed the spin term too. It is now computed by
  :mod:`humanoid_transfer.common.centroidal`, with both terms recorded, so
  which one dominates is a measurement.

The teacher defaults to the pinned ``g1/policy/teacher_policy.pt`` and its
sha256 is checked against the manifest and stored in the archive, so a rollout
says which teacher produced it.

Usage::

    # the teacher's gait, as the certification and gait statistics read it
    python g1/scripts/capture_rollouts.py --headless --task play

    # training data for the latent model: command range, noisy actions
    python g1/scripts/capture_rollouts.py --headless --task train \
        --num-envs 128 --steps 1000 --action-noise 0.1 \
        --output g1/rollouts/teacher_dart.npz
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
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

PINNED_TEACHER = _REPO_ROOT / "g1" / "policy" / "teacher_policy.pt"
TEACHER_MANIFEST = _REPO_ROOT / "g1" / "policy" / "teacher_policy.json"
BODY_CARD = _REPO_ROOT / "g1" / "body" / "body_card.json"

parser = argparse.ArgumentParser(description="Record teacher rollouts for transfer.")
parser.add_argument(
    "--checkpoint", type=Path, default=PINNED_TEACHER, help="TorchScript teacher policy."
)
parser.add_argument(
    "--task",
    choices=("play", "train"),
    default="play",
    help="play: one fixed speed, no noise. train: the teacher's command range and noise.",
)
parser.add_argument(
    "--task-id",
    default=None,
    help="Any registered task id, instead of the teacher's play/train pair (e.g. G1Track-Play-v0). "
    "The driver reads the 'policy' observations; the teacher reads 'teacher' when the task has it.",
)
parser.add_argument("--num-envs", type=int, default=32, help="Parallel rollouts.")
parser.add_argument("--steps", type=int, default=600, help="Control steps to record.")
parser.add_argument(
    "--settle-steps", type=int, default=100, help="Steps discarded before recording."
)
parser.add_argument(
    "--action-noise",
    type=float,
    default=0.0,
    help="Std of Gaussian noise added to the executed action, in action units (DART).",
)
parser.add_argument(
    "--driver-policy",
    type=Path,
    default=None,
    help="TorchScript policy that drives the robot instead of the teacher (DAgger). "
    "The teacher still labels every step. A driver with a reset(done) method (a memory) "
    "is told which environments were reset after every step.",
)
parser.add_argument(
    "--raw-envs", type=int, default=4, help="Environments whose per-body state is kept raw."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output",
    type=Path,
    default=_REPO_ROOT / "g1" / "rollouts" / "teacher.npz",
    help="Where to write the archive.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import humanoid_transfer.tasks  # noqa: F401  (registers the task ids)
from humanoid_transfer.common.centroidal import (
    angular_momentum,
    center_of_mass,
    com_acceleration,
    com_velocity,
)
from humanoid_transfer.common.frames import quat_to_matrix
from humanoid_transfer.common.similarity import UnitSystem
from humanoid_transfer.common.skill_state import SKILL_LAYOUT, MultibodySnapshot, encode
from humanoid_transfer.g1.tasks.g1_walk import PLAY_TASK_ID, TASK_ID
from humanoid_transfer.g1.tasks.g1_walk.g1_walk_env_cfg import FROUDE_MATCHED_SPEED

BASE_BODY = "pelvis"
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    task_id = args_cli.task_id or (PLAY_TASK_ID if args_cli.task == "play" else TASK_ID)
    cfg = parse_env_cfg(task_id, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    env = gym.make(task_id, cfg=cfg)
    inner = env.unwrapped
    device = inner.device
    robot = inner.scene["robot"]
    sensor = inner.scene["contact_forces"]
    num_envs = inner.num_envs

    digest = _sha256(args_cli.checkpoint)
    pinned_digest = json.loads(TEACHER_MANIFEST.read_text(encoding="utf-8"))["policy"]["sha256"]
    teacher_is_pinned = digest == pinned_digest
    if not teacher_is_pinned:
        print(f"WARNING: {args_cli.checkpoint} is not the pinned teacher (sha256 {digest[:12]})")

    teacher = torch.jit.load(str(args_cli.checkpoint), map_location=device).eval()
    driver = (
        torch.jit.load(str(args_cli.driver_policy), map_location=device).eval()
        if args_cli.driver_policy is not None
        else None
    )

    # Bodies, in each object's own order -- the bug this script used to have.
    base_ids, _ = robot.find_bodies(BASE_BODY)
    foot_ids, foot_names = robot.find_bodies(FOOT_BODIES, preserve_order=True)
    sensor_foot_ids, sensor_foot_names = sensor.find_bodies(FOOT_BODIES, preserve_order=True)
    if list(foot_names) != FOOT_BODIES or list(sensor_foot_names) != FOOT_BODIES:
        raise RuntimeError(
            f"foot bodies disagree: articulation {foot_names}, sensor {sensor_foot_names}"
        )

    card = json.loads(BODY_CARD.read_text(encoding="utf-8"))
    view = robot.root_physx_view
    masses = view.get_masses()[0].cpu().numpy().astype(float)
    inertia = view.get_inertias()[0].cpu().numpy().astype(float).reshape(-1, 3, 3)
    com_offset = robot.data.body_com_pos_b[0].cpu().numpy().astype(float)
    sole_from_com = np.stack(
        [
            np.array(card["foot"][name]["sole_point_link"]) - com_offset[foot_ids[i]]
            for i, name in enumerate(FOOT_BODIES)
        ]
    )
    units = UnitSystem(length=card["leg_length"], mass=float(masses.sum()))
    bodies = len(masses)
    masses_b = np.broadcast_to(masses, (num_envs, bodies))
    inertia_b = np.broadcast_to(inertia, (num_envs, bodies, 3, 3))
    raw = min(args_cli.raw_envs, num_envs)

    def body_state() -> dict[str, np.ndarray]:
        data = robot.data
        origins = inner.scene.env_origins.cpu().numpy()[:, None, :]
        return {
            "link_quat": data.body_link_quat_w.cpu().numpy().astype(float),
            "com_pos": data.body_com_pos_w.cpu().numpy().astype(float) - origins,
            "com_lin_vel": data.body_com_lin_vel_w.cpu().numpy().astype(float),
            "ang_vel": data.body_com_ang_vel_w.cpu().numpy().astype(float),
            "com_acc": data.body_com_acc_w[..., :3].cpu().numpy().astype(float),
            "foot_force": sensor.data.net_forces_w[:, sensor_foot_ids, :].cpu().numpy(),
        }

    def teacher_view(observation: dict) -> torch.Tensor:
        return observation["teacher"] if "teacher" in observation else observation["policy"]

    def act(observation: dict) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            label = teacher(teacher_view(observation))
            executed = driver(observation["policy"]) if driver is not None else label.clone()
        if args_cli.action_noise > 0.0:
            executed = executed + args_cli.action_noise * torch.randn_like(executed)
        return label, executed

    def episodes_ended(done: torch.Tensor) -> None:
        """A driver with a memory forgets the environments that were just reset."""
        if driver is not None and hasattr(driver, "reset"):
            with torch.inference_mode():
                driver.reset(done)

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        _, executed = act(obs)
        obs, _, terminated, truncated, _ = env.step(executed)
        episodes_ended(terminated | truncated)

    keys = (
        "obs", "action_teacher", "action_executed", "command", "done", "terminated",
        "skill_state",
        "joint_pos", "joint_vel", "com", "com_vel", "com_acc", "angular_momentum",
        "angular_momentum_spin", "foot_force", "foot_pos", "base_quat", "base_lin_vel",
        "raw_link_quat", "raw_com_pos", "raw_com_lin_vel", "raw_ang_vel",
    )  # fmt: skip
    contract = inner.command_manager.get_term("base_velocity")
    if not hasattr(contract, "last_error_other"):
        contract = None
    else:
        keys = keys + ("contract_error", "contract_error_other", "contract_age")
    record: dict[str, list] = {k: [] for k in keys}

    for _ in range(args_cli.steps):
        # Everything about the state is read *before* the step, so it describes
        # the state the recorded observation was computed from.
        state = body_state()
        link_rot = quat_to_matrix(state["link_quat"])
        snapshot = MultibodySnapshot(
            masses=masses_b,
            inertia_local=inertia_b,
            link_rot=link_rot,
            com_pos=state["com_pos"],
            com_lin_vel=state["com_lin_vel"],
            ang_vel=state["ang_vel"],
            foot_force=state["foot_force"],
            base_index=base_ids[0],
            foot_index=(foot_ids[0], foot_ids[1]),
            sole_from_com=sole_from_com,
        )
        momentum = angular_momentum(
            masses_b, inertia_b, link_rot, state["com_pos"], state["com_lin_vel"], state["ang_vel"]
        )

        policy_obs = teacher_view(obs)
        label, executed = act(obs)
        origins = inner.scene.env_origins

        record["obs"].append(policy_obs.cpu().numpy())
        record["action_teacher"].append(label.cpu().numpy())
        record["action_executed"].append(executed.cpu().numpy())
        record["command"].append(inner.command_manager.get_command("base_velocity").cpu().numpy())
        record["skill_state"].append(encode(snapshot, units))
        record["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
        record["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
        record["com"].append(center_of_mass(masses_b, state["com_pos"]))
        record["com_vel"].append(com_velocity(masses_b, state["com_lin_vel"]))
        record["com_acc"].append(com_acceleration(masses_b, state["com_acc"]))
        record["angular_momentum"].append(momentum.total)
        record["angular_momentum_spin"].append(momentum.spin)
        record["foot_force"].append(state["foot_force"])
        record["foot_pos"].append(
            (robot.data.body_link_pos_w[:, foot_ids, :] - origins.unsqueeze(1)).cpu().numpy()
        )
        record["base_quat"].append(robot.data.root_quat_w.cpu().numpy())
        record["base_lin_vel"].append(robot.data.root_lin_vel_b.cpu().numpy())
        record["raw_link_quat"].append(state["link_quat"][:raw])
        record["raw_com_pos"].append(state["com_pos"][:raw])
        record["raw_com_lin_vel"].append(state["com_lin_vel"][:raw])
        record["raw_ang_vel"].append(state["ang_vel"][:raw])

        obs, _, terminated, truncated, _ = env.step(executed)
        episodes_ended(terminated | truncated)
        if contract is not None:
            record["contract_error"].append(contract.last_error.cpu().numpy())
            record["contract_error_other"].append(contract.last_error_other.cpu().numpy())
            record["contract_age"].append(contract.age.cpu().numpy())
        record["done"].append((terminated | truncated).cpu().numpy())
        record["terminated"].append(terminated.cpu().numpy())

    arrays = {
        k: np.stack(v).astype(
            np.bool_
            if k in ("done", "terminated")
            else np.int64
            if k == "contract_age"
            else np.float32
        )
        for k, v in record.items()
    }

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        joint_names=np.array(robot.data.joint_names),
        foot_names=np.array(FOOT_BODIES),
        body_names=np.array(robot.body_names),
        skill_components=np.array(SKILL_LAYOUT.component_names()),
        control_dt=np.float32(inner.step_dt),
        commanded_speed=np.float32(FROUDE_MATCHED_SPEED),
        total_mass=np.float64(masses.sum()),
        leg_length=np.float64(units.length),
        masses=masses,
        inertia_link=inertia,
        com_offset_link=com_offset,
        base_index=np.int64(base_ids[0]),
        foot_index=np.array(foot_ids, dtype=np.int64),
        sole_from_com=sole_from_com,
        task=np.array(task_id),
        action_noise=np.float32(args_cli.action_noise),
        driver=np.array(str(args_cli.driver_policy or "teacher")),
        teacher_sha256=np.array(digest),
        teacher_is_pinned=np.bool_(teacher_is_pinned),
        seed=np.int64(args_cli.seed),
        settle_steps=np.int64(args_cli.settle_steps),
        **arrays,
    )

    weight = masses.sum() * units.gravity
    load = arrays["foot_force"][..., 2].sum(axis=-1) / weight
    spin = np.linalg.norm(arrays["angular_momentum_spin"], axis=-1)
    total = np.linalg.norm(arrays["angular_momentum"], axis=-1)
    speed = float(np.linalg.norm(arrays["com_vel"][..., :2], axis=-1).mean())
    size_mb = args_cli.output.stat().st_size / 1e6

    print("\n" + "=" * 78)
    print("TEACHER ROLLOUTS CAPTURED")
    print("=" * 78)
    print(f"  file                  : {args_cli.output}  ({size_mb:.1f} MB)")
    print(f"  task                  : {task_id}")
    print(f"  shape                 : {args_cli.steps} steps x {num_envs} envs")
    print(f"  teacher               : {digest[:12]}  pinned={teacher_is_pinned}")
    print(
        f"  driver / action noise : {args_cli.driver_policy or 'teacher'} / {args_cli.action_noise}"
    )
    print(f"  mean |v_xy|           : {speed:.3f} m/s")
    print(f"  mean foot load / mg   : {load.mean():.4f}   (1.0 if the forces are real)")
    print(f"  median |k_spin|/|k|   : {np.median(spin / np.maximum(total, 1e-9)):.3f}")
    print(
        f"  episode ends          : {int(arrays['done'].sum())} "
        f"({int(arrays['terminated'].sum())} terminated, the rest timed out)"
    )
    print("=" * 78 + "\n")

    env.close()
    return 0


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
