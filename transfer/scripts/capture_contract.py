"""Capture a tracker following a contract, on any body, in the form the evaluator reads.

``g1/scripts/capture_rollouts.py`` records the G1 with its teacher labelling every
step; a second body has no teacher. This capture needs only a contract task
(``G1Track-Play-v0``, ``NaoTrack-Play-v0``, ...) and the tracker that drives it:

* ``skill_state``: the body's ``z*``, from the task's own GPU transcription
  (checked against the NumPy encoder by ``g1/scripts/check_contract_env.py``);
* ``contract_due``: the plan's ``z*`` for the state each step reaches, and the
  errors the reward scored (``contract_error``, ``_other``, ``_state``,
  ``_fields``, ``contract_age``);
* the body's units, so speeds compare across bodies in ``V = sqrt(g L)``;
* ``encoder_input``: what the body's encoder reads, to fit one
  (``train_body_encoder.py``);
* with ``--joints``, the joint angles and speeds and the projected gravity; with
  ``--policy-obs``, the tracker's observation (both before each step);
* with ``--replay-plans``, fixed plans every body follows alike: per step each env's
  instant along its chain (``replay_clock``), and its chain (``replay_chain``);
* with ``--procedural-fraction`` or ``--gait-probes``, plans from the
  point-mass biped instead of the task (``transfer/latent/procedural.py``):
  per step which envs follow it (``procedural``), and whether their biped has
  fallen or is recovering; with probes, which probe each env follows
  (``probe_index``, ``probes``);
* with ``--pact-model``, a PACT policy (``pact_gate_d.py``) drives the body and the
  ``--driver-policy`` tracker becomes the teacher: it labels every step
  (``action_teacher``) and, with ``--teacher-share``, drives that share of the
  environments (``teacher_driven``) -- DAgger's rollouts (G-D1).

Usage::

    python transfer/scripts/capture_contract.py --headless --task-id NaoTrack-Play-v0 \
        --driver-policy logs/rsl_rl/nao_track/<run>/exported/model_1500.pt \
        --reference physical --contract-from-truth --output nao/rollouts/track.npz
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

parser = argparse.ArgumentParser(description="Capture a tracker following a contract.")
parser.add_argument("--task-id", required=True, help="A contract task, e.g. NaoTrack-Play-v0.")
parser.add_argument("--driver-policy", type=Path, required=True, help="TorchScript tracker.")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--settle-steps", type=int, default=100)
parser.add_argument(
    "--command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Hold one command, in the body's own units (m/s, m/s, rad/s).",
)
parser.add_argument("--reference", choices=("autonomous", "physical"), default=None)
parser.add_argument(
    "--standing-fraction",
    type=float,
    default=None,
    help="Share of environments commanded to stand at each resample: starts and stops.",
)
parser.add_argument(
    "--resample-seconds", type=float, nargs=2, default=None, metavar=("LOW", "HIGH")
)
parser.add_argument(
    "--contract-from-truth",
    action="store_true",
    help="Feed the generator the true z* plus its noise, not the body's encoder.",
)
parser.add_argument(
    "--estimator",
    choices=("encoder", "kinematic"),
    default=None,
    help="On the deployed path: the learned encoder alone, or the body's kinematics for every "
    "field but the loads.",
)
parser.add_argument(
    "--noise-scale",
    type=float,
    default=None,
    help="On the true-state path: the noise on z* as a multiple of the contract's (0: none).",
)
parser.add_argument("--start-fraction", type=float, default=None)
parser.add_argument(
    "--procedural-fraction",
    type=float,
    default=None,
    help="Share of episodes whose plans come from the point-mass biped, with random schedules.",
)
parser.add_argument(
    "--gait-probes",
    type=Path,
    default=None,
    help="JSON list of fixed gaits of the family; env i follows probe i modulo their number.",
)
parser.add_argument("--procedural-seed", type=int, default=0)
parser.add_argument(
    "--replay-plans",
    type=Path,
    default=None,
    help="A replay file (latent/replay.py): env i follows chain i %% K, the same on every body.",
)
parser.add_argument("--replay-spacing", type=float, default=None)
parser.add_argument(
    "--push",
    type=float,
    nargs=2,
    default=None,
    metavar=("SPEED", "INTERVAL"),
    help="Robustness probe: push the base by up to SPEED (V = sqrt(g L)) in each horizontal "
    "direction every INTERVAL pendulum times (T = sqrt(L / g)) on average -- the same "
    "disturbance, dynamically, on every body.",
)
parser.add_argument(
    "--estimate-noise",
    type=float,
    default=None,
    help="Robustness probe on the deployed path: noise on the estimate, in multiples of the "
    "estimator's precision.",
)
parser.add_argument(
    "--joints",
    action="store_true",
    help="Also record the joint angles and speeds and the projected gravity before each step "
    "(what the PACT anchors and encoder read).",
)
parser.add_argument(
    "--policy-obs",
    action="store_true",
    help="Also record the tracker's observation before each step (distillation data).",
)
parser.add_argument(
    "--pact-model",
    type=Path,
    default=None,
    help="A PACT policy checkpoint (pact_gate_d.py) drives; --driver-policy labels each step.",
)
parser.add_argument("--pact-body", default=None, help="The body's name in transfer/bodies.")
parser.add_argument(
    "--teacher-share",
    type=float,
    default=0.0,
    help="With --pact-model: the share of environments the teacher drives (DAgger's mixture).",
)
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
from humanoid_transfer.common.skill_state import SKILL_LAYOUT
from humanoid_transfer.transfer.tasks.tracking.balance import balance_state

COMMAND = "base_velocity"


def drive_parameters(robot) -> dict | None:
    """Per joint, the drives' stiffness, damping and effort limit as the simulation holds
    them, and the default joint positions; None if any cannot be read."""
    try:
        count = len(robot.joint_names)
        out = {
            key: np.full(count, np.nan, dtype=np.float32)
            for key in ("drive_stiffness", "drive_damping", "drive_effort_limit")
        }
        for actuator in robot.actuators.values():
            index = actuator.joint_indices
            if isinstance(index, slice):
                index = list(range(count))[index]
            index = [int(i) for i in (index.tolist() if hasattr(index, "tolist") else index)]
            for key, value in (
                ("drive_stiffness", actuator.stiffness),
                ("drive_damping", actuator.damping),
                ("drive_effort_limit", actuator.effort_limit),
            ):
                row = value[0] if isinstance(value, torch.Tensor) and value.ndim == 2 else value
                row = row.detach().cpu().numpy() if isinstance(row, torch.Tensor) else row
                out[key][index] = np.broadcast_to(np.asarray(row, dtype=np.float32), len(index))
        out["default_joint_pos"] = robot.data.default_joint_pos[0].cpu().numpy()
        if any(np.isnan(v).any() for v in out.values()):
            raise ValueError("a joint without an actuator model")
        return out
    except Exception as error:  # noqa: BLE001
        print(f"[capture] drive parameters not recorded: {error}")
        return None


def make_pact(inner, term, robot):
    """A :class:`PactDriver` for this body, from ``--pact-model`` and its card."""
    from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
    from humanoid_transfer.transfer.latent.pact.descriptors import contact_joints, describe
    from humanoid_transfer.transfer.latent.pact.driver import PactDriver
    from pact_check_anchors import sole_override

    body = args_cli.pact_body
    if body is None:
        raise ValueError("--pact-model needs --pact-body")
    registry = json.loads((_REPO_ROOT / "transfer/bodies/bodies.json").read_text("utf-8"))
    entry = registry[body]
    card = json.loads((_REPO_ROOT / entry["card"]).read_text("utf-8"))
    unit = entry.get("leg_length_unit")
    descriptor = describe(card, leg_length=unit)
    if tuple(robot.joint_names) != tuple(descriptor.joint_names):
        raise ValueError(f"{body}: the card's joints are not the simulation's")
    anchors = KinematicAnchors(card, unit, sole_override(body))
    drive = DriveLoads(card, anchors)
    drives = drive_parameters(robot)
    if drives is None:
        raise ValueError("PACT's drive loads need the drives' gains")
    gains = (drives[k] for k in ("drive_stiffness", "drive_damping", "drive_effort_limit"))
    drive.set_drives(*gains)
    actions = inner.action_manager
    if len(actions.active_terms) != 1:
        raise ValueError(f"PACT drives one joint-position action, not {actions.active_terms}")
    action = actions.get_term(actions.active_terms[0])
    joints = action._joint_ids
    joints = joints if isinstance(joints, slice) else [int(j) for j in joints]
    scale = action._scale
    scale = scale[0] if isinstance(scale, torch.Tensor) else float(scale)
    offset = action._offset
    default = robot.data.default_joint_pos[0]
    if isinstance(offset, torch.Tensor) and not torch.allclose(offset[0], default[joints]):
        raise ValueError("PACT's targets are relative to the default pose; the action's are not")
    checkpoint = torch.load(args_cli.pact_model, map_location="cpu", weights_only=False)
    print(f"[capture] PACT {args_cli.pact_model} drives {body}; view {term.view}")
    return PactDriver(
        checkpoint,
        descriptor,
        contact_joints(card),
        anchors,
        drive,
        default,
        joints,
        scale,
        term.view,
        term.generator.skill_norm.mean,
        term.generator.skill_norm.std,
        float(inner.step_dt),
        inner.num_envs,
        inner.device,
    )


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args_cli.task_id, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    command_cfg = getattr(cfg.commands, COMMAND)
    if args_cli.reference is not None:
        command_cfg.autonomous = args_cli.reference == "autonomous"
    # The deployed path reads the body's encoder; --contract-from-truth the true state.
    command_cfg.use_encoder = not args_cli.contract_from_truth
    if args_cli.noise_scale is not None:
        command_cfg.noise_scale = args_cli.noise_scale
    if args_cli.estimator is not None:
        command_cfg.estimator = args_cli.estimator
    if args_cli.start_fraction is not None:
        command_cfg.start_fraction = args_cli.start_fraction
    if args_cli.standing_fraction is not None:
        command_cfg.rel_standing_envs = args_cli.standing_fraction
    if args_cli.resample_seconds is not None:
        command_cfg.resampling_time_range = tuple(args_cli.resample_seconds)
    if args_cli.procedural_fraction is not None:
        command_cfg.procedural_fraction = args_cli.procedural_fraction
    probes = None
    if args_cli.gait_probes is not None:
        probes = json.loads(args_cli.gait_probes.read_text(encoding="utf-8"))
        command_cfg.procedural_probes = probes
    command_cfg.procedural_seed = args_cli.procedural_seed
    if args_cli.replay_plans is not None:
        command_cfg.replay_plans = str(args_cli.replay_plans)
    if args_cli.replay_spacing is not None:
        command_cfg.replay_spacing = args_cli.replay_spacing
    if args_cli.estimate_noise is not None:
        command_cfg.estimate_noise = args_cli.estimate_noise
    if args_cli.push is not None:
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg

        speed, interval = args_cli.push
        length = float(command_cfg.leg_length)
        velocity = speed * (9.81 * length) ** 0.5
        seconds = interval * (length / 9.81) ** 0.5
        cfg.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.5 * seconds, 1.5 * seconds),
            params={"velocity_range": {"x": (-velocity, velocity), "y": (-velocity, velocity)}},
        )
    if args_cli.command is not None:
        vx, vy, wz = args_cli.command
        command_cfg.ranges.lin_vel_x = (vx, vx)
        command_cfg.ranges.lin_vel_y = (vy, vy)
        command_cfg.ranges.ang_vel_z = (wz, wz)
        command_cfg.heading_command = False
        command_cfg.rel_standing_envs = 0.0
    env = gym.make(args_cli.task_id, cfg=cfg)
    inner = env.unwrapped
    term = inner.command_manager.get_term(COMMAND)
    robot = inner.scene["robot"]
    driver = torch.jit.load(str(args_cli.driver_policy), map_location=inner.device).eval()
    if not hasattr(term, "due"):
        raise ValueError(f"{args_cli.task_id} has no contract command")
    pact = make_pact(inner, term, robot) if args_cli.pact_model is not None else None
    generator = torch.Generator(device="cpu").manual_seed(args_cli.seed)
    teacher_driven = (torch.rand(inner.num_envs, generator=generator) < args_cli.teacher_share).to(
        inner.device
    )

    def act(obs) -> tuple[torch.Tensor, torch.Tensor]:
        """``(executed, teacher)`` actions for this step."""
        with torch.inference_mode():
            teacher = driver(obs["policy"])
            if pact is None:
                return teacher, teacher
            student = pact(
                obs["policy"],
                robot.data.joint_pos,
                robot.data.joint_vel,
                robot.data.joint_pos_target,
                robot.data.projected_gravity_b,
                robot.data.root_ang_vel_b,
            )
        return torch.where(teacher_driven[:, None], teacher, student), teacher

    def step(action):
        out = env.step(action)
        if pact is not None:
            ended = torch.nonzero(out[2] | out[3]).flatten()
            if len(ended):
                pact.reset(ended)
        return out

    def skill_now() -> np.ndarray:
        return term.generator.skill_norm.inverse(term.true_skill()).cpu().numpy()

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        action, _ = act(obs)
        obs, _, _, _, _ = step(action)

    keys = (
        "skill_state", "command", "action_executed", "base_lin_vel", "base_ang_vel",
        "base_quat", "base_pos", "contract_due", "contract_error", "contract_error_other",
        "contract_error_state", "contract_fields", "contract_age", "done", "terminated",
        "encoder_input", "procedural", "biped_fallen", "biped_recovering", "estimate",
        "balance", "biped_phase", "plan_gain",
    )  # fmt: skip
    if args_cli.joints:
        keys += ("joint_pos", "joint_vel", "projected_gravity")
    # With --joints, also each joint's PD target before the step (the previous action's, as
    # the drives hold it) and the drives' gains once: what the PACT drive-load anchors read.
    # Never allowed to fail a capture: without these attributes the keys are left out.
    drives = drive_parameters(robot) if args_cli.joints else None
    if drives is not None:
        try:
            robot.data.joint_pos_target.cpu()
            keys += ("joint_pos_target",)
        except Exception as error:  # noqa: BLE001
            print(f"[capture] joint_pos_target not recorded: {error}")
    if args_cli.policy_obs:
        keys += ("policy_obs",)
    if pact is not None:
        keys += ("action_teacher",)
    replay = getattr(term, "replay", None)
    if replay is not None:
        keys += ("replay_clock",)
    record: dict[str, list] = {k: [] for k in keys}
    origins = inner.scene.env_origins
    for _ in range(args_cli.steps):
        # The state is read before the step; the plan's due state is the target of the
        # state this step reaches (the next row of skill_state).
        record["skill_state"].append(skill_now())
        # What the generator and the tracker read for that same state (computed when the
        # previous step ended): the row it estimates is this row of skill_state.
        record["estimate"].append(term.generator.skill_norm.inverse(term.estimate).cpu().numpy())
        record["encoder_input"].append(term.encoder_input().cpu().numpy())
        record["command"].append(inner.command_manager.get_command(COMMAND).cpu().numpy())
        record["base_lin_vel"].append(robot.data.root_lin_vel_b.cpu().numpy())
        record["base_ang_vel"].append(robot.data.root_ang_vel_b.cpu().numpy())
        record["base_quat"].append(robot.data.root_quat_w.cpu().numpy())
        record["base_pos"].append((robot.data.root_pos_w - origins).cpu().numpy())
        record["contract_due"].append(term.generator.skill_norm.inverse(term.due()).cpu().numpy())
        record["procedural"].append(term.procedural.cpu().numpy())
        biped = term.biped
        none = np.zeros(inner.num_envs, dtype=bool)
        record["biped_fallen"].append(none if biped is None else biped.fallen.cpu().numpy())
        record["biped_recovering"].append(none if biped is None else biped.recovering.cpu().numpy())
        # How the body is held up (each foot's centre of pressure and ground force), and where
        # in its gait the biped is: what pairs the same gait's states across bodies.
        record["balance"].append(balance_state(inner, COMMAND).cpu().numpy())
        phase = np.zeros(inner.num_envs, dtype=np.float32)
        record["biped_phase"].append(phase if biped is None else biped.phase.cpu().numpy())
        # How far each env's plan is trusted (1 everywhere without the command's plan_trust).
        record["plan_gain"].append(term.plan_gain()[:, 0].cpu().numpy())
        if args_cli.joints:
            record["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
            record["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
            record["projected_gravity"].append(robot.data.projected_gravity_b.cpu().numpy())
        if "joint_pos_target" in record:
            record["joint_pos_target"].append(robot.data.joint_pos_target.cpu().numpy())
        if args_cli.policy_obs:
            record["policy_obs"].append(obs["policy"].cpu().numpy())
        if replay is not None:
            # The instant of the chain this row's state is paired at (body times).
            record["replay_clock"].append(term.replay_clock.cpu().numpy())
        action, teacher = act(obs)
        record["action_executed"].append(action.cpu().numpy())
        if pact is not None:
            record["action_teacher"].append(teacher.cpu().numpy())
        obs, _, terminated, truncated, _ = step(action)
        record["contract_error"].append(term.last_error.cpu().numpy())
        record["contract_error_other"].append(term.last_error_other.cpu().numpy())
        record["contract_error_state"].append(term.last_error_state.cpu().numpy())
        record["contract_fields"].append(term.field_error.cpu().numpy())
        record["contract_age"].append(term.age.cpu().numpy())
        record["done"].append((terminated | truncated).cpu().numpy())
        record["terminated"].append(terminated.cpu().numpy())

    arrays = {
        k: np.stack(v).astype(
            np.bool_
            if k in ("done", "terminated", "procedural", "biped_fallen", "biped_recovering")
            else np.int64
            if k == "contract_age"
            else np.float32
        )
        for k, v in record.items()
    }
    units = term.units
    if drives is not None:
        arrays.update(drives)
    if replay is not None:
        arrays["replay_chain"] = term.replay_chain.cpu().numpy()
        arrays["replay_file"] = np.array(str(args_cli.replay_plans))
    if probes is not None:
        arrays["probe_index"] = term.probe_index.cpu().numpy()
        arrays["probes"] = np.array(json.dumps(probes))
    if pact is not None:
        arrays["teacher_driven"] = teacher_driven.cpu().numpy()
        arrays["pact_model"] = np.array(str(args_cli.pact_model))
        # What turns the teacher's actions into joint targets: q0 + scale * a on these joints.
        joints = np.arange(len(robot.joint_names))[pact.action_joints]
        scale = pact.action_scale
        scale = scale.cpu().numpy() if isinstance(scale, torch.Tensor) else scale
        arrays["action_joints"] = joints
        arrays["action_scale"] = np.broadcast_to(np.float32(scale), joints.shape).copy()
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        # Without --pact-model the tracker is its own label: the slot repeats the action.
        action_teacher=arrays.pop("action_teacher", arrays["action_executed"]),
        joint_names=np.array(robot.data.joint_names),
        skill_components=np.array(SKILL_LAYOUT.component_names()),
        control_dt=np.float32(inner.step_dt),
        leg_length=np.float64(units.length),
        total_mass=np.float64(units.mass),
        task=np.array(args_cli.task_id),
        driver=np.array(str(args_cli.driver_policy)),
        reference=np.array(
            args_cli.reference or ("autonomous" if term.cfg.autonomous else "physical")
        ),
        encoder=np.bool_(term.encoder is not None),
        resample_space=np.array(getattr(term, "resample_space", "linear")),
        push=np.array(args_cli.push if args_cli.push is not None else [0.0, 0.0]),
        estimate_noise=np.float64(args_cli.estimate_noise or 0.0),
        sole_points=np.array(term.cfg.sole_points, dtype=np.float64),
        seed=np.int64(args_cli.seed),
        settle_steps=np.int64(args_cli.settle_steps),
        **arrays,
    )
    z = arrays["skill_state"]
    speed = SKILL_LAYOUT.unpack(z)["com_velocity"][..., 0].mean()
    print(
        f"{args_cli.task_id}: {args_cli.steps} x {inner.num_envs}, falls "
        f"{int(arrays['terminated'].sum())}, forward speed {speed:.4f} V "
        f"({speed * units.velocity:.3f} m/s), median contract error "
        f"{np.median(arrays['contract_error']):.2f}; written {args_cli.output}"
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
