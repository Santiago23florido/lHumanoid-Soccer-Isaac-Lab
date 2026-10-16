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
* with ``--procedural-fraction`` or ``--gait-probes``, plans from the
  point-mass biped instead of the task (``transfer/latent/procedural.py``):
  per step which envs follow it (``procedural``), and whether their biped has
  fallen or is recovering; with probes, which probe each env follows
  (``probe_index``, ``probes``).

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


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args_cli.task_id, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    command_cfg = getattr(cfg.commands, COMMAND)
    if args_cli.reference is not None:
        command_cfg.autonomous = args_cli.reference == "autonomous"
    # The deployed path reads the body's encoder; --contract-from-truth the true state.
    command_cfg.use_encoder = not args_cli.contract_from_truth
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

    def skill_now() -> np.ndarray:
        return term.generator.skill_norm.inverse(term.true_skill()).cpu().numpy()

    obs, _ = env.reset()
    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            action = driver(obs["policy"])
        obs, _, _, _, _ = env.step(action)

    keys = (
        "skill_state", "command", "action_executed", "base_lin_vel", "base_ang_vel",
        "base_quat", "base_pos", "contract_due", "contract_error", "contract_error_other",
        "contract_error_state", "contract_fields", "contract_age", "done", "terminated",
        "encoder_input", "procedural", "biped_fallen", "biped_recovering", "estimate",
        "balance", "biped_phase", "plan_gain",
    )  # fmt: skip
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
        record["plan_gain"].append(term.plan_gain().cpu().numpy())
        with torch.inference_mode():
            action = driver(obs["policy"])
        record["action_executed"].append(action.cpu().numpy())
        obs, _, terminated, truncated, _ = env.step(action)
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
    if probes is not None:
        arrays["probe_index"] = term.probe_index.cpu().numpy()
        arrays["probes"] = np.array(json.dumps(probes))
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.output,
        # No teacher on this body: the label slot repeats the executed action.
        action_teacher=arrays["action_executed"],
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
