"""Check the physics prior's Jacobians and statics against the simulator itself.

:mod:`~humanoid_transfer.transfer.tasks.tracking.physics_prior` builds its targets
from PhysX's Jacobians and gravity terms, under conventions (world frame, a
floating base's six columns first, linear rows at the link origin) that are
worth measuring rather than assuming:

* **Kinematics.** Each env's leg joints are moved by a small random ``δq`` with
  the base held where it is; the soles' motion relative to the CoM, and each
  foot's rotation, are compared with ``J δq`` from the prior's own Jacobians.
* **Statics.** A body standing still on both feet holds itself with joint
  torques ``τ = g(q) − Σ_f J_fᵀ F_f``; with ``F_f`` the measured ground reactions
  (:func:`~...tracking.balance.foot_reactions`, checked against Newton) this
  predicts the actuators' torques, compared with the implicit drives' own
  (``applied_torque``) on the legs.

Usage::

    python transfer/scripts/check_prior.py --headless --output transfer/results/check_prior_g1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task-id", default="G1TrackPhysics-Play-v0")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--settle-steps", type=int, default=100)
parser.add_argument("--delta", type=float, default=0.01, help="Joint perturbation, rad.")
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402, F401
import torch  # noqa: E402
from humanoid_transfer.transfer.latent.whole_body import (  # noqa: E402
    com_jacobian,
    ground_reaction,
    point_jacobian,
)
from isaaclab.utils.math import matrix_from_quat, quat_apply  # noqa: E402


def kinematics(term, robot):
    """Soles relative to the CoM and the feet's rotations, world frame, and their Jacobians."""
    data = robot.data
    jacobian = term._fold(robot.root_physx_view.get_jacobians())
    link_pos, com = data.body_link_pos_w, data.body_com_pos_w
    rotation = matrix_from_quat(data.body_link_quat_w)
    masses = term._masses
    c = torch.einsum("b,nbi->ni", masses, com) / masses.sum()
    feet = term._feet
    soles = link_pos[:, feet] + torch.einsum("nfij,fj->nfi", rotation[:, feet], term._soles)
    relative = soles - c[:, None]
    relative_jac = (
        point_jacobian(jacobian[:, feet], link_pos[:, feet], soles)
        - com_jacobian(jacobian, link_pos, com, masses)[:, None]
    )
    return relative, rotation[:, feet], relative_jac, jacobian[:, feet, 3:]


def main() -> int:
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args_cli.task_id, num_envs=args_cli.num_envs)
    cfg.commands.base_velocity.use_encoder = False
    cfg.events.reset_robot_joints = None
    env = gym.make(args_cli.task_id, cfg=cfg)
    inner = env.unwrapped
    robot = inner.scene["robot"]
    term = inner.action_manager.get_term("joint_pos")
    command = inner.command_manager.get_term("base_velocity")
    env.reset()

    # Statics: stand still at the default pose (zero action is the default pose for the
    # plain tracker; here the residual's offset is the default pose on every joint).
    hold = robot.data.default_joint_pos.clone()
    statics = []
    for step in range(args_cli.settle_steps):
        robot.set_joint_position_target(hold)
        robot.set_joint_effort_target(torch.zeros_like(hold))
        robot.write_data_to_sim()
        inner.sim.step(render=False)
        robot.update(inner.physics_dt)
        if step >= args_cli.settle_steps - 20:
            jac = term._fold(robot.root_physx_view.get_jacobians())
            data = robot.data
            feet = term._feet
            quat = data.body_link_quat_w[:, feet]
            wrench = data.body_incoming_joint_wrench_b[:, feet]
            joint_pos = data.body_link_pos_w[:, feet]
            force, moment = ground_reaction(
                quat_apply(quat, wrench[..., :3]), quat_apply(quat, wrench[..., 3:]), joint_pos,
                data.body_com_pos_w[:, feet], command.body.masses[feet],
            )  # fmt: skip
            gravity = term._fold(robot.root_physx_view.get_gravity_compensation_forces())
            columns = term._ik_columns
            # The ground's whole wrench, about the foot link's origin where J is taken.
            contact = torch.einsum("nfid,nfi->nd", jac[:, feet, :3][..., columns], force)
            contact = contact + torch.einsum("nfid,nfi->nd", jac[:, feet, 3:][..., columns], moment)
            predicted = gravity.to(term.device)[:, columns] - contact
            applied = data.applied_torque[:, term._ik_joints]
            statics.append(torch.stack([predicted, applied], dim=0))
    statics = torch.stack(statics).mean(dim=0)  # (2, N, legs)
    predicted, applied = statics[0], statics[1]
    static_error = (predicted - applied).norm(dim=-1) / applied.norm(dim=-1).clamp(min=1e-6)
    flipped = (predicted + applied).norm(dim=-1) / applied.norm(dim=-1).clamp(min=1e-6)

    # Kinematics: move the legs by δq with the base held, compare with J δq.
    root = robot.data.root_state_w.clone()
    q, qd = robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
    relative, foot_rot, relative_jac, turn_jac = kinematics(term, robot)
    generator = torch.Generator(device=q.device).manual_seed(0)
    delta = torch.zeros_like(q)
    legs = term._ik_joints
    delta[:, legs] = args_cli.delta * torch.randn(
        len(q), len(legs), device=q.device, generator=generator
    )
    columns = torch.arange(q.shape[1], device=q.device) + 6
    moved_q = q + delta
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(moved_q, qd)
    inner.sim.forward()
    robot.update(inner.physics_dt)
    relative_after, foot_rot_after, _, _ = kinematics(term, robot)
    measured = relative_after - relative
    predicted_move = torch.einsum("nfid,nd->nfi", relative_jac[..., columns], delta)
    turn = 0.5 * torch.linalg.cross(foot_rot, foot_rot_after, dim=-2).sum(dim=-1)
    predicted_turn = torch.einsum("nfid,nd->nfi", turn_jac[..., columns], delta)
    move_error = (measured - predicted_move).norm(dim=-1) / measured.norm(dim=-1).clamp(min=1e-9)
    turn_error = (turn - predicted_turn).norm(dim=-1) / turn.norm(dim=-1).clamp(min=1e-9)

    report = {
        "task": args_cli.task_id,
        "kinematics": {
            "delta_rad": args_cli.delta,
            "sole_motion_relative_error_median": float(move_error.median()),
            "foot_rotation_relative_error_median": float(turn_error.median()),
            "sole_motion_measured_mm_median": float(1000 * measured.norm(dim=-1).median()),
        },
        "statics": {
            "relative_error_median": float(static_error.median()),
            "relative_error_if_sign_flipped_median": float(flipped.median()),
            "applied_norm_median": float(applied.norm(dim=-1).median()),
            "predicted_first_env": [round(float(x), 2) for x in predicted[0]],
            "applied_first_env": [round(float(x), 2) for x in applied[0]],
        },
    }
    print(json.dumps(report, indent=2))
    if args_cli.output is not None:
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
