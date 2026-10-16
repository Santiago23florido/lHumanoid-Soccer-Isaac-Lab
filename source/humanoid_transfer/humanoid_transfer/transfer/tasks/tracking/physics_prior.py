"""A tracker's action as a correction to what the body's physics already says (decoder v2).

The tracker of §8 maps a plan of ``z*`` straight to joint targets, so it has to
learn the body's inverse kinematics and the torques that hold it up -- on the
plans it practised. Off them (a crouched gait, a high swing) it has nothing to
extrapolate with. Here both come from the body's model, every step, and the
policy's output is a residual on top (residual policy learning, Silver et al.
2018, Johannink et al. 2019; over an analytic controller as in Jeon et al.'s
Residual MPC, 2025, and HybridMimic, 2026):

* **Kinematic prior.** One damped least-squares step from the current joints
  towards the plan's next pose: each sole where the plan's ``foot_position``
  asks, relative to the CoM, and each foot flat -- both expressed in the base
  frame the plan's ``gravity`` gives (its tilt), so with the feet on the ground
  the step also brings the trunk to the planned tilt
  (:func:`~...latent.whole_body.damped_least_squares`). Only the leg joints move
  in it; the others keep the default pose.
* **Static prior.** The plan's loads, as forces along the legs through the CoM
  (the procedural family's pendulum), mapped through the soles' Jacobians, plus
  gravity compensation: ``tau = g(q) - sum_f J_f^T F_f``, sent as the actuators'
  feed-forward effort.
* **Trust.** With ``trust_plan`` the prior reads the plan fused with the body's
  state by the plan's own measured sampling noise (``latent/plan_trust.py``):
  ``now + K (plan - now)``, so a generator's arbitrary details are not turned
  into joint motion; the biped's plans keep ``K ≈ 1``. With ``slow_tilt`` the
  trunk's planned tilt is its mean over the plan's commitment: the feet and
  loads are what a plan says about contact, the trunk's sway within a stride is
  left to the residual's feedback.
* **The residual.** ``q_target = q_prior + scale * a``. The prior commutes with
  the body's mirror, so the residual mirrors as the joints do, without offset on
  the legs; the default pose keeps its offset on the other joints, as in
  :class:`~isaaclab.envs.mdp.JointPositionAction`.

The Jacobians, the gravity terms and the link states are the simulator's -- what
the body's own model would compute from its joint encoders and IMU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

from ....common.skill_state import SKILL_LAYOUT
from ...latent.whole_body import (
    com_jacobian,
    damped_least_squares,
    leg_forces,
    orientation_error,
    point_jacobian,
    tilt_from_gravity,
    yaw_rotation,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_FEET = SKILL_LAYOUT.slice_of("foot_position")
_LOAD = SKILL_LAYOUT.slice_of("foot_load")
_TILT = SKILL_LAYOUT.slice_of("gravity")
_GRAVITY = 9.81


class PhysicsPriorAction(JointPositionAction):
    """Joint position targets: the body's kinematic and static prior plus the policy's residual."""

    cfg: PhysicsPriorActionCfg

    def __init__(self, cfg: PhysicsPriorActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        robot = self._asset
        joints = (
            list(range(robot.num_joints))
            if isinstance(self._joint_ids, slice)
            else list(self._joint_ids)
        )
        self._action_joints = torch.tensor(joints, device=self.device)
        ik_ids, _ = robot.find_joints(cfg.ik_joint_names, preserve_order=True)
        missing = [j for j in ik_ids if j not in joints]
        if missing:
            raise ValueError(f"prior joints {missing} are not driven by this action term")
        self._ik_joints = torch.tensor(ik_ids, device=self.device)
        self._ik_slots = torch.tensor([joints.index(j) for j in ik_ids], device=self.device)
        # Jacobian columns: a floating base's six first.
        self._ik_columns = self._ik_joints + 6
        self._joint_columns = self._action_joints + 6
        # A mimic joint follows its driver one-to-one (the NAO's RHipYawPitch): the driver's
        # column carries both, and so does its torque.
        self._folds: list[tuple[int, int]] = []
        for follower, leader in (cfg.mimic or {}).items():
            follower_ids, _ = robot.find_joints(follower)
            leader_ids, _ = robot.find_joints(leader)
            self._folds.append((int(leader_ids[0]) + 6, int(follower_ids[0]) + 6))
        # The default pose stays the offset where the prior does not act; on the prior's
        # joints the residual is added to the prior's targets instead.
        offset = robot.data.default_joint_pos[:, joints].clone()
        offset[:, self._ik_slots] = 0.0
        self._offset = offset
        self._mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        self._mask[self._ik_slots] = True
        foot_ids, _ = robot.find_bodies(cfg.foot_bodies, preserve_order=True)
        base_ids, _ = robot.find_bodies(cfg.base_body)
        self._feet = list(foot_ids)
        self._base = int(base_ids[0])
        self._soles = torch.tensor(cfg.sole_points, dtype=torch.float32, device=self.device)
        self._masses = robot.root_physx_view.get_masses()[0].to(self.device, torch.float32)
        self._weight = float(self._masses.sum()) * _GRAVITY
        self._prior = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        """``q_prior - q`` on the prior's joints, zero elsewhere: what the policy is shown."""
        self._targets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._feedforward = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._stamp = -1

    # -- the prior ------------------------------------------------------------

    def prior(self) -> torch.Tensor:
        """``(N, action_dim)``: ``q_prior - q`` on the prior's joints, computed once per step."""
        stamp = self._env.common_step_counter
        if stamp != self._stamp:
            self._compute_prior()
            self._stamp = stamp
        return self._prior

    def _compute_prior(self) -> None:
        robot, data = self._asset, self._asset.data
        term = self._env.command_manager.get_term(self.cfg.command_name)
        length = term.units.length
        jacobian = self._fold(robot.root_physx_view.get_jacobians())
        link_pos, com = data.body_link_pos_w, data.body_com_pos_w
        rotation = matrix_from_quat(data.body_link_quat_w)
        masses = self._masses
        c = torch.einsum("b,nbi->ni", masses, com) / masses.sum()
        com_jac = com_jacobian(jacobian, link_pos, com, masses)
        feet = self._feet
        foot_rot = rotation[:, feet]
        soles = link_pos[:, feet] + torch.einsum("nfij,fj->nfi", foot_rot, self._soles)
        sole_jac = point_jacobian(jacobian[:, feet], link_pos[:, feet], soles)
        base = rotation[:, self._base]
        heading = yaw_rotation(base)

        rows = torch.arange(term.num_envs, device=self.device)
        window = term.age[:, None] + torch.arange(term.commit, device=self.device)
        ahead = term.plan[rows[:, None], window.clamp(max=term.plan.shape[1] - 1)]
        if self.cfg.trust_plan:
            # Each row by the gain of its own lead time since the plan was drawn.
            lead = window.clamp(max=term.plan.shape[1] - 1)
            gain = term.plan_gain()[rows[:, None], lead]
            ahead = term._true_now[:, None] + gain * (ahead - term._true_now[:, None])
        ahead = term.generator.skill_norm.inverse(ahead)
        plan = ahead[:, 0]
        tilt = ahead[..., _TILT].mean(dim=1) if self.cfg.slow_tilt else plan[:, _TILT]
        planned_feet = plan[:, _FEET].reshape(-1, 2, 3)
        load = plan[:, _LOAD].clamp(min=0.0)
        has_plan = term.has_plan

        q = data.joint_pos
        targets = data.default_joint_pos[:, self._action_joints].clone()
        if self.cfg.kinematic:
            # The planned pose in the base frame, where only the joints move the soles: the
            # plan's feet relative to the CoM and flat feet, seen from a base at the plan's tilt.
            to_planned = tilt_from_gravity(tilt).transpose(-1, -2)[:, None]
            to_base = base.transpose(-1, -2)[:, None]
            wanted = to_planned @ (length * planned_feet)[..., None]
            current = to_base @ (soles - c[:, None])[..., None]
            position_error = (wanted - current)[..., 0] / length
            relative_jac = sole_jac - com_jac[:, None]
            position_jac = (to_base @ relative_jac)[..., self._ik_columns] / length
            # Each foot flat, keeping its yaw against the heading; the base-fixed step aims at
            # that pose relative to the base as it is now.
            foot_yaw = yaw_rotation(heading.transpose(-1, -2)[:, None] @ foot_rot)
            turn_error = orientation_error(foot_rot, base[:, None] @ (to_planned @ foot_yaw))
            turn_jac = jacobian[:, feet, 3:][..., self._ik_columns]
            task = torch.cat([position_jac, turn_jac], dim=2).flatten(1, 2)
            error = torch.cat([position_error, turn_error], dim=2).flatten(1)
            step = damped_least_squares(task, error, self.cfg.damping)
            step = step.clamp(-self.cfg.max_step, self.cfg.max_step)
            limits = data.soft_joint_pos_limits[:, self._ik_joints]
            moved = q[:, self._ik_joints] + step
            moved = torch.maximum(torch.minimum(moved, limits[..., 1]), limits[..., 0])
            held = targets[:, self._ik_slots]
            targets[:, self._ik_slots] = torch.where(has_plan[:, None], moved, held)
        self._targets = targets
        prior = torch.zeros_like(self._prior)
        prior[:, self._ik_slots] = targets[:, self._ik_slots] - q[:, self._ik_joints]
        self._prior = prior

        feedforward = torch.zeros_like(self._feedforward)
        if self.cfg.feedforward:
            forces = leg_forces(load, length * planned_feet, self._weight)
            world = (heading[:, None] @ forces[..., None])[..., 0]
            gravity = robot.root_physx_view.get_gravity_compensation_forces().to(self.device)
            gravity = self._fold(gravity)[:, self._joint_columns]
            # The plan says how much each foot carries, not where under it: the forces act
            # below the ankle, so the prior asks no ankle torque and the residual supplies the
            # centre of pressure. (At the sole point, 3.6 cm ahead of the G1's ankle, they
            # pressed the toes down and tipped a still body backwards.)
            below_ankle = link_pos[:, feet].clone()
            below_ankle[..., 2] = soles[..., 2]
            ankle_jac = point_jacobian(jacobian[:, feet], link_pos[:, feet], below_ankle)
            contact = torch.einsum("nfid,nfi->nd", ankle_jac[..., self._joint_columns], world)
            torque = self.cfg.feedforward_gain * (gravity - contact)
            limit = data.joint_effort_limits[:, self._action_joints]
            torque = torch.maximum(torch.minimum(torque, limit), -limit)
            feedforward = torch.where(has_plan[:, None], torque, 0.0)
        self._feedforward = feedforward

    def _fold(self, columns: torch.Tensor) -> torch.Tensor:
        """Each mimic follower's last-axis column added to its driver's."""
        if not self._folds:
            return columns
        columns = columns.clone()
        for leader, follower in self._folds:
            columns[..., leader] += columns[..., follower]
        return columns

    # -- action term hooks --------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        self.prior()
        self._raw_actions[:] = actions
        residual = self._raw_actions * self._scale
        on_prior = torch.where(self._mask, self._targets, 0.0)
        self._processed_actions = residual + self._offset + on_prior
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(self._feedforward, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._stamp = -1


@configclass
class PhysicsPriorActionCfg(JointPositionActionCfg):
    """Configuration of :class:`PhysicsPriorAction`."""

    class_type: type = PhysicsPriorAction

    command_name: str = "base_velocity"
    """The :class:`~...tracking.command.ReferenceCommand` whose plan the prior follows."""

    ik_joint_names: list[str] = MISSING
    """The joints the kinematic prior moves (the legs)."""

    foot_bodies: list[str] = MISSING
    """Left and right foot bodies, in that order (the command's)."""

    sole_points: list[list[float]] = MISSING
    """Each foot's sole point in its link frame, metres (the command's)."""

    base_body: str = MISSING
    """The body whose yaw is the heading frame (the command's)."""

    mimic: dict[str, str] | None = None
    """Mimic joints the action does not drive, ``{follower: driver}``, one-to-one."""

    trust_plan: bool = False
    """Read the plan fused with the body's state by its measured sampling noise (needs the
    command's ``plan_trust``)."""

    slow_tilt: bool = False
    """Aim the trunk at the plan's tilt averaged over its commitment, not its next step's."""

    kinematic: bool = True
    """Move the legs' targets by the least-squares step towards the plan's feet."""

    damping: float = 0.05
    """Of the least-squares step, in leg lengths and radians."""

    max_step: float = 0.3
    """Largest change the step makes to one joint, radians."""

    feedforward: bool = True
    """Send gravity compensation minus the plan's leg forces as feed-forward effort."""

    feedforward_gain: float = 1.0


def prior_offset(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """``(N, joints)``: how far the prior's targets are from the joints, zero off its joints."""
    term = env.action_manager.get_term(action_name)
    return term.prior()


__all__ = ["PhysicsPriorAction", "PhysicsPriorActionCfg", "prior_offset"]
