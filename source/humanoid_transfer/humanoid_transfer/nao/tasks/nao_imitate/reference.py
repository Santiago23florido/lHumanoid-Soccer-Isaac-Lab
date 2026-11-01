"""A reference motion as a command: the NAO's joint targets, retargeted from the G1.

The command term walks through a clip of ``transfer/scripts/retarget_reference.py``,
one of its rows per control step. Each episode starts at a random row of a random
clip with the NAO put in the reference's state there (reference state
initialisation, Peng et al. 2018), and ends, as a time-out, when the clip runs out.
``command`` is the base velocity the reference holds one step ahead, so Isaac Lab's
velocity-tracking terms apply unchanged.

The actions are joint targets about the reference (:class:`ReferenceJointPositionAction`):
the policy's zero action plays the reference back, and what it learns is the feedback
around it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class ReferenceMotionCommand(CommandTerm):
    cfg: ReferenceMotionCommandCfg

    def __init__(self, cfg: ReferenceMotionCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        data = np.load(cfg.reference)
        self.names = names = [str(n) for n in data["joint_names"]]
        self.joint_ids, _ = self.robot.find_joints(names, preserve_order=True)
        # The NAO's right hip yaw-pitch mimics the left one, multiplier 1 (nao.urdf).
        mimic, _ = self.robot.find_joints(["RHipYawPitch"])
        self.mimic_id, self.master = mimic[0], names.index("LHipYawPitch")

        def tensor(key: str) -> torch.Tensor:
            return torch.tensor(data[key], device=self.device, dtype=torch.float32)

        self.q, self.qd, self.base = (
            tensor("joint_pos"),
            tensor("joint_vel"),
            tensor("base_velocity"),
        )
        self.clip_start = torch.tensor(data["clip_start"], device=self.device)
        self.clip_length = torch.tensor(data["clip_length"], device=self.device)
        self.margin = max(cfg.offsets)
        self.frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.end = torch.zeros_like(self.frame)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.default = self.robot.data.default_joint_pos[:, self.joint_ids]
        self.metrics["joint_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """``(N, 3)``: the reference's base velocity one step ahead (vx, vy in the heading
        frame, m/s; yaw rate, rad/s)."""
        return self._command

    def ahead(self, offset: int) -> torch.Tensor:
        return torch.minimum(self.frame + offset, self.end)

    def target(self, offset: int = 1) -> torch.Tensor:
        """``(N, J)``: the reference's joint angles ``offset`` steps ahead."""
        return self.q[self.ahead(offset)]

    def target_rate(self, offset: int = 1) -> torch.Tensor:
        return self.qd[self.ahead(offset)]

    def ended(self) -> torch.Tensor:
        return self.frame + self.margin >= self.end

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if isinstance(env_ids, slice):
            env_ids = range(self.num_envs)[env_ids]
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(ids) == 0:
            return
        count = len(self.clip_start)
        if self.cfg.in_order:
            clip = ids % count
        else:
            clip = torch.randint(0, count, ids.shape, device=self.device)
        start, length = self.clip_start[clip], self.clip_length[clip]
        room = torch.clamp(length - self.margin - self.cfg.shortest, min=1)
        if self.cfg.from_start:
            offset = torch.zeros_like(room)
        else:
            offset = (torch.rand(ids.shape, device=self.device) * room).long()
        # ``frame`` is the row the state matches once the command has advanced: a reset inside a
        # step is followed by that step's advance (reward, reset, command, observation), so the
        # row the body is put in is one ahead of it.
        first = start + offset
        self.frame[ids] = first - 1
        self.end[ids] = start + length - 1
        self._command[ids] = self.base[first]

        # Reference state initialisation: the joints, and the base moving as the reference does.
        q, qd = self.q[first], self.qd[first]
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.joint_ids, env_ids=ids)
        self.robot.write_joint_state_to_sim(
            q[:, self.master : self.master + 1],
            qd[:, self.master : self.master + 1],
            joint_ids=[self.mimic_id],
            env_ids=ids,
        )
        w, x, y, z = self.robot.data.root_quat_w[ids].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v = self.base[first]
        velocity = torch.zeros(len(ids), 6, device=self.device)
        velocity[:, 0] = torch.cos(yaw) * v[:, 0] - torch.sin(yaw) * v[:, 1]
        velocity[:, 1] = torch.sin(yaw) * v[:, 0] + torch.cos(yaw) * v[:, 1]
        velocity[:, 5] = v[:, 2]
        self.robot.write_root_velocity_to_sim(velocity, env_ids=ids)

    def _update_command(self) -> None:
        self.frame = self.ahead(1)
        self._command[:] = self.base[self.ahead(1)]

    def _update_metrics(self) -> None:
        error = self.robot.data.joint_pos[:, self.joint_ids] - self.target(1)
        self.metrics["joint_error"] = error.abs().mean(dim=-1)


@configclass
class ReferenceMotionCommandCfg(CommandTermCfg):
    class_type: type = ReferenceMotionCommand
    asset_name: str = MISSING
    reference: str = MISSING
    """A ``retarget_reference.py`` archive."""
    offsets: tuple[int, ...] = (1, 2, 4, 8)
    """Steps ahead at which the policy sees the reference."""
    shortest: int = 25
    """Steps an episode is at least given before its clip runs out."""
    in_order: bool = False
    """Environment ``i`` follows clip ``i`` (modulo their number) instead of a random one."""
    from_start: bool = False
    """Start every episode at its clip's first row instead of a random one."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)


class ReferenceJointPositionAction(JointPositionAction):
    """Joint position targets about the reference: ``q_ref(t + 1) + scale · a``.

    Isaac Lab's ``JointPositionAction`` with its offset, the nominal crouch, replaced each
    step by the reference's next row, the one the step's reward scores. The zero action
    plays the reference back; the policy learns the feedback around it (residual policy
    learning, Silver et al. 2018; on a biped's reference motion, Xie et al. 2018). About
    the crouch instead, the retargeted references sit so far from the zero action that
    ``exp(-2 Σ Δq²)`` is ~0 there, and the pose reward gave no gradient.
    """

    cfg: ReferenceJointPositionActionCfg

    def __init__(self, cfg: ReferenceJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._columns: list[int] | None = None

    def process_actions(self, actions: torch.Tensor) -> None:
        term: ReferenceMotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        if self._columns is None:
            self._columns = [term.names.index(n) for n in self._joint_names]
        # Actions are processed before the physics and the command's advance: target(1) is
        # the row the step ends on.
        self._offset = term.target(1)[:, self._columns]
        super().process_actions(actions)


@configclass
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ReferenceJointPositionAction
    command_name: str = MISSING
    """The :class:`ReferenceMotionCommand` whose next row is the targets' centre."""


def reference_joints(env, command_name: str) -> torch.Tensor:
    """``(N, J × offsets)``: the reference's joint angles ahead, about the default posture."""
    term: ReferenceMotionCommand = env.command_manager.get_term(command_name)
    return torch.cat([term.target(k) - term.default for k in term.cfg.offsets], dim=-1)


def pose_tracking(env, command_name: str, scale: float) -> torch.Tensor:
    """``exp(-scale Σ (q - q_ref)²)``, DeepMimic's pose reward.

    Rewards are computed after the physics step and before the command advances, so the
    state reached is the one the reference holds a step ahead.
    """
    term: ReferenceMotionCommand = env.command_manager.get_term(command_name)
    error = term.robot.data.joint_pos[:, term.joint_ids] - term.target(1)
    return torch.exp(-scale * error.square().sum(dim=-1))


def rate_tracking(env, command_name: str, scale: float) -> torch.Tensor:
    """``exp(-scale Σ (q̇ - q̇_ref)²)``, DeepMimic's velocity reward."""
    term: ReferenceMotionCommand = env.command_manager.get_term(command_name)
    error = term.robot.data.joint_vel[:, term.joint_ids] - term.target_rate(1)
    return torch.exp(-scale * error.square().sum(dim=-1))


def reference_ended(env, command_name: str) -> torch.Tensor:
    term: ReferenceMotionCommand = env.command_manager.get_term(command_name)
    return term.ended()
