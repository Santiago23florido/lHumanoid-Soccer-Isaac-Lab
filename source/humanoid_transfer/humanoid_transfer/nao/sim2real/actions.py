"""Joint targets as a NAO's motor boards receive them: late, and in encoder coordinates.

:class:`RobotJointPositionAction` is Isaac Lab's joint position action with two
things a real NAO adds between the policy and the joints:

* **latency**: every target reaches the drives a number of physics steps after
  the policy sent it -- drawn per episode from ``delay_s`` -- covering the
  sensor read, the policy, the robot's cycle and the motor's own delay;
* **calibration**: a board servoes its encoder's reading to the target, so the
  joint settles at the target minus the encoder's offset
  (:attr:`~humanoid_transfer.nao.sim2real.state.RobotErrors.encoder_offset`).

With ``delay_s = (0, 0)`` and no offsets it is the plain action.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .state import robot_errors


class RobotJointPositionAction(JointPositionAction):
    cfg: RobotJointPositionActionCfg

    def __init__(self, cfg: RobotJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        physics_dt = env.physics_dt
        low, high = cfg.delay_s
        self._delay_range = (int(round(low / physics_dt)), int(round(high / physics_dt)))
        self._length = self._delay_range[1] + 1
        self._history = self._offset_targets().unsqueeze(0).repeat(self._length, 1, 1)
        self._head = 0
        self.delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Each environment's latency, physics steps."""
        self._rows = torch.arange(self.num_envs, device=self.device)

    def _offset_targets(self) -> torch.Tensor:
        """The targets of a zero action: the default posture."""
        if isinstance(self._offset, torch.Tensor):
            return self._offset.clone().expand(self.num_envs, -1).clone()
        return torch.full((self.num_envs, self.action_dim), float(self._offset), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = self._rows if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        low, high = self._delay_range
        self.delay[ids] = torch.randint(low, high + 1, (len(ids),), device=self.device)
        self._history[:, ids] = self._offset_targets()[ids]

    def apply_actions(self) -> None:
        self._head = (self._head + 1) % self._length
        self._history[self._head] = self.processed_actions
        index = (self._head - self.delay) % self._length
        targets = self._history[index, self._rows]
        offset = robot_errors(self._env).encoder_offset[:, self._joint_ids]
        self._asset.set_joint_position_target(targets - offset, joint_ids=self._joint_ids)


@configclass
class RobotJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RobotJointPositionAction

    delay_s: tuple[float, float] = (0.0, 0.0)
    """Range of the latency from the policy's output to the drives, seconds, drawn per
    episode and rounded to physics steps."""


__all__ = ["RobotJointPositionAction", "RobotJointPositionActionCfg"]
