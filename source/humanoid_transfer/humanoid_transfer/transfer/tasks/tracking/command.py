"""A velocity command that also carries the task's plan of ``z*``.

The velocity command is sampled exactly as Isaac Lab's
:class:`~isaaclab.envs.mdp.UniformVelocityCommand` samples it, because the
task's generator was trained on the commands a teacher saw. What this term
adds is the contract:

* **History.** Every control step the body's ``ẑ*`` is appended to a buffer
  that spans the generator's history grid in body time
  (:class:`~humanoid_transfer.transfer.latent.memory.TimeGrid`), so a body
  with a different control step in units of ``T`` reads the same physical
  window. While training, ``ẑ*`` is the true state plus noise the size of the
  encoder's measured error -- the input the generator was trained on, and
  one the tracker cannot steer through an encoder's quirks. With
  ``use_encoder`` it is the body's encoder: the path that is deployed.
* **Plans.** A plan is drawn from the generator, interpolated onto the
  body's own control instants, held for the commitment and redrawn from the
  history that happened -- also when the command is resampled, and from the
  first trustworthy step after a reset. Holding and redrawing is ancestral
  sampling of the trajectory (see :mod:`~...latent.contract`).
* **The first step after a reset** is read before the simulator has applied
  the reset pose, as the captures found; it is neither stored nor rewarded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import mdp
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

from ....common.similarity import UnitSystem
from ....common.skill_state import SKILL_LAYOUT
from ...latent.contract import load_contract, plan_integrals
from ...latent.memory import TimeGrid
from ...latent.skill_torch import BodyConstants, skill_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_TOLERANCE = 1e-6
_PLAN_LIMIT = 50.0
"""Normalised units: fifty standard deviations of the teacher's ``z*``."""


def plan_interpolation(grid_step: float, horizon: int, dt_star: float) -> torch.Tensor:
    """``(L, horizon + 1)``: a plan (now, then ``horizon`` grid samples) at the body's steps.

    Row ``j`` is the instant ``(j + 1) * dt_star`` after the plan was drawn, as
    long as it lies within the plan. When the body's step equals the grid's,
    this selects the plan's samples one by one.
    """
    length = math.floor(horizon * grid_step / dt_star + _TOLERANCE)
    matrix = torch.zeros(length, horizon + 1)
    for j in range(length):
        position = (j + 1) * dt_star / grid_step
        low = min(math.floor(position + _TOLERANCE), horizon)
        weight = position - low
        if weight < _TOLERANCE or low == horizon:
            matrix[j, low] = 1.0
        else:
            matrix[j, low] = 1.0 - weight
            matrix[j, low + 1] = weight
    return matrix


class ReferenceCommand(UniformVelocityCommand):
    """Velocity command plus the plan of ``z*`` the body is asked to reach."""

    cfg: ReferenceCommandCfg

    def __init__(self, cfg: ReferenceCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        contract = load_contract(cfg.contract, self.device)
        self.generator = contract.generator
        if cfg.flow_solver is not None:
            self.generator.flow_solver = cfg.flow_solver
        if cfg.flow_steps is not None:
            self.generator.flow_steps = cfg.flow_steps
        self.encoder = contract.encoder if cfg.use_encoder else None
        if cfg.use_encoder and self.encoder is None:
            raise ValueError(f"{cfg.contract} carries no encoder")
        self.sigma = contract.sigma.to(self.device) * cfg.noise_scale

        self.sensor = env.scene[cfg.sensor_name]
        foot_ids, foot_names = self.robot.find_bodies(cfg.foot_bodies, preserve_order=True)
        sensor_ids, sensor_names = self.sensor.find_bodies(cfg.foot_bodies, preserve_order=True)
        if list(foot_names) != list(cfg.foot_bodies) or list(sensor_names) != list(foot_names):
            raise RuntimeError(f"foot bodies disagree: {foot_names} / {sensor_names}")
        base_ids, _ = self.robot.find_bodies(cfg.base_body)
        self.sensor_ids = sensor_ids

        view = self.robot.root_physx_view
        masses = view.get_masses()[0].to(self.device, torch.float32)
        inertia = view.get_inertias()[0].to(self.device, torch.float32).reshape(-1, 3, 3)
        com_offset = self.robot.data.body_com_pos_b[0].to(torch.float32)
        sole_points = torch.tensor(cfg.sole_points, device=self.device, dtype=torch.float32)
        units = UnitSystem(cfg.leg_length, float(masses.sum()))
        self.units = units
        self.body = BodyConstants(
            masses=masses,
            inertia_local=inertia,
            base_index=int(base_ids[0]),
            foot_index=(int(foot_ids[0]), int(foot_ids[1])),
            sole_from_com=sole_points - com_offset[foot_ids],
            units=units,
        ).to(self.device)
        self.scales = self.body.scales()

        self.dt_star = env.step_dt / units.time
        self.command_scale = torch.tensor(
            [1.0 / units.velocity, 1.0 / units.velocity, units.time], device=self.device
        )
        generator = self.generator
        grid_step = float(generator.grid_step)
        grid = TimeGrid(step=grid_step, samples=generator.history)
        self.interpolation = torch.tensor(
            grid.interpolation(self.dt_star), dtype=torch.float32, device=self.device
        )
        self.plan_matrix = plan_interpolation(grid_step, generator.horizon, self.dt_star).to(
            self.device
        )
        length = self.plan_matrix.shape[0]
        self.commit = max(1, round(generator.commit * grid_step / self.dt_star))
        self.view = math.floor(generator.view * grid_step / self.dt_star + _TOLERANCE)
        if self.commit - 1 + self.view > length:
            raise ValueError("the plan does not cover the commitment and the view on this body")
        nearest = [
            min(max(round((j + 1) * self.dt_star / grid_step) - 1, 0), generator.horizon - 1)
            for j in range(length)
        ]
        self.tolerance = generator.tolerance[nearest].to(self.device)
        self.integral_tolerance = generator.integral_tolerance[nearest].to(self.device)

        n, size = self.num_envs, generator.skill
        steps = self.interpolation.shape[1]
        self.history = torch.zeros(n, steps, size, device=self.device)
        self.plan = torch.zeros(n, length, size, device=self.device)
        self.age = torch.zeros(n, dtype=torch.long, device=self.device)
        self.estimate = torch.zeros(n, size, device=self.device)
        self.just_reset = torch.ones(n, dtype=torch.bool, device=self.device)
        self.needs_refill = torch.ones(n, dtype=torch.bool, device=self.device)
        self.replan = torch.ones(n, dtype=torch.bool, device=self.device)
        fields = SKILL_LAYOUT.fields
        self.field_mean = torch.zeros(size, len(fields), device=self.device)
        for f, field in enumerate(fields):
            span = SKILL_LAYOUT.slice_of(field.name)
            self.field_mean[span, f] = 1.0 / field.size
        self.field_error = torch.zeros(n, len(fields) + 1, device=self.device)
        self.last_error = torch.zeros(n, device=self.device)
        self.last_error_state = torch.zeros(n, device=self.device)
        self.integrals = torch.zeros(n, length, 3, device=self.device)
        self.start_com = torch.zeros(n, 2, device=self.device)
        self.start_yaw = torch.zeros(n, device=self.device)
        zeros = (torch.zeros(n, 2, device=self.device), torch.zeros(n, device=self.device))
        self._reward_frame = zeros
        self._frame_now = zeros
        self.last_error_other = torch.zeros(n, device=self.device)
        self.metrics["contract_error"] = torch.zeros(n, device=self.device)
        self._rows = torch.arange(n, device=self.device)
        self._reward_stamp = -1
        self._reward_true = torch.zeros(n, size, device=self.device)
        self._true_now = torch.zeros(n, size, device=self.device)

    def __str__(self) -> str:
        return (
            f"ReferenceCommand: contract {self.cfg.contract}, dt* {self.dt_star:.4f}, "
            f"commit {self.commit}, view {self.view}, encoder {self.encoder is not None}\n"
            + super().__str__()
        )

    # -- state of the body -------------------------------------------------

    def true_skill(self) -> torch.Tensor:
        """``(N, 27)`` the body's true ``z*``, normalised in the contract's units."""
        data = self.robot.data
        origins = self._env.scene.env_origins[:, None, :]
        z = skill_state(
            self.body,
            data.body_link_quat_w,
            data.body_com_pos_w - origins,
            data.body_com_lin_vel_w,
            data.body_com_ang_vel_w,
            self.sensor.data.net_forces_w[:, self.sensor_ids],
        )
        return self.generator.skill_norm(z)

    def frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The whole-body CoM ``(N, 2)`` in the world plane, metres, and the base yaw ``(N,)``."""
        data = self.robot.data
        masses = self.body.masses
        com = torch.einsum("b,nbi->ni", masses, data.body_com_pos_w[..., :2]) / masses.sum()
        w, x, y, z = data.body_link_quat_w[:, self.body.base_index].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return com, yaw

    def _estimate(self) -> torch.Tensor:
        if self.encoder is None:
            return self._true_now + self.sigma * torch.randn_like(self._true_now)
        env = self._env
        observation = torch.cat(
            [
                mdp.base_lin_vel(env),
                mdp.base_ang_vel(env),
                mdp.projected_gravity(env),
                mdp.joint_pos_rel(env),
                mdp.joint_vel_rel(env),
                mdp.last_action(env),
            ],
            dim=-1,
        )
        with torch.no_grad():
            return self.encoder(observation)

    # -- the plan ------------------------------------------------------------

    def due(self) -> torch.Tensor:
        """``(N, 27)`` the planned ``z*`` for the step just simulated."""
        return self.plan[self._rows, self.age]

    def upcoming(self) -> torch.Tensor:
        """``(N, view, 27)`` the held plan from the next step on, as the tracker sees it."""
        index = self.age[:, None] + torch.arange(self.view, device=self.device)
        return self.plan[self._rows[:, None], index]

    def tracking_error(self) -> torch.Tensor:
        """``(N, fields)`` mean squared distance to the due plan per field, in tolerances.

        Each field of ``z*`` (height, CoM velocity, gravity, ..., angular momentum)
        is one physical quantity; averaging within it keeps a field with six
        components from outweighing one with a single component.
        """
        stamp = self._env.common_step_counter
        if stamp != self._reward_stamp:
            self._reward_true = self.true_skill()
            self._reward_frame = self.frame()
            self._reward_stamp = stamp
        due, tolerance = self.due(), self.tolerance[self.age]
        error = (self._reward_true - due) / tolerance
        # The same state against the plan another environment holds: if a tracker
        # ignored its plan, the two errors would be alike.
        other = (self._reward_true - due.roll(1, 0)) / tolerance.roll(1, 0)
        self.last_error_other = (other.pow(2) @ self.field_mean).mean(dim=-1)

        com, yaw = self._reward_frame
        c, s = torch.cos(self.start_yaw), torch.sin(self.start_yaw)
        moved = (com - self.start_com) / self.units.length
        turned = yaw - self.start_yaw
        reached = torch.stack(
            [
                c * moved[:, 0] + s * moved[:, 1],
                -s * moved[:, 0] + c * moved[:, 1],
                torch.atan2(torch.sin(turned), torch.cos(turned)),
            ],
            dim=-1,
        )
        planned = self.integrals[self._rows, self.age]
        progress = (reached - planned) / self.integral_tolerance[self.age]
        progress = progress.pow(2).mean(dim=-1)
        return torch.cat([error.pow(2) @ self.field_mean, progress[:, None]], dim=-1)

    def tracking_reward(self, kernel: float) -> torch.Tensor:
        """``exp(-e / (2 kernel^2))``, ``e`` the mean of the fields' errors; 0 without a plan.

        One likelihood for the whole state: every field has to be reached at once.
        Scoring fields apart and averaging the scores let a body give up a field
        cheaply -- measured, it gave up walking: standing still reached seven
        fields and paid only in CoM velocity.
        """
        error = self.tracking_error()
        valid = ~self.needs_refill
        self.field_error = torch.where(valid[:, None], error, self.field_error)
        error = torch.nan_to_num(error, nan=1e6, posinf=1e6)
        joint = error.mean(dim=-1)
        self.last_error = torch.where(valid, joint, self.last_error)
        state = error[:, :-1].mean(dim=-1)
        self.last_error_state = torch.where(valid, state, self.last_error_state)
        return torch.exp(-0.5 * joint / kernel**2) * valid

    def _draw(self, ids: torch.Tensor) -> None:
        window = torch.einsum("hs,nsd->nhd", self.interpolation, self.history[ids])
        command = self.vel_command_b[ids] * self.command_scale
        noise = torch.randn(len(ids), self.generator.noise_size, device=self.device)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, self.cfg.half_precision):
            plan = self.generator.sample(window, command, noise).float()
        # A body still learning to stand writes histories no teacher ever did; the
        # generator's answer to them is meaningless but must stay finite.
        plan = torch.nan_to_num(plan, nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)
        full = torch.cat([window[:, -1:], plan], dim=1)
        self.plan[ids] = torch.einsum("lk,nkd->nld", self.plan_matrix, full)
        step = float(self.generator.grid_step)
        integrals = plan_integrals(self.generator.skill_norm.inverse(plan), step)
        integrals = torch.cat([torch.zeros_like(integrals[:, :1]), integrals], dim=1)
        self.integrals[ids] = torch.einsum("lk,nkd->nld", self.plan_matrix, integrals)
        com, yaw = self._frame_now
        self.start_com[ids] = com[ids]
        self.start_yaw[ids] = yaw[ids]
        self.age[ids] = 0

    # -- command term hooks ------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.just_reset[ids] = True
        self.needs_refill[ids] = True
        self.age[ids] = 0
        return extras

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        self.replan[env_ids] = True

    def _update_metrics(self) -> None:
        super()._update_metrics()
        metric = self.metrics["contract_error"]
        self.metrics["contract_error"] = 0.98 * metric + 0.02 * self.last_error

    def _update_command(self) -> None:
        super()._update_command()
        self._true_now = self.true_skill()
        self._frame_now = self.frame()
        estimate = self._estimate()
        ready = ~self.just_reset
        refill = ready & self.needs_refill
        carried = ready & ~self.needs_refill

        shifted = torch.cat([self.history[:, 1:], estimate[:, None]], dim=1)
        filled = estimate[:, None].expand_as(self.history)
        self.history = torch.where(
            refill[:, None, None],
            filled,
            torch.where(carried[:, None, None], shifted, self.history),
        )
        self.age = torch.where(carried, self.age + 1, self.age)
        draw = refill | (carried & (self.replan | (self.age >= self.commit)))
        ids = draw.nonzero().squeeze(-1)
        if len(ids) > 0:
            self._draw(ids)
        self.replan = self.replan & ~draw
        self.needs_refill = self.needs_refill & ~ready
        self.just_reset[:] = False
        self.estimate = estimate

    # -- what the tracker and its critic read --------------------------------

    def observation(self) -> torch.Tensor:
        """``(N, view * 27 + 27)``: the held plan relative to ``ẑ*``, then ``ẑ*``."""
        relative = self.upcoming() - self.estimate[:, None]
        relative = torch.where(self.needs_refill[:, None, None], 0.0, relative)
        view = torch.cat([relative.flatten(1), self.estimate], dim=-1)
        return torch.nan_to_num(view, nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)

    def privileged(self) -> torch.Tensor:
        """``(N, view * 27 + 28)``: the plan relative to the true ``z*``, the true ``z*``, age."""
        true = self._true_now
        relative = self.upcoming() - true[:, None]
        relative = torch.where(self.needs_refill[:, None, None], 0.0, relative)
        age = (self.age.float() / self.commit)[:, None]
        view = torch.cat([relative.flatten(1), true, age], dim=-1)
        return torch.nan_to_num(view, nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)


@configclass
class ReferenceCommandCfg(UniformVelocityCommandCfg):
    """Configuration of :class:`ReferenceCommand`."""

    class_type: type = ReferenceCommand

    contract: str = MISSING
    """Path to the task's ``generator.pt`` (:func:`~...latent.contract.save_contract`)."""

    base_body: str = MISSING
    """The body carrying both hips: its yaw is the heading frame."""

    foot_bodies: list[str] = MISSING
    """Left and right foot bodies, in that order."""

    sole_points: list[list[float]] = MISSING
    """Each foot's sole point in its link frame, metres."""

    leg_length: float = MISSING
    """The body's unit of length, metres."""

    sensor_name: str = "contact_forces"
    use_encoder: bool = False
    """Read ``ẑ*`` from the contract's encoder (deployed path) instead of truth plus noise."""

    noise_scale: float = 1.0
    """Multiplies the encoder error used as noise on the true state while training."""

    flow_solver: str | None = None
    """Overrides the generator's integrator (``euler`` or ``midpoint``); None keeps it."""

    flow_steps: int | None = None
    """Overrides the generator's integration steps; None keeps them."""

    half_precision: bool = False
    """Draw plans under float16 autocast (the state, history and plan stay float32)."""


__all__ = ["ReferenceCommand", "ReferenceCommandCfg", "plan_interpolation"]
