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
from ...latent.contract import load_contract
from ...latent.memory import TimeGrid
from ...latent.skill_torch import BodyConstants, skill_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_TOLERANCE = 1e-6


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

        n, size = self.num_envs, generator.skill
        steps = self.interpolation.shape[1]
        self.history = torch.zeros(n, steps, size, device=self.device)
        self.plan = torch.zeros(n, length, size, device=self.device)
        self.age = torch.zeros(n, dtype=torch.long, device=self.device)
        self.estimate = torch.zeros(n, size, device=self.device)
        self.just_reset = torch.ones(n, dtype=torch.bool, device=self.device)
        self.needs_refill = torch.ones(n, dtype=torch.bool, device=self.device)
        self.replan = torch.ones(n, dtype=torch.bool, device=self.device)
        self.last_error = torch.zeros(n, device=self.device)
        self.metrics["contract_error"] = torch.zeros(n, device=self.device)
        self._rows = torch.arange(n, device=self.device)

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

    def _estimate(self) -> torch.Tensor:
        if self.encoder is None:
            true = self.true_skill()
            return true + self.sigma * torch.randn_like(true)
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
        """``(N,)`` mean squared distance to the due plan, in units of its tolerance."""
        error = (self.true_skill() - self.due()) / self.tolerance[self.age]
        return error.pow(2).mean(dim=-1)

    def tracking_reward(self, kernel: float) -> torch.Tensor:
        error = self.tracking_error()
        valid = ~self.needs_refill
        self.last_error = torch.where(valid, error, self.last_error)
        return torch.exp(-0.5 * error / kernel**2) * valid

    def _draw(self, ids: torch.Tensor) -> None:
        window = torch.einsum("hs,nsd->nhd", self.interpolation, self.history[ids])
        command = self.vel_command_b[ids] * self.command_scale
        noise = torch.randn(len(ids), self.generator.noise_size, device=self.device)
        with torch.no_grad():
            plan = self.generator.sample(window, command, noise)
        full = torch.cat([window[:, -1:], plan], dim=1)
        self.plan[ids] = torch.einsum("lk,nkd->nld", self.plan_matrix, full)
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
        return torch.cat([relative.flatten(1), self.estimate], dim=-1)

    def privileged(self) -> torch.Tensor:
        """``(N, view * 27 + 28)``: the plan relative to the true ``z*``, the true ``z*``, age."""
        true = self.true_skill()
        relative = self.upcoming() - true[:, None]
        relative = torch.where(self.needs_refill[:, None, None], 0.0, relative)
        age = (self.age.float() / self.commit)[:, None]
        return torch.cat([relative.flatten(1), true, age], dim=-1)


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


__all__ = ["ReferenceCommand", "ReferenceCommandCfg", "plan_interpolation"]
