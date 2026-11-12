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
* **Plans no task wrote.** With ``procedural_fraction``, that share of
  episodes follows the point-mass biped of
  :mod:`~humanoid_transfer.transfer.latent.procedural` instead of the task's
  generator: a plan from the whole family of feasible gaits, anchored on the
  body at the episode's first trustworthy step, with exact integrals. The
  tracker is not told which source a plan came from. The fields the biped
  cannot derive are scored at its measured model error where that exceeds
  the encoder's (``transfer/results/procedural_calibration.json``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.envs import mdp
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from ....common.similarity import UnitSystem
from ....common.skill_state import SKILL_LAYOUT
from ...latent.contract import load_contract, plan_integrals
from ...latent.memory import TimeGrid
from ...latent.plan_contact import PLAN_CONTACTS, apply_plan_contact
from ...latent.plan_trust import PlanTrust, smooth_noise, sway
from ...latent.procedural import GaitRanges, PointMassBiped
from ...latent.resampling import RESAMPLE_SPACES, plan_interpolation, resample_rows
from ...latent.skill_torch import BodyConstants, proprioceptive_skill_state, skill_state
from ...latent.units import change_of_units, tracker_view

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_TOLERANCE = 1e-6
_PLAN_LIMIT = 50.0
LOAD_EPSILON = 0.01
"""The generator's load warp, ``log(load + LOAD_EPSILON)``."""
_VX = SKILL_LAYOUT.slice_of("com_velocity").start
_YAW_RATE = SKILL_LAYOUT.slice_of("base_angular_velocity").start + 2
_LOAD = SKILL_LAYOUT.slice_of("foot_load")
_TILT = SKILL_LAYOUT.slice_of("gravity")
"""Normalised units: fifty standard deviations of the teacher's ``z*``."""


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
        self.observed_units = None
        """``(scale, shift)`` from the contract's units to the tracker's (``observe_units``);
        ``None`` observes in the contract's."""
        if cfg.observe_units is not None:
            own = load_contract(cfg.observe_units, self.device).generator.skill_norm
            self.observed_units = change_of_units(self.generator.skill_norm, own)
        self.sigma = contract.sigma.to(self.device) * cfg.noise_scale
        self.precision = contract.sigma.to(self.device)
        """The body estimator's measured error per dimension (normalised): the unit of
        ``estimate_noise``."""

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
        self.sole_corners = None
        if cfg.sole_corners is not None:
            corners = torch.tensor(cfg.sole_corners, device=self.device, dtype=torch.float32)
            self.sole_corners = corners - com_offset[foot_ids][:, None]
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
        if cfg.resample_space not in RESAMPLE_SPACES:
            raise ValueError(f"unknown resample_space {cfg.resample_space!r}")
        self.resample_space = cfg.resample_space
        if cfg.plan_contact not in PLAN_CONTACTS:
            raise ValueError(f"unknown plan_contact {cfg.plan_contact!r}")
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
        self.tasks = None
        """With ``cfg.tasks``: per task, its generator, plan matrix, length, commitment,
        ranges in this body's units, and starts. Task 0 is the contract's."""
        if cfg.tasks:
            if cfg.tolerance == "spread" or cfg.plan_trust or cfg.procedural_plan_noise:
                raise ValueError("several tasks need tolerance 'precision' and no plan trust")
            self.tasks = self._load_tasks(cfg, grid_step, length)
            length = max(task["length"] for task in self.tasks)
        nearest = [
            min(max(round((j + 1) * self.dt_star / grid_step) - 1, 0), generator.horizon - 1)
            for j in range(length)
        ]
        if cfg.tolerance == "spread":
            self.tolerance = generator.tolerance[nearest].to(self.device)
            self.integral_tolerance = generator.integral_tolerance[nearest].to(self.device)
        else:
            # A drawn plan is a commitment: a body is judged against it at the
            # precision the state is known with -- the encoder's error -- not at
            # how uncertain the future was before the plan was drawn.
            sigma = contract.sigma.to(self.device)
            self.tolerance = sigma.expand(length, -1).clone()
            physical = sigma * generator.skill_norm.std.to(self.device)
            rates = torch.stack([physical[_VX], physical[_VX + 1], physical[_YAW_RATE]])
            # Integrating a velocity known to sigma: the error grows like sqrt(time),
            # counted from when the reference last started from the body (below).
            self.integral_rate = rates * self.dt_star
            self.tolerance_procedural = self._procedural_tolerance(sigma, length)

        n, size = self.num_envs, generator.skill
        steps = self.interpolation.shape[1]
        self.history = torch.zeros(n, steps, size, device=self.device)
        self.physical = torch.zeros(n, steps, size, device=self.device)
        self.far_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.resyncs = torch.zeros(n, device=self.device)
        self.plan = torch.zeros(n, length, size, device=self.device)
        self._load = SKILL_LAYOUT.slice_of("foot_load")
        self.load_log_tolerance = None
        if cfg.load_space == "log":
            report = json.loads(Path(cfg.load_precision).read_text(encoding="utf-8"))
            self.load_log_tolerance = float(report["log_rms"])
        elif cfg.load_space != "linear":
            raise ValueError(f"unknown load_space {cfg.load_space!r}")
        self.plan_second = torch.zeros(n, length, size, device=self.device)
        """With ``plan_trust``: a second sample from the same history, row for row (the plan
        itself where the biped drew it), so a prior can average the two instead of shrinking
        the plan towards the body's state."""
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
        self.anchored = torch.zeros(n, dtype=torch.long, device=self.device)
        zeros = (torch.zeros(n, 2, device=self.device), torch.zeros(n, device=self.device))
        self._reward_frame = zeros
        self._frame_now = zeros
        self.last_error_other = torch.zeros(n, device=self.device)
        self.last_lag = torch.zeros(n, device=self.device)
        self.metrics["contract_error"] = torch.zeros(n, device=self.device)
        self.metrics["lag"] = torch.zeros(n, device=self.device)
        self._rows = torch.arange(n, device=self.device)
        self.has_plan = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.task = torch.zeros(n, dtype=torch.long, device=self.device)
        """The task each env's episode follows (0 without ``cfg.tasks``)."""
        self.commit_env = torch.full((n,), self.commit, dtype=torch.long, device=self.device)
        """Each env's commitment in body steps: its task's (the contract's for the biped)."""
        self.starts = self._load_starts(cfg.start_states) if cfg.start_states else None
        self.windows = self._load_windows(cfg.start_windows) if cfg.start_windows else None
        self.procedural = torch.zeros(n, dtype=torch.bool, device=self.device)
        """Envs whose episode follows the point-mass biped instead of the task's generator."""
        self.biped = None
        if cfg.procedural_fraction > 0.0 or cfg.procedural_probes:
            ranges = GaitRanges(load_shape=tuple(float(v) for v in cfg.procedural_load_shape))
            # Probes that name a load profile need the shaped biped, whatever the ranges.
            named = any("load_shape" in p for p in (cfg.procedural_probes or []))
            self.biped = PointMassBiped(
                n,
                ranges,
                body_step=self.dt_star,
                substep=grid_step / 4,
                horizon=length * self.dt_star,
                device=self.device,
                seed=cfg.procedural_seed,
                shaped=ranges.shaped or named,
            )
            offsets = torch.arange(1, length + 1, dtype=torch.float64) * self.dt_star
            self.biped.set_offsets(offsets)
            self._row_times = offsets.to(self.device, torch.float32)
            if cfg.procedural_probes:
                self._set_probes(cfg.procedural_probes)
        self.metrics["procedural"] = torch.zeros(n, device=self.device)
        self.metrics["error_procedural"] = torch.zeros(n, device=self.device)
        self.metrics["error_task"] = torch.zeros(n, device=self.device)
        self._reward_stamp = -1
        self._reward_true = torch.zeros(n, size, device=self.device)
        self._true_now = torch.zeros(n, size, device=self.device)
        # How far each source's plans can be trusted (plan_trust.py): row 0 the task's
        # generator, row 1 the biped.
        self.trust = None
        if cfg.plan_trust or cfg.procedural_plan_noise:
            self.trust = PlanTrust(2, length, size, cfg.trust_rate, self.device)
        self.sway_amplitude = torch.zeros(n, 2, device=self.device)
        """Each procedural episode's trunk sway, pitch and roll, radians."""

    def __str__(self) -> str:
        return (
            f"ReferenceCommand: contract {self.cfg.contract}, dt* {self.dt_star:.4f}, "
            f"commit {self.commit}, view {self.view}, encoder {self.encoder is not None}, "
            f"resample {self.resample_space}, plan contact {self.cfg.plan_contact}, "
            f"observed in {self.cfg.observe_units or 'the contract'}"
            f" ({'mapped' if self.observed_units is not None else 'identity'})\n"
            + super().__str__()
        )

    def _load_tasks(self, cfg: ReferenceCommandCfg, grid_step: float, length: int) -> list[dict]:
        """Task 0 is the contract's; each further one brings its own generator and ranges."""
        primary = self.generator
        tasks = [
            {
                "name": "contract",
                "generator": primary,
                "plan_matrix": self.plan_matrix,
                "renormalise": False,
                "length": length,
                "commit": self.commit,
                "ranges": None,
                "starts": None,
                "windows": None,
            }
        ]
        for spec in cfg.tasks:
            generator = load_contract(spec["contract"], self.device).generator
            if cfg.flow_solver is not None:
                generator.flow_solver = cfg.flow_solver
            if cfg.flow_steps is not None:
                generator.flow_steps = cfg.flow_steps
            same = (
                abs(float(generator.grid_step) - grid_step) < _TOLERANCE
                and generator.history == primary.history
                and generator.view == primary.view
                and generator.skill == primary.skill
                and generator.warp_loads == primary.warp_loads
            )
            if not same:
                raise ValueError(f"{spec['contract']}: not on the contract's grid and units")
            # A generator normalises z* with its own statistics. The G1's generators share
            # one set; the NAO's walk contract has its own, so its history is mapped into each
            # task generator's units and the plan back (an exact affine map, via z*).
            renormalise = not (
                torch.allclose(generator.skill_norm.mean, primary.skill_norm.mean)
                and torch.allclose(generator.skill_norm.std, primary.skill_norm.std)
            )
            matrix = plan_interpolation(grid_step, generator.horizon, self.dt_star).to(self.device)
            commit = max(1, round(generator.commit * grid_step / self.dt_star))
            if commit - 1 + self.view > matrix.shape[0]:
                raise ValueError(f"{spec['contract']}: the plan does not cover commit and view")
            ranges = spec["ranges"]
            scale = self.command_scale
            low = torch.tensor(
                [ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_z"][0]],
                device=self.device,
            )
            high = torch.tensor(
                [ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]],
                device=self.device,
            )
            tasks.append(
                {
                    "name": spec["name"],
                    "generator": generator,
                    "plan_matrix": matrix,
                    "renormalise": renormalise,
                    "length": matrix.shape[0],
                    "commit": commit,
                    # Dimensionless to this body's units: the inverse of what the
                    # generator reads (command_scale).
                    "ranges": (low / scale, high / scale),
                    "heading": tuple(float(v) for v in ranges["heading"]),
                    "starts": self._load_starts(spec["starts"]) if spec.get("starts") else None,
                    "windows": (
                        self._load_windows(spec["windows"]) if spec.get("windows") else None
                    ),
                }
            )
        weights = cfg.task_weights or [1.0] * len(tasks)
        if len(weights) != len(tasks) or min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError(f"task_weights {weights} do not fit {len(tasks)} tasks")
        self.task_probability = torch.tensor(weights, device=self.device) / sum(weights)
        return tasks

    def _task_commands(self, ids: torch.Tensor) -> None:
        """Commands of the envs in ``ids`` whose task is not the contract's, from its ranges."""
        for t, task in enumerate(self.tasks[1:], start=1):
            chosen = ids[self.task[ids] == t]
            if len(chosen) == 0:
                continue
            low, high = task["ranges"]
            draw = torch.rand(len(chosen), 3, device=self.device)
            self.vel_command_b[chosen] = low + (high - low) * draw
            if self.cfg.heading_command:
                lo, hi = task["heading"]
                self.heading_target[chosen] = lo + (hi - lo) * torch.rand(
                    len(chosen), device=self.device
                )

    def _task_headings(self) -> None:
        """Heading control clipped to each task's own yaw-rate range, as its teacher's was."""
        heading = self.is_heading_env & (self.task > 0)
        for t, task in enumerate(self.tasks[1:], start=1):
            ids = (heading & (self.task == t)).nonzero().squeeze(-1)
            if len(ids) == 0:
                continue
            low, high = task["ranges"]
            error = torch.atan2(
                torch.sin(self.heading_target[ids] - self.robot.data.heading_w[ids]),
                torch.cos(self.heading_target[ids] - self.robot.data.heading_w[ids]),
            )
            self.vel_command_b[ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * error, min=low[2], max=high[2]
            )
        self.vel_command_b[self.is_standing_env] = 0.0

    def _draw_tasks(
        self, ids: torch.Tensor, window: torch.Tensor, command: torch.Tensor, space: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each env's plan from its own task's generator, padded to the longest plan.

        Rows past a task's own plan repeat its last row. They are never read: an env's
        commitment and view lie inside its own plan (checked at load).
        """
        length = self.plan.shape[1]
        rows = torch.zeros(len(ids), length, self.plan.shape[2], device=self.device)
        integrals = torch.zeros(len(ids), length, 3, device=self.device)
        for t, task in enumerate(self.tasks):
            chosen = (self.task[ids] == t).nonzero().squeeze(-1)
            if len(chosen) == 0:
                continue
            generator, matrix = task["generator"], task["plan_matrix"]
            history = window[chosen]
            if task["renormalise"]:
                history = generator.skill_norm(self.generator.skill_norm.inverse(history))
            noise = torch.randn(len(chosen), generator.noise_size, device=self.device)
            with torch.no_grad(), torch.autocast("cuda", torch.float16, self.cfg.half_precision):
                plan = generator.sample(history, command[chosen], noise).float()
            plan = torch.nan_to_num(plan, nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)
            full = torch.cat([history[:, -1:], plan], dim=1)
            own = resample_rows(generator, matrix, full, space)
            own = apply_plan_contact(own, generator.skill_norm, self.cfg.plan_contact)
            if task["renormalise"]:
                own = self.generator.skill_norm(generator.skill_norm.inverse(own))
            moved = plan_integrals(generator.skill_norm.inverse(plan), float(generator.grid_step))
            moved = torch.cat([torch.zeros_like(moved[:, :1]), moved], dim=1)
            moved = torch.einsum("lk,nkd->nld", matrix, moved)
            pad = length - own.shape[1]
            rows[chosen] = torch.cat([own, own[:, -1:].expand(-1, pad, -1)], dim=1)
            integrals[chosen] = torch.cat([moved, moved[:, -1:].expand(-1, pad, -1)], dim=1)
        return rows, integrals

    def _procedural_tolerance(self, sigma: torch.Tensor, length: int) -> torch.Tensor:
        """``(length, 27)``: the precision tolerance, widened to the biped's model error where
        that is the larger -- the fields a point mass cannot derive (tilt, base rate,
        angular momentum). A procedural plan is a likelihood under the model it was written
        with; scoring those fields at the encoder's precision would score the model."""
        tolerance = sigma.clone()
        if self.cfg.procedural_tolerance == "calibrated" and self.cfg.procedural_calibration:
            report = json.loads(Path(self.cfg.procedural_calibration).read_text(encoding="utf-8"))
            names = SKILL_LAYOUT.component_names()
            std = self.generator.skill_norm.std.to(self.device)
            for name, value in report["sigma_model"].items():
                i = names.index(name)
                model = value / float(std[i])
                if model > float(sigma[i]):
                    tolerance[i] = math.sqrt(float(sigma[i]) ** 2 + model**2)
        return tolerance.expand(length, -1).clone()

    def _set_probes(self, probes: list[dict]) -> None:
        """Every env follows one fixed gait of the family, env ``i`` probe ``i % len``."""
        n = self.num_envs
        keys = ("forward", "lateral", "turn", "period", "width", "height", "clearance", "lean")
        which = [i % len(probes) for i in range(n)]
        params = {
            key: torch.tensor([float(probes[w][key]) for w in which], device=self.device)
            for key in keys
        }
        params["duty"] = torch.tensor([list(probes[w]["duty"]) for w in which], device=self.device)
        params["first"] = torch.tensor([i % 2 for i in range(n)], device=self.device)
        params["duration"] = torch.full((n,), 1e9, device=self.device)
        params["offset"] = torch.tensor(
            [float(probes[w].get("offset", 0.5)) for w in which], device=self.device
        )
        if any("load_shape" in p for p in probes):
            params["load_shape"] = torch.tensor(
                [float(probes[w].get("load_shape", 1.0)) for w in which], device=self.device
            )
        self.biped.set_probes(torch.arange(n, device=self.device), params)
        self.probe_index = torch.tensor(which, device=self.device)

    def _load_starts(self, path: str) -> dict[str, torch.Tensor]:
        """States of the task's own demonstrations, to start episodes from (RSI)."""
        data = np.load(path)
        names = [str(n) for n in data["joint_names"]]
        if names != list(self.robot.data.joint_names):
            raise ValueError(f"{path}: joints {names} are not this robot's")
        done = data["done"]
        steps = self.interpolation.shape[1]
        # A start needs a full history window inside one episode, away from a fall.
        clean = ~np.lib.stride_tricks.sliding_window_view(done, steps + 1, axis=0).any(-1)
        if "terminated" in data.files:
            near_fall = np.lib.stride_tricks.sliding_window_view(
                data["terminated"], steps + 60, axis=0
            ).any(-1)
            clean = clean[: len(near_fall)] & ~near_fall
        t, n = np.nonzero(clean)
        t = t + steps
        keys = (
            "joint_pos",
            "joint_vel",
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel",
            "command",
            "skill_state",
        )
        starts = {
            k: torch.as_tensor(data[k], device=self.device, dtype=torch.float32) for k in keys
        }
        starts["index"] = torch.as_tensor(np.stack([t, n], -1), device=self.device)
        return starts

    def _load_windows(self, path: str) -> dict[str, torch.Tensor]:
        """Dimensionless windows of the task's demonstrations, any body's (see below)."""
        data = np.load(path)
        units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
        done = data["done"]
        steps = self.interpolation.shape[1]
        clean = ~np.lib.stride_tricks.sliding_window_view(done, steps + 1, axis=0).any(-1)
        if "terminated" in data.files:
            near_fall = np.lib.stride_tricks.sliding_window_view(
                data["terminated"], steps + 60, axis=0
            ).any(-1)
            clean = clean[: len(near_fall)] & ~near_fall
        t, n = np.nonzero(clean)
        command = data["command"] * np.array([1 / units.velocity, 1 / units.velocity, units.time])
        return {
            "skill_state": torch.as_tensor(data["skill_state"], device=self.device),
            "command": torch.as_tensor(command, device=self.device, dtype=torch.float32),
            "index": torch.as_tensor(np.stack([t + steps, n], -1), device=self.device),
        }

    def _start_from_contract(self, ids: torch.Tensor, windows: dict | None = None) -> None:
        """Start a body with no demonstrations of its own inside the task's motion.

        Reference state initialisation needs states of this body, which a second
        body does not have. What the contract does have is the task's motion in
        dimensionless units: the generator's history is seeded with a window of
        it, the command is that window's (in this body's units), and the base is
        given the window's CoM velocity in this body's units. The joints stay in
        the default pose, so a body that does not step falls -- the plan asks for
        exactly the steps that catch it.
        """
        windows = self.windows if windows is None else windows
        pick = windows["index"][
            torch.randint(len(windows["index"]), (len(ids),), device=self.device)
        ]
        t, n = pick[:, 0], pick[:, 1]
        steps = self.interpolation.shape[1]
        offsets = torch.arange(steps - 1, -1, -1, device=self.device)
        window = windows["skill_state"][t[:, None] - offsets, n[:, None]].float()
        normalised = self.generator.skill_norm(window)
        self.history[ids] = normalised + self.sigma * torch.randn_like(normalised)
        self.physical[ids] = self.history[ids]
        self.vel_command_b[ids] = windows["command"][t, n] / self.command_scale
        self.is_standing_env[ids] = windows["command"][t, n].norm(dim=-1) == 0
        heading = window[:, -1, _VX : _VX + 2] * self.units.velocity
        # The root pose the reset just wrote; body poses are only refreshed by a step.
        w, x, y, z = self.robot.data.root_quat_w[ids].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = torch.cos(yaw), torch.sin(yaw)
        velocity = torch.zeros(len(ids), 6, device=self.device)
        velocity[:, 0] = c * heading[:, 0] - s * heading[:, 1]
        velocity[:, 1] = s * heading[:, 0] + c * heading[:, 1]
        self.robot.write_root_velocity_to_sim(velocity, env_ids=ids)
        self.needs_refill[ids] = False
        self.replan[ids] = True

    def _start_from_demonstrations(self, ids: torch.Tensor, starts: dict | None = None) -> None:
        """Reference state initialisation (Peng et al., 2018) on the contract.

        The body is put in a state the task's demonstrations visited, the command is
        the one they were following, and the generator's history is the ``z*`` that
        led there -- so the first plan is a plan the generator knows, and the body
        practises following the task's motion, not only starting it.
        """
        starts = self.starts if starts is None else starts
        pick = starts["index"][torch.randint(len(starts["index"]), (len(ids),), device=self.device)]
        t, n = pick[:, 0], pick[:, 1]
        quat = starts["base_quat"][t, n]
        position = starts["base_pos"][t, n] + self._env.scene.env_origins[ids]
        linear = quat_apply(quat, starts["base_lin_vel"][t, n])
        angular = quat_apply(quat, starts["base_ang_vel"][t, n])
        self.robot.write_root_pose_to_sim(torch.cat([position, quat], -1), env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.cat([linear, angular], -1), env_ids=ids)
        self.robot.write_joint_state_to_sim(
            starts["joint_pos"][t, n], starts["joint_vel"][t, n], env_ids=ids
        )
        self.vel_command_b[ids] = starts["command"][t, n]
        self.is_standing_env[ids] = starts["command"][t, n].norm(dim=-1) == 0
        steps = self.interpolation.shape[1]
        offsets = torch.arange(steps - 1, -1, -1, device=self.device)
        window = starts["skill_state"][t[:, None] - offsets, n[:, None]]
        window = self.generator.skill_norm(window)
        self.history[ids] = window + self.sigma * torch.randn_like(window)
        self.physical[ids] = self.history[ids]
        self.needs_refill[ids] = False
        self.replan[ids] = True

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

    def encoder_input(self) -> torch.Tensor:
        """What a body's encoder reads: its proprioception and last action, no command."""
        env = self._env
        return torch.cat(
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

    def _estimate(self) -> torch.Tensor:
        if self.encoder is None:
            return self._true_now + self.sigma * torch.randn_like(self._true_now)
        estimate = self._deployed_estimate()
        if self.cfg.estimate_noise > 0.0:
            # Robustness probe: the deployed estimate read worse, by multiples of its precision.
            precision = self.precision * self.cfg.estimate_noise
            estimate = estimate + precision * torch.randn_like(estimate)
        return estimate

    def _deployed_estimate(self) -> torch.Tensor:
        with torch.no_grad():
            learned = self.encoder(self.encoder_input())
        if self.cfg.estimator == "encoder":
            return learned
        # The kinematic part of z* from the body's own model, the loads from the encoder.
        load = self.generator.skill_norm.inverse(learned)[:, _LOAD]
        data = self.robot.data
        z = proprioceptive_skill_state(
            self.body,
            data.body_link_quat_w,
            data.body_com_pos_w,
            data.body_com_lin_vel_w,
            data.body_com_ang_vel_w,
            load,
            base_velocity=self.cfg.base_velocity,
            corners=self.sole_corners,
        )
        return self.generator.skill_norm(z)

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
        if self.biped is not None:
            widened = self.tolerance_procedural[self.age]
            tolerance = torch.where(self.procedural[:, None], widened, tolerance)
        error = (self._reward_true - due) / tolerance
        if self.load_log_tolerance is not None:
            # The loads as the generator models them, log(load + 0.01): a foot left on the
            # ground with a few per cent of the weight where the plan lifts it -- contact for
            # the duty factor -- costs several tolerances here, under one in linear units.
            plain = self.generator.skill_norm.inverse
            true = plain(self._reward_true)[:, self._load].clamp(min=0.0)
            asked = plain(due)[:, self._load].clamp(min=0.0)
            error[:, self._load] = (
                torch.log(true + LOAD_EPSILON) - torch.log(asked + LOAD_EPSILON)
            ) / self.load_log_tolerance
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
        if self.cfg.tolerance == "spread":
            tolerance = self.integral_tolerance[self.age]
        else:
            # A body that follows without bias drifts from the reference like a random
            # walk, sqrt(t); a biased one drifts like t, and only a long enough
            # integral tells the two apart.
            elapsed = (self.anchored + 1).float()[:, None]
            tolerance = self.integral_rate * elapsed.sqrt()
        # How far the body is from where the reference has got to, in leg lengths --
        # nothing while no plan is held (the start pose is the last episode's).
        lag = (reached[:, :2] - planned[:, :2]).norm(dim=-1)
        self.last_lag = torch.where(self.has_plan, lag, 0.0)
        progress = (reached - planned) / tolerance
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
        valid = self.has_plan
        self.field_error = torch.where(valid[:, None], error, self.field_error)
        error = torch.nan_to_num(error, nan=1e6, posinf=1e6)
        joint = error.mean(dim=-1)
        self.last_error = torch.where(valid, joint, self.last_error)
        state = error[:, :-1].mean(dim=-1)
        self.last_error_state = torch.where(valid, state, self.last_error_state)
        self._joint = joint
        return torch.exp(-0.5 * joint / kernel**2) * valid

    def tracking_likelihood(
        self, kernels: Sequence[float], weights: Sequence[float]
    ) -> torch.Tensor:
        """``Σ w_k exp(-e / (2 k²))``: the plan's likelihood with the error's scale unknown.

        How far a body can get from its plan is not only the encoder's error: a
        body cannot reproduce a model's plan, or another body's, to the
        millimetre. With that scale ``s`` unknown and a power-law prior on it, the
        likelihood is ``∫ p(s) N(e; 0, s²) ds``, heavy-tailed: it peaks at the plan
        and keeps a gradient at every distance. The scales ``k`` and weights ``w``
        are a quadrature of that integral (:data:`OCTAVES`).
        """
        valid = self.has_plan
        base = self.tracking_reward(1.0)
        joint = self._joint
        total = weights[0] * base
        for kernel, weight in zip(kernels[1:], weights[1:], strict=True):
            total = total + weight * torch.exp(-0.5 * joint / kernel**2) * valid
        return total

    def _draw(self, ids: torch.Tensor, anchor: torch.Tensor) -> None:
        """Draw plans for ``ids``; ``anchor`` marks those that start from the body.

        A plan drawn from the body's history starts at the body's CoM and heading.
        One drawn from the task's own trajectory (autonomous) starts where that
        trajectory has got to -- the previous plan's start moved by what it
        integrated over the steps it was held -- so the trajectory's displacement
        carries across plans, as a reference clip's root does.
        """
        held = (self.age[ids] - 1).clamp(min=0)
        moved = self.integrals[ids, held]
        c, s = torch.cos(self.start_yaw[ids]), torch.sin(self.start_yaw[ids])
        length = self.units.length
        reference_com = self.start_com[ids] + length * torch.stack(
            [c * moved[:, 0] - s * moved[:, 1], s * moved[:, 0] + c * moved[:, 1]], dim=-1
        )
        reference_yaw = self.start_yaw[ids] + moved[:, 2]
        space = self.resample_space
        window = resample_rows(self.generator, self.interpolation, self.history[ids], space)
        command = self.vel_command_b[ids] * self.command_scale
        if self.tasks is not None:
            rows, integrals = self._draw_tasks(ids, window, command, space)
        else:
            noise = torch.randn(len(ids), self.generator.noise_size, device=self.device)
            with torch.no_grad(), torch.autocast("cuda", torch.float16, self.cfg.half_precision):
                plan = self.generator.sample(window, command, noise).float()
            # A body still learning to stand writes histories no teacher ever did; the
            # generator's answer to them is meaningless but must stay finite.
            plan = torch.nan_to_num(plan, nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)
            full = torch.cat([window[:, -1:], plan], dim=1)
            rows = resample_rows(self.generator, self.plan_matrix, full, space)
            rows = apply_plan_contact(rows, self.generator.skill_norm, self.cfg.plan_contact)
            step = float(self.generator.grid_step)
            integrals = plan_integrals(self.generator.skill_norm.inverse(plan), step)
            integrals = torch.cat([torch.zeros_like(integrals[:, :1]), integrals], dim=1)
            integrals = torch.einsum("lk,nkd->nld", self.plan_matrix, integrals)
        if self.trust is not None:
            # A second sample from the same history: two plans at the same lead time differ
            # by the generator's sampling noise exactly, over the steps a prior reads.
            with torch.no_grad(), torch.autocast("cuda", torch.float16, self.cfg.half_precision):
                second = self.generator.sample(window, command, torch.randn_like(noise)).float()
            second = torch.nan_to_num(second, nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)
            second = resample_rows(
                self.generator, self.plan_matrix, torch.cat([window[:, -1:], second], dim=1), space
            )
            second = apply_plan_contact(second, self.generator.skill_norm, self.cfg.plan_contact)
            task_disagreement = (rows - second).pow(2)
        applied = torch.zeros(rows.shape[1:], device=self.device)
        if self.biped is not None:
            # The biped's trajectory, read at the body's own instants, with its exact
            # displacement and turn for integrals.
            ahead, moved, _ = self.biped.plan(ids)
            if self.cfg.procedural_sway > 0.0:
                # The trunk sways with the gait -- pitch twice a stride, roll once -- which
                # the point mass leaves free and every real gait has.
                phase = (
                    self.biped.phase[ids, None].float()
                    + self._row_times / self.biped.period[ids, None].float()
                )
                pitch = self.sway_amplitude[ids, :1] * torch.sin(4 * math.pi * phase)
                roll = self.sway_amplitude[ids, 1:] * torch.sin(2 * math.pi * phase)
                ahead = ahead.clone()
                ahead[..., _TILT] = sway(ahead[..., _TILT], pitch, roll)
            ahead = self.generator.skill_norm(ahead).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)
            if (
                self.cfg.procedural_plan_noise
                and self.trust is not None
                and bool(self.trust.seen[0])
            ):
                # The biped's plans carry the noise the task's generator is measured to have.
                applied = self.trust.noise[0].clone()
                ahead = ahead + smooth_noise(tuple(ahead.shape), applied, device=self.device)
            chosen = self.procedural[ids]
            rows = torch.where(chosen[:, None, None], ahead, rows)
            integrals = torch.where(chosen[:, None, None], moved, integrals)
        if self.trust is not None:
            # Row by row, the noise against how far the plan reaches from the body now; the
            # biped's noise is the one added to it, known exactly.
            far = (rows - self._true_now[ids][:, None]).pow(2)
            biped = self.procedural[ids]
            self.trust.update(0, task_disagreement[~biped], far[~biped])
            known = (2.0 * applied).expand(int(biped.sum()), -1, -1)
            self.trust.update(1, known, far[biped])
        self.plan[ids] = rows
        if self.trust is not None:
            other = second
            if self.biped is not None:
                other = torch.where(self.procedural[ids][:, None, None], rows, second)
            self.plan_second[ids] = other
        self.integrals[ids] = integrals
        com, yaw = self._frame_now
        self.start_com[ids] = torch.where(anchor[:, None], com[ids], reference_com)
        self.start_yaw[ids] = torch.where(anchor, yaw[ids], reference_yaw)
        self.anchored[ids] = torch.where(anchor, 0, self.anchored[ids])
        self.age[ids] = 0
        self.has_plan[ids] = True

    # -- command term hooks ------------------------------------------------

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.just_reset[ids] = True
        self.needs_refill[ids] = True
        self.has_plan[ids] = False
        self.age[ids] = 0
        self.anchored[ids] = 0
        self.far_steps[ids] = 0
        self.resyncs[ids] = 0.0
        every = torch.arange(self.num_envs, device=self.device)[ids]
        if self.biped is not None:
            # A procedural episode starts from the default pose: the biped starts from
            # the body, not the body from a demonstration.
            if self.cfg.procedural_probes:
                self.procedural[every] = True
            else:
                draw = torch.rand(len(every), device=self.device)
                self.procedural[every] = draw < self.cfg.procedural_fraction
            if self.cfg.procedural_sway > 0.0:
                amplitude = torch.rand(len(every), 2, device=self.device)
                sign = torch.where(torch.rand(len(every), 2, device=self.device) < 0.5, -1.0, 1.0)
                self.sway_amplitude[every] = self.cfg.procedural_sway * amplitude * sign
        task = every[~self.procedural[every]]
        if self.tasks is not None:
            self._reset_tasks(every, task)
            # The starts below are the contract's own, for its task's episodes.
            task = task[self.task[task] == 0]
        if self.starts is not None and self.cfg.start_fraction > 0.0:
            chosen = task[torch.rand(len(task), device=self.device) < self.cfg.start_fraction]
            if len(chosen) > 0:
                self._start_from_demonstrations(chosen)
        if self.windows is not None and self.cfg.start_fraction > 0.0:
            chosen = task[torch.rand(len(task), device=self.device) < self.cfg.start_fraction]
            if len(chosen) > 0:
                self._start_from_contract(chosen)
        return extras

    def _reset_tasks(self, every: torch.Tensor, task: torch.Tensor) -> None:
        """Draw the task of each new episode in ``task`` (not procedural), its commitment, its
        command from its ranges and, for a share of them, a start inside its motion."""
        self.task[every] = 0
        if len(task) > 0:
            self.task[task] = torch.multinomial(self.task_probability, len(task), replacement=True)
        commits = torch.tensor([t["commit"] for t in self.tasks], device=self.device)
        self.commit_env[every] = commits[self.task[every]]
        self._task_commands(task)
        if self.cfg.start_fraction <= 0.0:
            return
        for t, spec in enumerate(self.tasks[1:], start=1):
            mine = task[self.task[task] == t]
            chosen = mine[torch.rand(len(mine), device=self.device) < self.cfg.start_fraction]
            if len(chosen) == 0:
                continue
            if spec["starts"] is not None:
                self._start_from_demonstrations(chosen, spec["starts"])
            elif spec["windows"] is not None:
                self._start_from_contract(chosen, spec["windows"])

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        if self.tasks is not None:
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
            self._task_commands(ids[~self.procedural[ids]])
        self.replan[env_ids] = True

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["resyncs"] = self.resyncs
        metric = self.metrics["contract_error"]
        self.metrics["contract_error"] = 0.98 * metric + 0.02 * self.last_error
        self.metrics["lag"] = 0.98 * self.metrics["lag"] + 0.02 * self.last_lag
        procedural = self.procedural.float()
        self.metrics["procedural"] = procedural
        smoothed = self.metrics["contract_error"]
        self.metrics["error_procedural"] = smoothed * procedural
        self.metrics["error_task"] = smoothed * (1.0 - procedural)
        if self.tasks is not None:
            # Each task's share of the episodes that follow a task, logged per task.
            for t, task in enumerate(self.tasks):
                self.metrics[f"task_{task['name']}"] = ((self.task == t) & ~self.procedural).float()

    def _update_command(self) -> None:
        super()._update_command()
        if self.tasks is not None:
            self._task_headings()
        self._true_now = self.true_skill()
        self._frame_now = self.frame()
        estimate = self._estimate()
        ready = ~self.just_reset
        refill = ready & self.needs_refill
        carried = ready & ~self.needs_refill
        if self.biped is not None:
            self.biped.advance()
            start = (refill & self.procedural).nonzero().squeeze(-1)
            if len(start) > 0:
                self.biped.anchor(start, self.generator.skill_norm.inverse(estimate[start]))

        filled = estimate[:, None].expand_as(self.history)
        shifted = torch.cat([self.physical[:, 1:], estimate[:, None]], dim=1)
        self.physical = torch.where(
            refill[:, None, None],
            filled,
            torch.where(carried[:, None, None], shifted, self.physical),
        )
        if self.cfg.autonomous:
            # The task's own trajectory goes on: the generator reads the plan it
            # committed to (with the noise it was trained under), not the body.
            due = self.due()
            due = due + self.sigma * torch.randn_like(due)
            planned = torch.cat([self.history[:, 1:], due[:, None]], dim=1)
            following = carried & self.has_plan
            self.history = torch.where(
                refill[:, None, None],
                filled,
                torch.where(following[:, None, None], planned, self.history),
            )
            # Only a body that has left the task altogether is caught up with.
            far = following & (self.last_error > self.cfg.resync_error)
            self.far_steps = torch.where(far, self.far_steps + 1, 0)
            resync = self.far_steps >= self.cfg.resync_steps
            self.history = torch.where(resync[:, None, None], self.physical, self.history)
            # A plan drawn with none held -- after a reset, or a start inside the
            # demonstrations -- starts from the body: there is no trajectory yet.
            anchor = refill | resync | ~self.has_plan
            self.replan = self.replan | resync
            self.far_steps = torch.where(resync, 0, self.far_steps)
            self.resyncs = self.resyncs + resync.float()
        else:
            self.history = self.physical.clone()
            anchor = torch.ones_like(refill)
        self.age = torch.where(carried, self.age + 1, self.age)
        self.anchored = torch.where(carried, self.anchored + 1, self.anchored)
        draw = refill | (carried & (self.replan | (self.age >= self.commit_env)))
        ids = draw.nonzero().squeeze(-1)
        if len(ids) > 0:
            self._draw(ids, anchor[ids])
        self.replan = self.replan & ~draw
        self.needs_refill = self.needs_refill & ~ready
        self.just_reset[:] = False
        self.estimate = estimate

    def plan_gain(self) -> torch.Tensor:
        """``(N, length, 27)``: how far each env's plan is trusted at each lead time, by its
        source (1 without trust)."""
        if self.trust is None:
            return torch.ones_like(self.plan)
        task, biped = self.trust.gain(0), self.trust.gain(1)
        return torch.where(self.procedural[:, None, None], biped, task)

    # -- what the tracker and its critic read --------------------------------

    def observation(self) -> torch.Tensor:
        """``(N, view * 27 + 27)``: the held plan relative to ``ẑ*``, then ``ẑ*``, in the
        tracker's own units when ``observe_units`` is set (``latent/units.py``)."""
        return tracker_view(
            self.upcoming(), self.estimate, self.has_plan, _PLAN_LIMIT, self.observed_units
        )

    def path(self) -> torch.Tensor:
        """``(N, view * 3)``: where the reference will be over the view, seen from the body.

        For each of the next ``view`` steps, the reference's CoM in the body's
        heading frame relative to the body's CoM now (leg lengths) and the
        reference's heading relative to the body's (radians). ``z*`` says how the
        body should be moving, not where it should be; this is the where -- the
        target root position every motion tracker observes. Without it a reward
        on the reference's displacement depends on a lag the tracker cannot see.
        On the body's own history the reference starts at the body, and this is
        the plan's own path ahead (a real body integrates it from its velocity
        estimate over one plan).
        """
        index = self.age[:, None] + torch.arange(self.view, device=self.device)
        planned = self.integrals[self._rows[:, None], index]
        c, s = torch.cos(self.start_yaw)[:, None], torch.sin(self.start_yaw)[:, None]
        length = self.units.length
        com, yaw = self._frame_now
        dx = self.start_com[:, None, 0] - com[:, None, 0]
        dx = dx + length * (c * planned[..., 0] - s * planned[..., 1])
        dy = self.start_com[:, None, 1] - com[:, None, 1]
        dy = dy + length * (s * planned[..., 0] + c * planned[..., 1])
        cb, sb = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        turn = self.start_yaw[:, None] + planned[..., 2] - yaw[:, None]
        path = torch.stack(
            [
                (cb * dx + sb * dy) / length,
                (-sb * dx + cb * dy) / length,
                torch.atan2(torch.sin(turn), torch.cos(turn)),
            ],
            dim=-1,
        )
        path = torch.where(~self.has_plan[:, None, None], 0.0, path)
        return torch.nan_to_num(path.flatten(1), nan=0.0).clamp(-_PLAN_LIMIT, _PLAN_LIMIT)

    def privileged(self) -> torch.Tensor:
        """``(N, view * 27 + 28)``: the plan relative to the true ``z*``, the true ``z*``, age."""
        true = self._true_now
        relative = self.upcoming() - true[:, None]
        relative = torch.where(~self.has_plan[:, None, None], 0.0, relative)
        age = (self.age.float() / self.commit_env)[:, None]
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

    sole_corners: list[list[list[float]]] | None = None
    """Each foot's sole corners in its link frame, metres: where a pitched foot touches the
    ground, for the kinematic estimator's height. None: the sole points."""

    leg_length: float = MISSING
    """The body's unit of length, metres."""

    sensor_name: str = "contact_forces"
    use_encoder: bool = False
    """Read ``ẑ*`` from the contract's encoder (deployed path) instead of truth plus noise."""

    estimator: str = "encoder"
    """What ``use_encoder`` reads ``ẑ*`` with. ``encoder``: the learned encoder alone.
    ``kinematic``: the body's kinematic model for every field but the feet's loads, which
    stay the encoder's (:func:`~...latent.skill_torch.proprioceptive_skill_state`) -- the
    encoder learned on the states its body practised and errs far beyond them, where
    kinematics does not."""

    base_velocity: bool = True
    """Whether the body senses its base's linear velocity; without it, the kinematic
    estimator measures velocities from the more loaded foot (leg odometry)."""

    noise_scale: float = 1.0
    """Multiplies the encoder error used as noise on the true state while training."""

    estimate_noise: float = 0.0
    """On the deployed path only, a robustness probe: Gaussian noise added to the deployed
    estimate, in multiples of the estimator's measured precision (the contract's ``sigma``).
    0 in training and in every evaluation."""

    flow_solver: str | None = None
    """Overrides the generator's integrator (``euler`` or ``midpoint``); None keeps it."""

    flow_steps: int | None = None
    """Overrides the generator's integration steps; None keeps them."""

    load_space: str = "linear"
    """How the reward scores each foot's load: ``linear`` (normalised units, at the encoder's
    precision) or ``log`` -- ``log(load + 0.01)``, the generator's own space, at the deployed
    estimator's precision there (``load_precision``). In linear units a foot grazing the ground
    with 5 % of the weight where the plan lifts it costs under one tolerance; every tracker did
    so in ~19 % of its planned swing, and the duty factor fails 21-25 of its ~24 failing
    probes of the family."""

    load_precision: str = ""
    """With ``load_space="log"``: a ``load_precision.py`` report; its ``log_rms`` is the scale."""

    resample_space: str = "linear"
    """How the contract is read between the generator's grid and the body's instants (the
    history onto the grid, each plan back onto the body's steps): ``linear``, as every run
    before 2026-11-04, or ``generator`` -- in the generator's own space, the loads as
    ``log(load + 0.01)``. Linear blends of loads lengthen every stance a body is asked for,
    by 0.02-0.03 of the duty on the NAO (``latent/resampling.py``). A
    body whose step equals the grid's (the G1) is bit-identical in either space."""

    plan_contact: str = "as_drawn"
    """What the task's plans say about a loaded foot's height: ``as_drawn``, as the generator
    drew it, as every run before 2026-11-05; or ``ground`` -- each foot a plan loads above
    the contact threshold put on the ground by its height alone
    (:func:`~...latent.plan_contact.ground_loaded_feet`); or ``stick`` -- grounded, and each
    loaded foot's velocity set to zero (:func:`~...latent.plan_contact.stick_loaded_feet`).
    The generator's plans hold loaded feet above the ground and move them, where its
    teachers never did. The procedural biped's plans are
    untouched."""

    observe_units: str | None = None
    """A contract whose statistics the tracker's view (:meth:`ReferenceCommand.observation`)
    is normalised with: the one the tracker trained under. ``None``, as every run before
    2026-11-07, observes in ``contract``'s. L4 makes a task's generator the contract, and the
    NAO's walk contract normalises ``z*`` differently from the G1's (``latent/units.py``). Only the
    view changes; the generator and the encoder keep
    the contract's units."""

    tolerance: str = "precision"
    """``precision``: the encoder's error (a plan is a commitment); ``spread``: the
    generator's spread of plans, which proved too lenient -- where the future is
    uncertain, as at a start, not following the drawn plan cost almost nothing."""

    autonomous: bool = False
    """Let the task's trajectory run on its own plans instead of restarting each plan
    from the body's state. Restarting lets a body steer what it is asked to do -- every
    tracker so trained settled into standing, the easiest regime to follow precisely --
    while a reference the body cannot steer has to be followed (as in DeepMimic). The
    price is the generator reading its own plans (see ``chain_contract.py``)."""

    resync_error: float = 25.0
    """Joint error, in tolerances, beyond which a body counts as having left the task."""

    resync_steps: int = 6
    """Consecutive such steps before the task restarts from the body's own history."""

    start_states: str | None = None
    """A capture of the task's demonstrations (with ``base_pos``) to start episodes from."""

    start_windows: str | None = None
    """A capture of the task's demonstrations by any body: its ``z*`` windows seed starts
    on a body without demonstrations of its own (:meth:`ReferenceCommand._start_from_contract`)."""

    start_fraction: float = 0.0
    """Share of resets that start from a demonstrated state instead of the default pose."""

    half_precision: bool = False
    """Draw plans under float16 autocast (the state, history and plan stay float32)."""

    procedural_fraction: float = 0.0
    """Share of episodes whose plans come from the point-mass biped, not the task's generator."""

    procedural_probes: list[dict] | None = None
    """Fixed gaits of the family, one per env in turn (evaluation); every episode follows one."""

    procedural_tolerance: str = "calibrated"
    """``calibrated``: the biped's model error widens the tolerance on the fields it cannot
    derive; ``precision``: the encoder's error everywhere, as for the task's plans."""

    procedural_calibration: str | None = None
    """``procedural_calibration.json``: the biped's model error per component."""

    procedural_seed: int = 0

    procedural_load_shape: tuple[float, float] = (1.0, 1.0)
    """Range of the time warp of a walking stance's half-sine of load (``GaitRanges.load_shape``,
    ``latent/procedural.py``): ``(1, 1)``, the half-sine, as every tracker before 2026-11-04
    practised; ``(0.5, 2)`` adds the slow load transfers of real stances."""

    plan_trust: bool = False
    """Measure each plan source's sampling noise online (``latent/plan_trust.py``): two plans a
    commitment apart, on the steps they share. A physics prior weighs the plan by it."""

    trust_rate: float = 0.02
    """The running estimates' update rate, per draw."""

    procedural_plan_noise: bool = False
    """Give the biped's plans the noise measured on the task's generator's, smooth as a
    sampler's, so a tracker practises the plans it will be handed."""

    procedural_sway: float = 0.0
    """Largest trunk sway, pitch and roll, radians, of a procedural episode (uniform per
    episode, either sign). The walk teacher's sways 0.015 and 0.017 rad."""

    tasks: list[dict] | None = None
    """Further tasks for a decoder trained on several. ``contract`` and ``ranges`` are task 0. Each
    further entry gives:

    * ``name``;
    * ``contract``: a ``generator.pt``; only its generator is used, and it must share the
      contract's grid, history, view and units of ``z*``;
    * ``ranges``: ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` **dimensionless** (velocities
      over ``V``, yaw rate times ``T``), and ``heading`` in radians. The same task then
      reaches every body in its own units;
    * optionally ``starts`` (a capture of this body's states, as ``start_states``) or
      ``windows`` (any body's capture, as ``start_windows``) for its share of starts.

    Every episode that is not procedural follows one task, drawn by ``task_weights``. The
    tracker is not told which."""

    task_weights: list[float] | None = None
    """Relative weight of each task, task 0 first; uniform by default."""


OCTAVES = tuple(2.0**j for j in range(8))
"""Scales of the likelihood's quadrature, in tolerances: an octave apart, 1 to 128.

The first reward had three, at 1, 10 and 100 with weights 1, 0.5 and 0.25 --
the same power law, ``w ∝ k^(-0.301)``, sampled two decades apart. Between
them the likelihood was flat: at 15 tolerances, where a tracker practising
the procedural family sat, getting a swing right changed the reward by 0.2 %
per tolerance, and the swings were not learned. An octave apart, the same
integral keeps its gradient; the three old scales are among these, with
their weights."""

OCTAVE_WEIGHTS = tuple(k ** (-math.log10(2.0)) for k in OCTAVES)


__all__ = [
    "OCTAVES",
    "OCTAVE_WEIGHTS",
    "ReferenceCommand",
    "ReferenceCommandCfg",
    "plan_interpolation",
]
