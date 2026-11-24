"""PACT as an rsl-rl actor: the shared policy adapted by PPO to a body (G-N2).

A body enters the shared policy only through its descriptor tokens, so adapting it cheaply
means moving what is the body's: a residual on its ``E_b`` and on its descriptor tokens
(G-N1's ``etune``), optionally every bias and LayerNorm parameter as well (``bitfit``), or
everything (``full``). The action noise's standard deviation is always tuned.

PPO stores observations, not a recurrent state, so the encoder's inputs are an observation
group the environment computes (the per-step features of :func:`joint_features`,
:func:`imu_features` with the observation manager's history, and the anchor inputs of
:func:`~.driver.anchor_inputs`); the actor reads that group and the tracker's own
observation, whose tail is the plan. ``tests/test_pact_actor.py`` holds the actor to
:class:`~.driver.PactDriver`, which drives the bodies in G-D1.

Layout of the inputs group, per environment::

    joint features (history, joints, 3) | IMU (history, 6) | anchor inputs (27)

oldest step first, as the observation manager's history buffers return them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
from humanoid_transfer.transfer.latent.pact.descriptors import (
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import BodyBatch, PactDims, stack_bodies
from humanoid_transfer.transfer.latent.pact.policy import SKILL, PactPolicy, plan_from_view

ARMS = ("zero", "etune", "bitfit", "full")
ANCHOR = 27
IMU = 6


def joint_features(
    q: torch.Tensor,
    qd: torch.Tensor,
    target: torch.Tensor,
    q0: torch.Tensor,
    time_unit: float,
    drive,
) -> torch.Tensor:
    """``(N, J * 3)``: per joint ``q - q0`` (rad), ``qd * sqrt(L/g)`` and the torque the target
    commands (``drive.drive_ratio``, as ``PactDriver`` and the trainer read it)."""
    return torch.stack([q - q0, qd * time_unit, drive.drive_ratio(q, qd, target)], dim=-1).flatten(1)


def imu_features(gravity: torch.Tensor, omega: torch.Tensor, time_unit: float) -> torch.Tensor:
    """``(N, 6)``: projected gravity and the base angular velocity times ``sqrt(L/g)``."""
    return torch.cat([gravity, omega * time_unit], dim=-1)


def inputs_width(joints: int, history: int) -> int:
    return history * joints * 3 + history * IMU + ANCHOR


def split_inputs(
    flat: torch.Tensor, joints: int, history: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``flat (N, width)`` -> ``joint_obs (N, J, H, 3)``, ``imu (N, H, 6)``, ``anchor (N, 27)``."""
    n = flat.shape[0]
    a = history * joints * 3
    b = a + history * IMU
    if flat.shape[1] != b + ANCHOR:
        raise ValueError(f"inputs width {flat.shape[1]}, expected {b + ANCHOR}")
    joint_obs = flat[:, :a].reshape(n, history, joints, 3).transpose(1, 2)
    imu = flat[:, a:b].reshape(n, history, IMU)
    return joint_obs, imu, flat[:, b:]


class BodyResidual(nn.Module):
    """One body's ``E_b`` and descriptor tokens plus learned residuals (zero at the start).

    The descriptor is the same in every row of a batch of one body, so the embodiment
    encoder runs once per call and its tokens are broadcast.
    """

    def __init__(self, inner: nn.Module, table: BodyBatch, tokens: int, width: int) -> None:
        super().__init__()
        self.inner = inner
        self.table = table
        joints = table.mask.shape[1]
        self.delta_e = nn.Parameter(torch.zeros(tokens, width))
        self.delta_d = nn.Parameter(torch.zeros(joints, width))

    def forward(self, bodies: BodyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.table.mask.device != self.delta_e.device:
            self.table = self.table.to(self.delta_e.device)
        embodiment, descriptor = self.inner(self.table)
        n = bodies.mask.shape[0]
        return (
            (embodiment + self.delta_e).expand(n, -1, -1),
            (descriptor + self.delta_d).expand(n, -1, -1),
        )


def bias_and_norm(model: nn.Module) -> list[nn.Parameter]:
    """Every bias and every LayerNorm parameter (BitFit's set), once each."""
    chosen: dict[int, nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters(recurse=False):
                chosen[id(p)] = p
    for name, p in model.named_parameters():
        if name.endswith("bias"):
            chosen[id(p)] = p
    return list(chosen.values())


def load_body(repo: Path, body: str):
    """``(descriptor, contact joints, card, leg-length unit)`` from the body registry."""
    registry = json.loads((repo / "transfer/bodies/bodies.json").read_text("utf-8"))
    entry = registry[body]
    card = json.loads((repo / entry["card"]).read_text("utf-8"))
    unit = entry.get("leg_length_unit")
    return describe(card, leg_length=unit), contact_joints(card), card, unit


class PactActor(nn.Module):
    """A trained :class:`PactPolicy` for one body, as rsl-rl's actor (Gaussian output).

    Built by rsl-rl as ``PactActor(obs, obs_groups, "actor", num_actions, **cfg)``; the
    keyword arguments come from the runner configuration (see ``pact_adapt.py``).
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        checkpoint: str,
        repo: str,
        body: str,
        view: int,
        skill_mean: list[float],
        skill_std: list[float],
        control_dt: float,
        action_joints: list[int],
        action_scale: float,
        arm: str = "etune",
        init_std: float = 0.2,
        plan_group: str = "policy",
        inputs_group: str = "pact",
        drive_stiffness: list[float] | None = None,
        drive_damping: list[float] | None = None,
        drive_effort_limit: list[float] | None = None,
        default_joint_pos: list[float] | None = None,
        sole_points: list[list[float]] | None = None,
        **_unused,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"arm {arm!r} not in {ARMS}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        dims = PactDims(**state["dims"])
        descriptor, contacts, card, unit = load_body(Path(repo), body)
        table = stack_bodies([descriptor], [contacts])
        self.policy = PactPolicy(dims)
        self.policy.load_state_dict(state["state"])
        self.policy.embodiment = BodyResidual(
            self.policy.embodiment, table, dims.body_tokens, dims.width
        )
        self.impedance_anchor = bool(dims.impedance_anchor)
        self.joints = table.mask.shape[1]
        self.history = dims.history
        self.view = view
        self.plan_group, self.inputs_group = plan_group, inputs_group
        groups = obs_groups[obs_set]
        if plan_group not in groups or inputs_group not in groups:
            raise ValueError(f"the actor reads {plan_group!r} and {inputs_group!r}, not {groups}")
        width = obs[inputs_group].shape[-1]
        if width != inputs_width(self.joints, self.history):
            raise ValueError(f"{inputs_group!r} has {width} numbers, not this body's inputs")
        if len(action_joints) != output_dim:
            raise ValueError(f"{len(action_joints)} action joints for {output_dim} actions")

        def buffer(name: str, value) -> None:
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=torch.float32))

        for name in ("z_mean", "z_std", "a_mean", "a_std"):
            buffer(name, state[name])
        buffer("skill_mean", skill_mean)
        buffer("skill_std", skill_std)
        time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
        self.time_unit = time_unit
        self.alpha_dt = float(1.0 - math.exp(-(control_dt / time_unit) / 0.08))
        buffer("times", (np.arange(1, view + 1) * control_dt / time_unit)[None])
        self.register_buffer("action_joints", torch.as_tensor(action_joints, dtype=torch.long))
        buffer("action_scale", action_scale)
        if self.impedance_anchor:
            self.anchors = KinematicAnchors(card, unit, sole_points)
            self.drive = DriveLoads(card, self.anchors)
            if drive_stiffness is not None and drive_damping is not None and drive_effort_limit is not None:
                self.drive.set_drives(drive_stiffness, drive_damping, drive_effort_limit)
            q0 = card["joints"]["default_pos"] if default_joint_pos is None else default_joint_pos
            buffer("q0", q0)
        else:
            self.anchors = None
            self.drive = None
        # rsl_rl only where PPO runs: the offline trainer imports this module without it.
        from rsl_rl.modules.distribution import GaussianDistribution

        self.distribution = GaussianDistribution(output_dim, init_std=init_std, std_type="scalar")
        self.arm = arm
        self.tuned = self.select(arm)

    # ------------------------------------------------------------------ parameters

    def select(self, arm: str) -> list[nn.Parameter]:
        """Freeze every parameter the arm does not tune; return the tuned ones."""
        residual = self.policy.embodiment
        tuned = [self.distribution.std_param]
        if arm in ("etune", "bitfit", "full"):
            tuned += [residual.delta_e, residual.delta_d]
        if arm == "bitfit":
            tuned += bias_and_norm(self.policy)
        if arm == "full":
            tuned += list(self.policy.parameters())
        ids = {id(p) for p in tuned}
        for p in self.parameters():
            p.requires_grad_(id(p) in ids)
        return [p for p in self.parameters() if p.requires_grad]

    def tuned_count(self) -> int:
        return sum(p.numel() for p in self.tuned)

    # ------------------------------------------------------------------ forward

    def mean(self, obs) -> torch.Tensor:
        """The policy's action in the action term's units."""
        flat, policy_obs = obs[self.inputs_group], obs[self.plan_group]
        joint_obs, imu, anchor = split_inputs(flat, self.joints, self.history)
        n = flat.shape[0]
        tail = policy_obs[:, policy_obs.shape[-1] - (self.view * (SKILL + 3) + SKILL) :]
        upcoming, estimate, path = plan_from_view(tail, self.view, self.skill_mean, self.skill_std)
        bodies = self.policy.embodiment.table
        bodies = bodies.to(flat.device).take(torch.zeros(n, dtype=torch.long, device=flat.device))
        extra = {}
        if self.impedance_anchor:
            q = joint_obs[:, :, -1, 0] + self.q0
            qd = joint_obs[:, :, -1, 1] / self.time_unit
            gravity = imu[:, -1, :3]
            cmd_r = joint_obs[:, :, -1, 2]
            with torch.no_grad():
                phys = self.drive.operational_anchor_from_ratio(q, gravity, cmd_r)
                phys["alpha_dt"] = flat.new_full((n,), self.alpha_dt)
            extra["phys"] = phys
        _, out = self.policy(
            bodies,
            joint_obs,
            imu,
            (anchor - self.a_mean) / self.a_std,
            (upcoming - self.z_mean) / self.z_std,
            (upcoming - estimate[:, None]) / self.z_std,
            self.times.expand(n, -1),
            path,
            torch.ones(n, self.view, dtype=torch.bool, device=flat.device),
            **extra,
        )
        if self.impedance_anchor:
            out = self.drive.targets_from_torque(q, qd, out, default_pos=self.q0)
        return out[:, self.action_joints] / self.action_scale

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        mean = self.mean(obs)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    # ------------------------------------------------------------------ rsl-rl's interface

    def reset(self, dones=None, hidden_state=None) -> None:
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None) -> None:
        pass

    def update_normalization(self, obs) -> None:
        pass  # PACT normalises with its training statistics

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("PACT actors are saved as adapted checkpoints, not TorchScript")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("PACT actors are saved as adapted checkpoints, not ONNX")

    # ------------------------------------------------------------------ saving

    def adapted_checkpoint(self, base: dict) -> dict:
        """``base`` (the starting checkpoint) with this body's adapted weights and residuals.

        ``state`` keeps :class:`PactPolicy`'s layout (the residual wrapper unwrapped);
        ``residual`` holds ``delta_e``, ``delta_d`` and the body they belong to.
        """
        residual = self.policy.embodiment
        state = {}
        for key, value in self.policy.state_dict().items():
            if key.startswith("embodiment.delta_"):
                continue
            state[key.replace("embodiment.inner.", "embodiment.", 1)] = value.detach().cpu()
        out = dict(base)
        out["state"] = state
        out["residual"] = {
            "delta_e": residual.delta_e.detach().cpu(),
            "delta_d": residual.delta_d.detach().cpu(),
        }
        out["action_std"] = self.distribution.std_param.detach().cpu()
        out["arm"] = self.arm
        return out
