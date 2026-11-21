"""URMA as the end-to-end baseline: one policy for every body, with no physical anchor.

URMA ("One Policy to Run Them All", Bohlinger et al., CoRL 2024; ``references/INDEX.md``
#13) handles any joint set with three pieces, kept here:

- **Joint encoder.** Each joint's description ``d_j`` is embedded (``e_j``); the joint's
  observations, with ``e_j``, are encoded into a per-joint latent ``h_j``.
- **Attention pooling.** A softmax over the joints, per latent feature, of a projection of
  ``e_j`` divided by a learned temperature weights the ``h_j`` into one body latent: no joint
  count or order is assumed.
- **Universal decoder.** A core MLP turns the body latent and the general observations into
  an action latent; each joint's action is an MLP of that latent and its own ``e_j``.

Each input group goes through its own LayerNorm, as URMA's. What it shares with PACT, so the
comparison isolates the anchor and the architecture: the same descriptors (the Pi groups of
:mod:`.descriptors`, wide-range ones in log10), the same per-joint observations and IMU
history, the same task input (the tracker's plan in ``z*``, its difference to the estimate,
its path and time ahead), the same labels, losses and DAgger. What it does not have: the
analytic anchor inputs, the anchored entity latent and its ``z*`` readout loss, the
embodiment tokens and the graph-biased attention.

Its ``forward`` takes :class:`~.policy.PactPolicy`'s arguments, so the trainer
(``pact_gate_d.py``, arm ``urma_d``) and the driver (:class:`~.driver.PactDriver`) run either
model; the anchor inputs are accepted and ignored, and the learned ``z*`` fields it returns are
zeros (it has none).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from humanoid_transfer.transfer.latent.pact.descriptors import BODY_FEATURES, JOINT_FEATURES
from humanoid_transfer.transfer.latent.pact.model import BodyBatch, descriptor_inputs

SKILL = 27
LEARNED = 6
"""Width of the learned ``z*`` fields PACT returns beside its targets (zeros here)."""


@dataclass(frozen=True)
class UrmaDims:
    """Sizes of the URMA baseline (about PACT's parameter count at the defaults)."""

    observation: int = 3
    imu: int = 6
    history: int = 4
    plan_steps: int = 9
    """Plan steps the core reads; a body's shorter plan is zero-padded and its mask says so."""
    description: int = 128
    joint: int = 256
    core: int = 1024
    action: int = 256
    decoder: int = 512


def _mlp(inputs: int, *widths: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in widths:
        layers += [nn.Linear(inputs, width), nn.ELU()]
        inputs = width
    return nn.Sequential(*layers)


class UrmaPolicy(nn.Module):
    """URMA's joint encoder, attention pooling and universal decoder."""

    def __init__(self, dims: UrmaDims) -> None:
        super().__init__()
        self.dims = dims
        d, j = dims.description, dims.joint
        self.describe = nn.Sequential(nn.LayerNorm(len(JOINT_FEATURES)), _mlp(len(JOINT_FEATURES), d, d))
        self.observe = nn.LayerNorm(dims.history * dims.observation)
        self.encode = _mlp(dims.history * dims.observation + d, j, j)
        self.attention = nn.Linear(d, j)
        self.log_temperature = nn.Parameter(torch.zeros(()))
        plan_width = dims.plan_steps * (2 * SKILL + 3 + 1 + 1)
        self.imu_norm = nn.LayerNorm(dims.history * dims.imu)
        self.body_norm = nn.LayerNorm(len(BODY_FEATURES))
        self.plan_norm = nn.LayerNorm(plan_width)
        general = dims.history * dims.imu + len(BODY_FEATURES) + plan_width
        self.core = _mlp(j + general, dims.core, dims.core, dims.action)
        self.decode = nn.Sequential(_mlp(dims.action + d, dims.decoder, dims.decoder), nn.Linear(dims.decoder, 1))

    def forward(
        self,
        bodies: BodyBatch,
        joint_obs: torch.Tensor,
        imu: torch.Tensor,
        anchor_in: torch.Tensor,
        plan: torch.Tensor,
        plan_relative: torch.Tensor,
        plan_time: torch.Tensor,
        plan_path: torch.Tensor,
        plan_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``-> (zeros (B, 6), targets (B, J))``; arguments as :class:`~.policy.PactPolicy`."""
        batch, count = joint_obs.shape[:2]
        mask = bodies.mask
        e = self.describe(descriptor_inputs(bodies.joints))
        h = self.encode(torch.cat([self.observe(joint_obs.reshape(batch, count, -1)), e], dim=-1))
        logits = self.attention(e) / self.log_temperature.exp()
        logits = logits.masked_fill(~mask[..., None], float("-inf"))
        pooled = (torch.softmax(logits, dim=1) * h).sum(dim=1)

        steps = plan.shape[1]
        if steps > self.dims.plan_steps:
            raise ValueError(f"a plan of {steps} steps; this model reads {self.dims.plan_steps}")
        keep = plan_mask.to(plan.dtype)[..., None]
        per_step = torch.cat(
            [plan, plan_relative, plan_path, plan_time[..., None], torch.ones_like(plan_time)[..., None]],
            dim=-1,
        ) * keep
        padded = per_step.new_zeros(batch, self.dims.plan_steps, per_step.shape[-1])
        padded[:, :steps] = per_step
        general = torch.cat(
            [
                self.imu_norm(imu.reshape(batch, -1)),
                self.body_norm(bodies.body.to(imu.dtype)),
                self.plan_norm(padded.flatten(1)),
            ],
            dim=-1,
        )
        action = self.core(torch.cat([pooled, general], dim=-1))
        out = self.decode(torch.cat([action[:, None].expand(-1, count, -1), e], dim=-1))[..., 0]
        return joint_obs.new_zeros(batch, LEARNED), out * mask


__all__ = ["UrmaDims", "UrmaPolicy"]
