"""HPT as the per-body-stem baseline: body-specific stems and heads around a shared trunk.

HPT ("Heterogeneous Pre-trained Transformers", Wang et al., NeurIPS 2024; ``references/
INDEX.md`` #18) shares a transformer trunk across embodiments and gives each embodiment its
own **stem** (its observations, as features attended by 16 learned tokens) and **head** (the
trunk's pooled output to its actions). Kept here: a proprioception stem per body (each step
of the joint and IMU window one feature, read by 16 learned queries through cross-attention),
one shared stem for the plan (each plan step one feature: the plan in ``z*`` is the same
format for every body), a shared trunk over both stems' tokens, and a head per body that
outputs that body's joint targets.

Nothing describes a body to it: a body is its own stem and head, so a body it was not trained
on has none (no zero-shot body); adding one means training a new stem and head, which is
HPT's transfer and the comparison for PACT's addition of a body (G-N). Its ``forward`` takes
:class:`~.policy.PactPolicy`'s arguments plus ``body_ids`` (each row's index into the bodies
it was built for); the anchor inputs are ignored and the learned ``z*`` fields are zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from humanoid_transfer.transfer.latent.pact.model import BodyBatch

SKILL = 27
LEARNED = 6


@dataclass(frozen=True)
class HptDims:
    """Sizes of the HPT baseline (about PACT's parameter count for the four bodies at the
    defaults); ``joints`` is each body's joint count, in body order."""

    bodies: tuple[str, ...] = ()
    joints: tuple[int, ...] = ()
    observation: int = 3
    imu: int = 6
    history: int = 4
    width: int = 192
    heads: int = 6
    trunk_layers: int = 3
    stem_tokens: int = 16
    head_hidden: int = 256


class Stem(nn.Module):
    """Features ``(B, T, F)`` -> ``(B, tokens, W)``: an MLP, then learned queries attend."""

    def __init__(self, features: int, width: int, heads: int, tokens: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(64, width) * 0.02)
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.attend = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        keys = self.embed(features) + self.position[: features.shape[1]]
        queries = self.queries.expand(features.shape[0], -1, -1)
        out, _ = self.attend(queries, keys, keys, key_padding_mask=None if mask is None else ~mask)
        return self.norm(out + queries)


class HptPolicy(nn.Module):
    """Per-body proprioception stems and heads, a shared plan stem and trunk."""

    def __init__(self, dims: HptDims) -> None:
        super().__init__()
        if len(dims.bodies) != len(dims.joints) or not dims.bodies:
            raise ValueError("HptDims needs each body's name and joint count")
        self.dims = dims
        w = dims.width
        self.proprio = nn.ModuleList(
            Stem(j * dims.observation + dims.imu, w, dims.heads, dims.stem_tokens)
            for j in dims.joints
        )
        self.plan = Stem(2 * SKILL + 3 + 1, w, dims.heads, dims.stem_tokens)
        self.source = nn.Parameter(torch.randn(2, w) * 0.02)
        layer = nn.TransformerEncoderLayer(w, dims.heads, 4 * w, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, dims.trunk_layers)
        self.norm = nn.LayerNorm(w)
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(w, dims.head_hidden), nn.GELU(), nn.Linear(dims.head_hidden, j))
            for j in dims.joints
        )

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
        body_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``-> (zeros (B, 6), targets (B, J))``; ``body_ids (B,)`` index ``dims.bodies``."""
        batch, count = joint_obs.shape[:2]
        steps = torch.cat([plan, plan_relative, plan_path, plan_time[..., None]], dim=-1)
        plan_tokens = self.plan(steps, plan_mask) + self.source[1]
        out = joint_obs.new_zeros(batch, count)
        for b, joints in enumerate(self.dims.joints):
            rows = torch.nonzero(body_ids == b, as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            per_step = torch.cat(
                [joint_obs[rows, :joints].transpose(1, 2).flatten(2), imu[rows]], dim=-1
            )  # (n, history, J * obs + imu)
            tokens = torch.cat([self.proprio[b](per_step) + self.source[0], plan_tokens[rows]], dim=1)
            pooled = self.norm(self.trunk(tokens)).mean(dim=1)
            out[rows, :joints] = self.heads[b](pooled)
        return joint_obs.new_zeros(batch, LEARNED), out * bodies.mask


__all__ = ["HptDims", "HptPolicy"]
