"""Cross-body translation: one body's ``z*`` trajectory to another's, through their ``E_b``.

On a co-execution (the same plan replayed on every body, ``latent/replay.py``) two bodies
executing the same instant of the same plan differ only by what each body can do. A
translator reads the source body's window of ``z*`` with the source's embodiment tokens
and writes the target's, conditioned on the target's: the trajectory part of PACT's
cross-body attention, trained by masked prediction (hide one body, predict it from
another).

The output is the source window plus a learned correction, so copying the source -- what
a body-blind contract assumes, every body executing the same ``z*`` -- is where training
starts, and what the correction adds is what the bodies' descriptions explain.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import Block, BodyBatch, EmbodimentEncoder, PactDims


class CrossBodyTranslator(nn.Module):
    """``(source window, source body, target body) -> target window``, normalised ``z*``."""

    def __init__(
        self, dims: PactDims, window: int, skill: int = 27, layers: int = 2, use_body: bool = True
    ) -> None:
        super().__init__()
        w = dims.width
        self.use_body = use_body
        self.embodiment = EmbodimentEncoder(dims) if use_body else None
        self.state = nn.Sequential(nn.Linear(skill, w), nn.GELU(), nn.Linear(w, w))
        self.time = nn.Parameter(torch.randn(window, w) * 0.02)
        self.source_role = nn.Parameter(torch.zeros(w))
        self.target_role = nn.Parameter(torch.zeros(w))
        self.encoder = nn.ModuleList(Block(w, dims.heads, dims.ff_mult) for _ in range(layers))
        self.cross = nn.ModuleList(
            Block(w, dims.heads, dims.ff_mult, cross=True) for _ in range(layers)
        )
        self.self_attend = nn.ModuleList(Block(w, dims.heads, dims.ff_mult) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, skill))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def bodies(self, table: BodyBatch) -> torch.Tensor | None:
        """``(B, body_tokens, W)``: every body's embodiment tokens (None without bodies)."""
        return None if self.embodiment is None else self.embodiment(table)[0]

    def forward(
        self,
        source: torch.Tensor,
        source_body: torch.Tensor,
        target_body: torch.Tensor,
        tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        """``source (N, T, skill)``, body indices ``(N,)`` into ``tokens`` (``bodies()``)."""
        memory = self.state(source) + self.time[None]
        queries = self.time[None].expand(len(source), -1, -1)
        if self.use_body:
            e_source = tokens[source_body] + self.source_role
            e_target = tokens[target_body] + self.target_role
            memory = torch.cat([memory, e_source, e_target], dim=1)
            queries = queries + e_target.mean(dim=1, keepdim=True)
        for layer in self.encoder:
            memory = layer(memory)
        for cross, attend in zip(self.cross, self.self_attend, strict=True):
            queries = attend(cross(queries, memory=memory))
        return source + self.out(queries)


__all__ = ["CrossBodyTranslator"]
