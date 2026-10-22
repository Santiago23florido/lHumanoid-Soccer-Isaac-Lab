"""Decoder v2's deep-learning core: a temporal encoder of the plan and a residual trunk.

The plain tracker reads its plan as 243 numbers in a flat MLP: the rows of the view
are separate inputs, and nothing in the network says that row ``k`` and row ``k + 1``
are the same quantity a step apart. Here the plan is read as what it is, a short
sequence of tokens -- each row's ``z*`` relative to ``ẑ*`` and where the reference
will be (:meth:`ReferenceCommand.tokens`) -- by two 1-D convolutions shared over
time, so a pattern learned at one lead time is recognised at another. The rest of
the observation (the body's proprioception with a short history, ``ẑ*``, the
prior's offset) joins the plan's features in a residual MLP with LayerNorm
(pre-norm residual blocks, He et al. 2016; Xiong et al. 2020), which trains deeper
than a plain MLP without its features drifting in scale.

The input is the actor's flat, normalised observation, the plan's tokens first
(``tokens * token_width`` columns): that order is the core policy group's
(:func:`~humanoid_transfer.transfer.tasks.tracking.core_cfg.core_policy`), so the
network needs no other description of its input and an exported actor is driven
by the same one tensor as the plain one. Only torch is imported, so the exporter
rebuilds it without a simulator.
"""

from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"swish": nn.SiLU, "elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
"""By rsl-rl's names (its ``swish`` is SiLU), so an actor's config names both networks alike."""


class ResidualBlock(nn.Module):
    """``x + W2 σ(W1 LN(x))``: a pre-norm residual block of one width."""

    def __init__(self, width: int, activation: str) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.inner = nn.Linear(width, width)
        self.act = ACTIVATIONS[activation]()
        self.outer = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.outer(self.act(self.inner(self.norm(x))))


class PlanCoreNet(nn.Module):
    """Plan tokens through a temporal encoder, then everything through a residual trunk.

    ``hidden_dims`` reads as the plain MLP's: the first width is the trunk's, every
    following equal width adds a residual block, and a smaller one ends the trunk with
    a projection; its last width is the features' (what the physics heads read).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256),
        activation: str = "swish",
        tokens: int = 8,
        token_width: int = 30,
        plan_channels: int = 64,
        plan_features: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= tokens * token_width:
            raise ValueError(f"an input of {input_dim} has no room for {tokens} plan tokens")
        self.tokens = tokens
        self.token_width = token_width
        self.plan_width = tokens * token_width
        act = ACTIVATIONS[activation]
        # Kept in the state dict, so an exported actor is rebuilt with the activation it had.
        self.register_buffer("activation_code", torch.tensor(list(ACTIVATIONS).index(activation)))
        self.plan_encoder = nn.Sequential(
            nn.Conv1d(token_width, plan_channels, kernel_size=3, padding=1),
            act(),
            nn.Conv1d(plan_channels, plan_channels, kernel_size=3, padding=1),
            act(),
        )
        self.plan_projection = nn.Sequential(
            nn.Linear(plan_channels * tokens, plan_features), act()
        )
        widths = list(hidden_dims)
        rest = input_dim - self.plan_width
        self.stem = nn.Sequential(nn.Linear(plan_features + rest, widths[0]), act())
        blocks: list[nn.Module] = []
        width = widths[0]
        for following in widths[1:]:
            if following == width:
                blocks.append(ResidualBlock(width, activation))
            else:
                projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, following), act())
                blocks.append(projection)
                width = following
        self.trunk = nn.Sequential(*blocks)
        self.feature_dim = width
        self.head = nn.Linear(width, output_dim)
        self.recompute = True
        """In training with gradients, recompute the activations in the backward pass instead
        of keeping them (activation checkpointing): the same gradients, a third of the memory.
        PPO's update on 4096 envs otherwise needed 2.5 GB for this actor against 0.9 GB for
        the plain MLP, and spilled out of an 8 GB card at 82 s per iteration."""

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        plan = x[:, : self.plan_width].reshape(-1, self.tokens, self.token_width)
        encoded = self.plan_encoder(plan.transpose(1, 2)).flatten(1)
        joined = torch.cat([self.plan_projection(encoded), x[:, self.plan_width :]], dim=-1)
        return self.trunk(self.stem(joined))

    @torch.jit.unused
    def _recomputed(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._features, x, use_reentrant=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting() and self.recompute and self.training:
            if torch.is_grad_enabled():
                return self._recomputed(x)
        return self._features(x)

    def action(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action(self.features(x))


CORE_OPTIONS = ("tokens", "token_width", "plan_channels", "plan_features")
"""The keyword arguments of :class:`PlanCoreNet` an actor's config carries (``network_cfg``)."""


def core_from_state(state: dict, prefix: str = "mlp.") -> PlanCoreNet:
    """Rebuild a :class:`PlanCoreNet` from an actor's state dict, its sizes read off the weights."""
    weights = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    conv = weights["plan_encoder.0.weight"]  # (channels, token_width, 3)
    projection = weights["plan_projection.0.weight"]  # (plan_features, channels * tokens)
    stem = weights["stem.0.weight"]  # (width, plan_features + rest)
    channels, token_width = conv.shape[0], conv.shape[1]
    tokens = projection.shape[1] // channels
    plan_features = projection.shape[0]
    input_dim = stem.shape[1] - plan_features + tokens * token_width
    widths = [stem.shape[0]]
    index = 0
    while f"trunk.{index}.inner.weight" in weights or f"trunk.{index}.1.weight" in weights:
        if f"trunk.{index}.inner.weight" in weights:
            widths.append(widths[-1])
        else:
            widths.append(weights[f"trunk.{index}.1.weight"].shape[0])
        index += 1
    output_dim = weights["head.weight"].shape[0]
    activation = list(ACTIVATIONS)[int(weights["activation_code"])]
    net = PlanCoreNet(
        input_dim, output_dim, widths, activation, tokens, token_width, channels, plan_features
    )
    net.load_state_dict(weights)
    return net


def is_core_state(state: dict, prefix: str = "mlp.") -> bool:
    return f"{prefix}plan_encoder.0.weight" in state


__all__ = ["CORE_OPTIONS", "PlanCoreNet", "ResidualBlock", "core_from_state", "is_core_state"]
