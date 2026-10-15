"""A tracker's network that must also say how its body is held up (decoder v2's heads).

The actor's hidden features feed, besides the action, a head that predicts the
body's balance as measured (:mod:`~...tasks.tracking.balance`: each foot's centre of pressure and
ground force). The prediction is an auxiliary task: its loss shapes the shared
features towards the physics of support, which a plan-to-action map otherwise
never has to represent explicitly (auxiliary tasks in RL, Jaderberg et al.
2017; balance quantities in a humanoid's learning, Poddar et al. 2026).

Each predicted quantity carries a learned log-variance, and its loss is the
Gaussian negative log-likelihood (Kendall, Gal and Cipolla 2018): a quantity the
features cannot predict is down-weighted by the fit itself, so no weight is
chosen by hand. The variance is bounded so a near-perfect fit cannot swamp
PPO's gradient.

rsl-rl's PPO builds its loss itself. The auxiliary loss joins it through the
action's backward pass (:class:`_JoinLoss`: an identity on the action whose
backward also sends a unit gradient to the auxiliary loss), so PPO's update is
used unchanged; :class:`PhysicsPPO` only reports the loss.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict

LOG_VARIANCE = (math.log(0.01), math.log(10.0))
"""Bounds on each predicted quantity's log-variance, in its normalised units."""


class _JoinLoss(torch.autograd.Function):
    """``output`` unchanged; backwards, ``loss`` is differentiated as if added to the total."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        return output.view_as(output)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, torch.ones((), dtype=grad.dtype, device=grad.device)


class PhysicsActor(MLPModel):
    """An MLP actor whose last hidden layer also predicts the group ``physics_group``."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        physics_group: str = "physics",
        head_dims: tuple[int, ...] | list[int] = (128,),
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.physics_group = physics_group
        size = obs[physics_group].shape[-1]
        self.head = MLP(hidden_dims[-1], size, list(head_dims), activation)
        self.log_variance = torch.nn.Parameter(torch.zeros(size))
        self.target_normalizer = EmpiricalNormalization(size)
        self._losses: list[torch.Tensor] = []

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        features = self.get_latent(obs, masks, hidden_state)
        *hidden, last = list(self.mlp)
        for layer in hidden:
            features = layer(features)
        output = last(features)
        # The update's own forward pass: the one whose action log-probability is optimised.
        if stochastic_output and self.training and torch.is_grad_enabled():
            loss = self.physics_loss(features, obs[self.physics_group])
            output = _JoinLoss.apply(output, loss)
            self._losses.append(loss.detach())
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def physics_loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gaussian negative log-likelihood of the measured balance, per quantity, averaged."""
        with torch.no_grad():
            self.target_normalizer.update(target)
        normalised = self.target_normalizer(target)
        predicted = self.head(features)
        log_variance = self.log_variance.clamp(*LOG_VARIANCE)
        squared = (predicted - normalised).pow(2).mean(dim=0)
        return (0.5 * torch.exp(-log_variance) * squared + 0.5 * log_variance).mean()

    def drain_losses(self) -> dict[str, float]:
        """The mean auxiliary loss since the last call, and the mean predicted spread."""
        if not self._losses:
            return {}
        loss = float(torch.stack(self._losses).mean())
        self._losses.clear()
        spread = float(torch.exp(0.5 * self.log_variance.clamp(*LOG_VARIANCE)).mean())
        return {"physics": loss, "physics_spread": spread}


class PhysicsPPO(PPO):
    """rsl-rl's PPO, reporting the actor's auxiliary physics loss with its own."""

    def update(self) -> dict[str, float]:
        losses = super().update()
        drain = getattr(self.actor, "drain_losses", None)
        if drain is not None:
            losses.update(drain())
        return losses


__all__ = ["LOG_VARIANCE", "PhysicsActor", "PhysicsPPO"]
