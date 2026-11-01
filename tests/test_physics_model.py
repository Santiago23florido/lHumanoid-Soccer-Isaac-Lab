"""The tracker's physics heads: their loss joins PPO's through the action, and only in training."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from humanoid_transfer.transfer.latent.physics_model import PhysicsActor  # noqa: E402
from tensordict import TensorDict  # noqa: E402


def _actor() -> tuple[PhysicsActor, TensorDict]:
    torch.manual_seed(0)
    obs = TensorDict(
        {"policy": torch.randn(64, 12), "physics": torch.randn(64, 10)}, batch_size=[64]
    )
    actor = PhysicsActor(
        obs,
        {"actor": ["policy"]},
        "actor",
        output_dim=5,
        hidden_dims=[32, 16],
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        head_dims=[16],
    )
    return actor, obs


def test_the_physics_loss_reaches_the_head_only_through_the_update_pass() -> None:
    actor, obs = _actor()
    actor.train()
    actions = actor(obs, stochastic_output=True)
    log_prob = actor.get_output_log_prob(actions.detach())
    (-log_prob.mean()).backward()
    assert actor.head[0].weight.grad is not None
    assert actor.head[0].weight.grad.abs().sum() > 0
    assert actor.log_variance.grad is not None
    assert "physics" in actor.drain_losses()

    # The same pass without its stochastic draw (the mirror loss's) adds nothing.
    actor.zero_grad()
    actor(obs).sum().backward()
    assert actor.head[0].weight.grad is None or actor.head[0].weight.grad.abs().sum() == 0
    assert actor.drain_losses() == {}


def test_the_heads_change_nothing_the_policy_outputs() -> None:
    actor, obs = _actor()
    actor.eval()
    with torch.no_grad():
        mean = actor(obs)
        plain = actor.mlp(actor.get_latent(obs))
    torch.testing.assert_close(mean, plain)
