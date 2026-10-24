"""Decoder v2's core: a temporal plan encoder and a residual trunk, exportable as the MLP is."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from humanoid_transfer.transfer.latent.core_net import (  # noqa: E402
    PlanCoreNet,
    core_from_state,
    is_core_state,
)

TOKENS, WIDTH, REST = 8, 30, 50


def _net(**kwargs) -> PlanCoreNet:
    torch.manual_seed(0)
    return PlanCoreNet(TOKENS * WIDTH + REST, 7, [64, 64, 32], **kwargs).eval()


def test_the_trunk_reads_the_widths_as_the_mlp_does():
    net = _net()
    assert net.feature_dim == 32
    x = torch.randn(5, TOKENS * WIDTH + REST)
    assert net(x).shape == (5, 7)
    assert net.features(x).shape == (5, 32)
    # One residual block for the repeated width, one projection to the last.
    assert len(net.trunk) == 2


def test_the_plan_is_read_as_tokens_shared_over_time():
    """The same token at another lead time moves the convolution's output there, not elsewhere."""
    net = _net()
    token = torch.randn(WIDTH)
    early = torch.zeros(1, TOKENS, WIDTH)
    late = torch.zeros(1, TOKENS, WIDTH)
    early[0, 2], late[0, 5] = token, token
    with torch.no_grad():
        a = net.plan_encoder(early.transpose(1, 2))
        b = net.plan_encoder(late.transpose(1, 2))
    # Shared over time: the response moves with the token. Only the second layer's padding
    # differs from its input's background, so its two edge outputs are left out.
    torch.testing.assert_close(a[..., 1:4], b[..., 4:7])


def test_an_input_too_narrow_for_the_plan_is_refused():
    with pytest.raises(ValueError):
        PlanCoreNet(TOKENS * WIDTH, 7)


def test_the_core_is_rebuilt_from_its_state_and_scripts():
    net = _net(activation="elu")
    state = {f"mlp.{k}": v for k, v in net.state_dict().items()}
    assert is_core_state(state)
    rebuilt = core_from_state(state).eval()
    x = torch.randn(4, TOKENS * WIDTH + REST)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x), net(x))
        torch.testing.assert_close(torch.jit.script(rebuilt)(x), net(x))


def test_the_exporter_drives_a_core_checkpoint(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "transfer/scripts/export_actor.py"
    spec = importlib.util.spec_from_file_location("export_actor", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    net = _net()
    width = TOKENS * WIDTH + REST
    state = {f"mlp.{k}": v for k, v in net.state_dict().items()}
    state["obs_normalizer._mean"] = torch.randn(1, width)
    state["obs_normalizer._std"] = torch.rand(1, width) + 0.5
    actor = exporter.actor_from(state)
    x = torch.randn(3, width)
    expected = net((x - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2))
    with torch.no_grad():
        torch.testing.assert_close(torch.jit.script(actor)(x), expected)


rsl_rl = pytest.importorskip("rsl_rl")
from humanoid_transfer.transfer.latent.physics_model import PhysicsActor  # noqa: E402
from tensordict import TensorDict  # noqa: E402


@pytest.mark.parametrize("physics_group", [None, "physics"])
def test_the_actor_hosts_the_core_with_or_without_heads(physics_group):
    torch.manual_seed(0)
    obs = TensorDict(
        {"policy": torch.randn(32, TOKENS * WIDTH + REST), "physics": torch.randn(32, 10)},
        batch_size=[32],
    )
    actor = PhysicsActor(
        obs,
        {"actor": ["policy"]},
        "actor",
        output_dim=5,
        hidden_dims=[64, 64, 32],
        activation="swish",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        physics_group=physics_group,
        head_dims=[16],
        network="plan_core",
        network_cfg={"tokens": TOKENS, "token_width": WIDTH},
    )
    assert isinstance(actor.mlp, PlanCoreNet)
    actor.train()
    actions = actor(obs, stochastic_output=True)
    (-actor.get_output_log_prob(actions.detach()).mean()).backward()
    assert actor.mlp.plan_encoder[0].weight.grad.abs().sum() > 0
    if physics_group is None:
        assert not hasattr(actor, "head")
        assert actor.drain_losses() == {}
    else:
        assert actor.head[0].weight.grad.abs().sum() > 0
        assert "physics" in actor.drain_losses()
    actor.eval()
    with torch.no_grad():
        torch.testing.assert_close(actor(obs), actor.mlp(actor.get_latent(obs)))
        assert actor.trunk(obs).shape == (32, 32)


def test_recomputing_the_activations_changes_no_gradient():
    """Activation checkpointing trades memory for compute only: the gradients are the same."""
    torch.manual_seed(1)
    x = torch.randn(16, TOKENS * WIDTH + REST)
    grads = {}
    for recompute in (False, True):
        net = _net()
        net.train()
        net.recompute = recompute
        net(x).pow(2).sum().backward()
        grads[recompute] = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads[False]) == len(grads[True]) > 0
    for kept, recomputed in zip(grads[False], grads[True], strict=True):
        torch.testing.assert_close(kept, recomputed)
    # The scripted network (the exporter's) compiles around the recomputing path.
    torch.jit.script(_net())
