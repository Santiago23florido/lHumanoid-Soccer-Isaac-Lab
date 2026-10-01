"""Checks on the shared latent model's structure, plumbing and training.

The structural tests matter more than the training one. What makes the
latent space *shared* is which module is allowed to see what -- the skill
policy only the shared space, the decoder only the body's own joints -- and
that is a property of the wiring, so it is tested on the wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.similarity import VELOCITY, UnitSystem  # noqa: E402
from humanoid_transfer.transfer.latent.dataset import (  # noqa: E402
    Transitions,
    dimensionless_command,
    falling_steps,
    valid_steps,
)
from humanoid_transfer.transfer.latent.losses import LossWeights, latent_losses  # noqa: E402
from humanoid_transfer.transfer.latent.model import (  # noqa: E402
    G1_OBSERVATION,
    LatentDims,
    ObservationLayout,
)
from humanoid_transfer.transfer.latent.train import TrainConfig, build, train  # noqa: E402

UNITS = UnitSystem(length=0.65457, mass=32.2389)


def _synthetic(samples: int = 512, seed: int = 0) -> Transitions:
    """Transitions where skill and action are fixed linear maps of the observation.

    The action depends on the joints (the decoder's own context) and on the
    nine base terms, which can only reach the decoder through the 16-wide
    latent action -- so the problem is solvable, but only through the latent.
    """
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(samples, 123, generator=generator)
    next_obs = obs + 0.1 * torch.randn(samples, 123, generator=generator)
    to_skill = torch.randn(123, 27, generator=generator) / 11.0
    to_skill[9:12] = 0.0  # the encoder never sees the command, so the state cannot depend on it
    to_action = torch.zeros(123, 37)
    to_action[:9] = torch.randn(9, 37, generator=generator) / 3.0
    to_action[12:86] = torch.randn(74, 37, generator=generator) / 9.0
    return Transitions(
        obs=obs,
        next_obs=next_obs,
        skill=obs @ to_skill,
        next_skill=next_obs @ to_skill,
        command=obs[:, 9:12] * 0.1,
        action=obs @ to_action,
        env=torch.arange(samples) % 8,
        falling=torch.zeros(samples, dtype=torch.bool),
    )


# --- wiring -----------------------------------------------------------------


def test_the_skill_policy_sees_only_the_shared_space() -> None:
    dims = LatentDims()
    model = build(G1_OBSERVATION, 37, dims, _synthetic())
    first = model.skill.policy[0]
    assert first.in_features == dims.skill + dims.free + dims.command


def test_the_decoder_sees_the_bodys_joints_and_nothing_about_its_base() -> None:
    """Base velocity, tilt and command (0:12) and the last action (86:123) are excluded."""
    start, stop = G1_OBSERVATION.context
    assert (start, stop) == (12, 86)
    model = build(G1_OBSERVATION, 37, LatentDims(), _synthetic())
    assert model.decoder.net[0].in_features == LatentDims().action + (stop - start)


def test_the_encoder_describes_state_and_never_sees_the_command() -> None:
    observation = torch.arange(123, dtype=torch.float32)[None]
    encoder_input = G1_OBSERVATION.encoder_input(observation)
    assert encoder_input.shape[-1] == G1_OBSERVATION.encoder_size == 120
    assert not set(range(9, 12)) & set(encoder_input[0].int().tolist())


def test_a_layout_rejects_nothing_it_was_not_told_about() -> None:
    """The split is defined by indices only, so a second body's layout is just numbers."""
    layout = ObservationLayout(size=60, command=(6, 9), context=(9, 47))
    assert layout.encoder_size == 57
    assert layout.context_size == 38


# --- units ------------------------------------------------------------------


def test_the_dimensionless_command_is_froude_scaling() -> None:
    command = np.array([0.2488, 0.0, 0.3])
    star = dimensionless_command(command, UNITS)
    assert star[0] == pytest.approx(UNITS.to_dimensionless(0.2488, VELOCITY))
    assert star[2] == pytest.approx(0.3 * UNITS.time)


def test_the_exported_policy_converts_the_command_the_same_way() -> None:
    model = build(G1_OBSERVATION, 37, LatentDims(), _synthetic())
    policy = model.policy(UNITS.velocity, UNITS.time)
    observation = torch.zeros(1, 123)
    observation[0, 9:12] = torch.tensor([0.2488, 0.0, 0.3])
    expected = dimensionless_command(np.array([0.2488, 0.0, 0.3]), UNITS)
    assert policy.dimensionless_command(observation)[0].numpy() == pytest.approx(expected)


# --- data -------------------------------------------------------------------


def test_no_transition_crosses_or_starts_at_a_reset() -> None:
    done = np.zeros((6, 1), dtype=bool)
    done[2, 0] = True  # step 3 starts a new episode
    assert np.flatnonzero(valid_steps(done)[:, 0]).tolist() == [1, 4]


def test_the_steps_before_a_fall_are_flagged_and_time_outs_are_not() -> None:
    terminated = np.zeros((10, 2), dtype=bool)
    terminated[6, 0] = True
    falling = falling_steps(terminated, horizon=3)
    assert np.flatnonzero(falling[:, 0]).tolist() == [4, 5, 6]
    assert not falling[:, 1].any()


# --- objective and training -------------------------------------------------


def test_the_ablation_has_no_anchor_term() -> None:
    data = _synthetic()
    model = build(G1_OBSERVATION, 37, LatentDims(), data)
    losses = latent_losses(
        model.encoder,
        model.skill,
        model.decoder,
        G1_OBSERVATION,
        data,
        LossWeights(anchored=False),
    )
    assert float(losses["anchor"]) == 0.0


def test_training_drives_imitation_and_anchoring_down() -> None:
    data = _synthetic()
    config = TrainConfig(epochs=100, batch_size=128, dims=LatentDims(hidden=64))
    model = build(G1_OBSERVATION, 37, config.dims, data)
    before = latent_losses(
        model.encoder, model.skill, model.decoder, G1_OBSERVATION, data, config.weights, False
    )
    trained, history = train(data, G1_OBSERVATION, config)
    after = latent_losses(
        trained.encoder, trained.skill, trained.decoder, G1_OBSERVATION, data, config.weights, False
    )
    assert float(after["imitation"]) < 0.2 * float(before["imitation"])
    assert float(after["anchor"]) < 0.2 * float(before["anchor"])
    assert len(history) == config.epochs


def test_the_exported_policy_is_the_eager_one() -> None:
    data = _synthetic()
    model = build(G1_OBSERVATION, 37, LatentDims(hidden=32), data)
    policy = model.policy(UNITS.velocity, UNITS.time).eval()
    traced = torch.jit.trace(policy, data.obs[:4])
    assert torch.allclose(traced(data.obs[:16]), policy(data.obs[:16]), atol=1e-6)
