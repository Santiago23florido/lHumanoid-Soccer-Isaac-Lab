"""Trusting a plan by its own sampling noise: disagreement, gain, fusion, shaped noise, sway."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.plan_trust import (  # noqa: E402
    PlanTrust,
    fuse,
    smooth_noise,
    step_variation,
    sway,
)


def test_a_deterministic_source_is_trusted_and_a_noisy_one_is_not() -> None:
    trust = PlanTrust(sources=2, length=3, size=2, rate=0.5, device="cpu")
    assert torch.equal(trust.gain(0), torch.ones(3, 2))
    reach = torch.full((1, 3, 2), 4.0)
    trust.update(0, torch.zeros(1, 3, 2), reach)
    # Disagreement 2 is a sample noise of 1, a quarter of the reach: K = 0.75.
    trust.update(1, torch.tensor([[[2.0, 0.0]] * 3]), reach)
    torch.testing.assert_close(trust.gain(0), torch.ones(3, 2))
    torch.testing.assert_close(trust.gain(1), torch.tensor([[0.75, 1.0]] * 3))
    # Noise beyond the reach is never trusted below nothing.
    trust.update(1, torch.full((1, 3, 2), 100.0), reach)
    assert trust.gain(1).min() >= 0.0


def test_fusion_moves_from_the_body_towards_the_plan_by_the_gain() -> None:
    now = torch.zeros(1, 2)
    plan = torch.ones(1, 3, 2)
    fused = fuse(now, plan, torch.tensor([0.25, 1.0]))
    torch.testing.assert_close(fused, torch.tensor([[[0.25, 1.0]] * 3]))


def test_shaped_noise_carries_the_measured_variance_smoothly() -> None:
    gen = torch.Generator().manual_seed(0)
    noise = smooth_noise((20000, 20, 2), torch.tensor([1.0, 0.25]), generator=gen)
    torch.testing.assert_close(noise.var(dim=0).mean(dim=0), torch.tensor([1.0, 0.25]), atol=0.03,
                               rtol=0.05)  # fmt: skip
    # Smooth: consecutive steps differ far less than the noise itself.
    assert step_variation(noise).mean() < 0.05 * noise.var()


def test_sway_turns_gravity_by_the_opposite_rotation_and_keeps_it_unit() -> None:
    down = torch.tensor([[0.0, 0.0, -1.0]])
    turned = sway(down, torch.tensor([0.1]), torch.tensor([0.0]))
    # The base pitched forward: gravity in the base gains a forward component.
    assert turned[0, 0] > 0
    torch.testing.assert_close(turned.norm(dim=-1), torch.ones(1))
    # Rolled the mirror way, the lateral component flips.
    left = sway(down, torch.tensor([0.0]), torch.tensor([0.1]))
    right = sway(down, torch.tensor([0.0]), torch.tensor([-0.1]))
    torch.testing.assert_close(left[0, 1], -right[0, 1])
