"""The free part of the shared latent: tokens, bottleneck, physics, alignment, constraint."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.free_latent import (  # noqa: E402
    BodyClassifier,
    BodyStems,
    DualAscent,
    FreeLatentDims,
    FreeLatentEncoder,
    PhysicsPredictor,
    accuracy_excess,
    alignment_loss,
    entity_features,
    gaussian_kl,
    prediction_loss,
    reverse_gradient,
)

SLICES = {field.name: SKILL_LAYOUT.slice_of(field.name) for field in SKILL_LAYOUT.fields}


def _body(joints: int, actions: int, legs, acted, n: int = 32, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    encoder_input = torch.randn(n, 9 + 2 * joints + actions, generator=gen)
    skill = torch.randn(n, 27, generator=gen)
    return entity_features(encoder_input, skill, legs, acted, joints, SLICES), skill


def test_each_body_groups_its_sensors_into_the_same_entities() -> None:
    g1, _ = _body(37, 37, ((0, 3, 6), (1, 4, 7)), ((0, 3, 6), (1, 4, 7)))
    nao, _ = _body(42, 19, ((0, 2), (1, 3)), ((0, 2), (1, 3)))
    assert set(g1) == set(nao) == {"trunk", "left_foot", "right_foot"}
    # Trunk: the base (9) and the CoM's fields of z* (1 + 3 + 3 + 3 + 3).
    assert g1["trunk"].shape[-1] == nao["trunk"].shape[-1] == 22
    # A foot: its leg's positions, velocities, actions, and its sole's 3 + 3 + 1 of z*.
    assert g1["left_foot"].shape[-1] == 3 * 3 + 7
    assert nao["right_foot"].shape[-1] == 3 * 2 + 7


def test_a_step_of_training_moves_every_shared_module() -> None:
    dims = FreeLatentDims(width=16, heads=2, layers=1)
    g1, skill_g1 = _body(37, 37, ((0, 3, 6), (1, 4, 7)), ((0, 3, 6), (1, 4, 7)))
    nao, skill_nao = _body(42, 19, ((0, 2), (1, 3)), ((0, 2), (1, 3)), seed=1)
    stems = {
        "g1": BodyStems({k: v.shape[-1] for k, v in g1.items()}, dims.width),
        "nao": BodyStems({k: v.shape[-1] for k, v in nao.items()}, dims.width),
    }
    encoder, predictor, classifier = (
        FreeLatentEncoder(dims), PhysicsPredictor(dims), BodyClassifier(dims)
    )  # fmt: skip
    mean_g1, logvar_g1 = encoder(stems["g1"](g1))
    mean_nao, logvar_nao = encoder(stems["nao"](nao))
    balance = torch.randn(32, dims.balance)
    falls = torch.rand(32) < 0.2
    losses = prediction_loss(predictor, skill_g1, mean_g1, balance, falls)
    total = losses["balance"] + losses["falls"] + gaussian_kl(mean_g1, logvar_g1).mean()
    total = total + alignment_loss(mean_g1, mean_nao)
    free = torch.cat([mean_g1, mean_nao])
    labels = torch.cat([torch.ones(32), torch.zeros(32)])
    logits = classifier(reverse_gradient(free, 1.0))
    total = total + torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    total.backward()
    for module in (encoder, predictor, classifier, stems["g1"], stems["nao"]):
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), type(module).__name__


def test_the_reversed_gradient_flips_sign_and_scales() -> None:
    x = torch.randn(5, requires_grad=True)
    (reverse_gradient(x, 2.0) * 3.0).sum().backward()
    torch.testing.assert_close(x.grad, torch.full((5,), -6.0))


def test_the_multiplier_rises_while_violated_and_never_goes_negative() -> None:
    dual = DualAscent(rate=0.5)
    assert dual.step(0.2) == pytest.approx(0.1)
    assert dual.step(0.2) == pytest.approx(0.2)
    assert dual.step(-1.0) == 0.0


def test_chance_is_measured_against_the_larger_class() -> None:
    labels = torch.cat([torch.ones(75), torch.zeros(25)])
    excess, se = accuracy_excess(torch.ones(100), labels)
    assert excess == pytest.approx(0.0)
    assert se > 0


def test_alignment_is_not_satisfied_by_shrinking_the_latent() -> None:
    gen = torch.Generator().manual_seed(2)
    a, b = torch.randn(64, 8, generator=gen), torch.randn(64, 8, generator=gen)
    assert alignment_loss(a, b) == pytest.approx(float(alignment_loss(0.01 * a, 0.01 * b)))
    assert float(alignment_loss(a, a)) == 0.0
