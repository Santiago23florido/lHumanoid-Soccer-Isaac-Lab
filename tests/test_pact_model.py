"""PACT: entity tokens of z*, and one encoder/decoder for bodies of any joint count."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT, mirror  # noqa: E402
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    BodyDescriptor,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import (  # noqa: E402
    Pact,
    PactDims,
    stack_bodies,
)
from humanoid_transfer.transfer.latent.pact.tokens import (  # noqa: E402
    CENTROID_SIZE,
    CONTACT_SIZE,
    merge,
    mirror_entities,
    split,
)

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
DIMS = PactDims(width=32, heads=4, history=2, free_tokens=2, body_tokens=2, plan_steps=3)


def _body(name: str) -> tuple[BodyDescriptor, list[int]]:
    card = json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))
    return describe(card), contact_joints(card)


def _inputs(counts: list[int], seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    batch, count = len(counts), max(counts)
    joint_obs = torch.randn(batch, count, DIMS.history, DIMS.observation, generator=gen)
    imu = torch.randn(batch, DIMS.history, DIMS.imu, generator=gen)
    plan_c = torch.randn(batch, DIMS.plan_steps, CENTROID_SIZE, generator=gen)
    plan_k = torch.randn(batch, DIMS.plan_steps, 2, CONTACT_SIZE, generator=gen)
    return joint_obs, imu, plan_c, plan_k


def test_split_and_merge_are_inverse_and_cover_every_column():
    z = torch.randn(5, SKILL_LAYOUT.size, dtype=torch.float64)
    centroid, contacts = split(z)
    assert centroid.shape == (5, CENTROID_SIZE) and contacts.shape == (5, 2, CONTACT_SIZE)
    assert CENTROID_SIZE + 2 * CONTACT_SIZE == SKILL_LAYOUT.size
    torch.testing.assert_close(merge(centroid, contacts), z)


def test_the_entity_mirror_is_the_skill_state_mirror():
    z = np.random.default_rng(1).normal(size=(7, SKILL_LAYOUT.size))
    centroid, contacts = split(torch.as_tensor(z))
    mirrored = merge(*mirror_entities(centroid, contacts)).numpy()
    np.testing.assert_allclose(mirrored, mirror(z), atol=1e-12)


@pytest.mark.parametrize("body", [k for k in REGISTRY if not k.startswith("$")])
def test_every_registered_body_has_two_contact_joints(body):
    descriptor, contacts = _body(body)
    assert len(contacts) == 2 and contacts[0] != contacts[1]
    assert all(0 <= c < len(descriptor.joint_names) for c in contacts)


def test_bodies_of_different_joint_counts_share_one_model():
    (g1, g1_contacts), (nao, nao_contacts) = _body("g1"), _body("nao")
    bodies = stack_bodies([g1, nao], [g1_contacts, nao_contacts])
    model = Pact(DIMS).eval()
    with torch.no_grad():
        out = model(bodies, *_inputs([len(g1.joint_names), len(nao.joint_names)]))
    count = bodies.mask.shape[1]
    assert out["action"].shape == (2, count)
    assert out["centroid"].shape == (2, CENTROID_SIZE)
    assert out["contacts"].shape == (2, 2, CONTACT_SIZE)
    assert out["latent"].shape == (2, 1 + 2 + DIMS.free_tokens, DIMS.width)
    assert torch.isfinite(out["action"]).all() and torch.isfinite(out["latent"]).all()
    assert (out["action"][~bodies.mask] == 0).all()


def test_a_body_does_not_depend_on_the_padding_of_its_batch():
    (g1, g1_contacts), (nao, nao_contacts) = _body("g1"), _body("nao")
    n = len(nao.joint_names)
    model = Pact(DIMS).eval()
    joint_obs, imu, plan_c, plan_k = _inputs([len(g1.joint_names), n], seed=3)
    with torch.no_grad():
        both = model(
            stack_bodies([g1, nao], [g1_contacts, nao_contacts]), joint_obs, imu, plan_c, plan_k
        )
        alone = model(
            stack_bodies([nao], [nao_contacts]),
            joint_obs[1:, :n],
            imu[1:],
            plan_c[1:],
            plan_k[1:],
        )
    torch.testing.assert_close(both["latent"][1], alone["latent"][0], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(both["action"][1, :n], alone["action"][0], atol=1e-5, rtol=1e-4)


def test_reordering_a_bodys_joints_reorders_its_actions_only():
    descriptor, contacts = _body("h1")
    n = len(descriptor.joint_names)
    order = np.random.default_rng(4).permutation(n)
    inverse = np.argsort(order)
    shuffled = BodyDescriptor(
        name=descriptor.name,
        joint_names=tuple(descriptor.joint_names[i] for i in order),
        joints=descriptor.joints[order],
        body=descriptor.body,
        tree_distance=descriptor.tree_distance[np.ix_(order, order)],
        ancestor=descriptor.ancestor[np.ix_(order, order)],
        leg_length=descriptor.leg_length,
        mass=descriptor.mass,
    )
    shuffled_contacts = [int(inverse[c]) for c in contacts]
    model = Pact(DIMS).eval()
    joint_obs, imu, plan_c, plan_k = _inputs([n], seed=5)
    with torch.no_grad():
        original = model(stack_bodies([descriptor], [contacts]), joint_obs, imu, plan_c, plan_k)
        permuted = model(
            stack_bodies([shuffled], [shuffled_contacts]),
            joint_obs[:, torch.as_tensor(order)],
            imu,
            plan_c,
            plan_k,
        )
    torch.testing.assert_close(permuted["latent"], original["latent"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        permuted["action"][0], original["action"][0, torch.as_tensor(order)], atol=1e-5, rtol=1e-4
    )


def test_gradients_reach_every_parameter():
    (g1, g1_contacts), (cassie, cassie_contacts) = _body("g1"), _body("cassie")
    bodies = stack_bodies([g1, cassie], [g1_contacts, cassie_contacts])
    model = Pact(DIMS)
    out = model(bodies, *_inputs([len(g1.joint_names), len(cassie.joint_names)]))
    loss = out["action"].square().mean() + out["centroid"].square().mean()
    loss = loss + out["contacts"].square().mean()
    loss.backward()
    silent = [name for name, p in model.named_parameters() if p.grad is None or not p.grad.any()]
    assert not silent, silent
