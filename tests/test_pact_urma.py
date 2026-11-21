"""The URMA baseline: any joint set, padded joints and plan steps inert, PACT's interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    BodyDescriptor,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.model import stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.urma import UrmaDims, UrmaPolicy  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
DIMS = UrmaDims(history=2, plan_steps=9, description=16, joint=32, core=64, action=32, decoder=32)


def _body(name: str):
    card = json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))
    return describe(card), contact_joints(card)


def _inputs(count: int, steps: int, seed: int, batch: int = 2):
    gen = torch.Generator().manual_seed(seed)
    return dict(
        joint_obs=torch.randn(batch, count, DIMS.history, DIMS.observation, generator=gen),
        imu=torch.randn(batch, DIMS.history, DIMS.imu, generator=gen),
        anchor_in=torch.randn(batch, 27, generator=gen),
        plan=torch.randn(batch, steps, 27, generator=gen),
        plan_relative=torch.randn(batch, steps, 27, generator=gen),
        plan_time=torch.linspace(0.1, 0.8, steps).expand(batch, -1).clone(),
        plan_path=torch.randn(batch, steps, 3, generator=gen),
        plan_mask=torch.ones(batch, steps, dtype=torch.bool),
    )


def test_padded_joints_and_masked_plan_steps_are_inert():
    (g1, g1_c), (nao, nao_c) = _body("g1"), _body("nao")
    bodies = stack_bodies([g1, nao], [g1_c, nao_c])
    count = bodies.mask.shape[1]
    model = UrmaPolicy(DIMS).eval()
    x = _inputs(count, steps=9, seed=1)
    x["plan_mask"][1, 4:] = False  # the NAO's view has 4 steps
    with torch.no_grad():
        learned, targets = model(bodies, **x)
        for key in ("plan", "plan_relative", "plan_path"):
            x[key][1, 4:] = 1e3
        x["joint_obs"][1, ~bodies.mask[1]] = 1e3  # the NAO's padded joints
        _, again = model(bodies, **x)
    assert learned.shape == (2, 6) and (learned == 0).all()
    assert (targets[~bodies.mask] == 0).all()
    torch.testing.assert_close(targets, again)


def test_a_body_alone_or_padded_gets_the_same_targets():
    """A body's targets do not depend on the batch it is in (the padding)."""
    (g1, g1_c), (nao, nao_c) = _body("g1"), _body("nao")
    alone = stack_bodies([nao], [nao_c])
    padded = stack_bodies([nao, g1], [nao_c, g1_c])
    n = alone.mask.shape[1]
    model = UrmaPolicy(DIMS).eval()
    x = _inputs(padded.mask.shape[1], steps=4, seed=2)
    x_alone = {k: v[:1] for k, v in x.items()}
    x_alone["joint_obs"] = x["joint_obs"][:1, :n]
    with torch.no_grad():
        _, a = model(alone, **x_alone)
        _, b = model(padded, **x)
    torch.testing.assert_close(a[0], b[0, :n], rtol=1e-5, atol=1e-6)


def test_permuting_joints_permutes_the_targets():
    h1, h1_c = _body("h1")
    bodies = stack_bodies([h1], [h1_c])
    count = bodies.mask.shape[1]
    order = np.random.default_rng(3).permutation(count)
    inverse = np.argsort(order)
    permuted = BodyDescriptor(
        name=h1.name,
        joint_names=tuple(h1.joint_names[i] for i in order),
        joints=h1.joints[order],
        body=h1.body,
        tree_distance=h1.tree_distance[np.ix_(order, order)],
        ancestor=h1.ancestor[np.ix_(order, order)],
        leg_length=h1.leg_length,
        mass=h1.mass,
    )
    shuffled = stack_bodies([permuted], [[int(inverse[c]) for c in h1_c]])
    order = torch.as_tensor(order)
    model = UrmaPolicy(DIMS).eval()
    x = _inputs(count, steps=9, seed=4, batch=1)
    x_shuffled = dict(x, joint_obs=x["joint_obs"][:, order])
    with torch.no_grad():
        _, a = model(bodies, **x)
        _, b = model(shuffled, **x_shuffled)
    torch.testing.assert_close(a[:, order], b, rtol=1e-5, atol=1e-6)


def test_gradients_reach_every_parameter_and_long_plans_are_refused():
    (g1, g1_c), (nao, nao_c) = _body("g1"), _body("nao")
    bodies = stack_bodies([g1, nao], [g1_c, nao_c])
    model = UrmaPolicy(DIMS)
    x = _inputs(bodies.mask.shape[1], steps=6, seed=5)
    _, targets = model(bodies, **x)
    targets.square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    with pytest.raises(ValueError):
        model(bodies, **_inputs(bodies.mask.shape[1], steps=10, seed=6))


def test_the_driver_runs_a_urma_checkpoint_as_the_model_does():
    """A checkpoint of kind "urma" drives through PactDriver with the same windows and plan."""
    from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
    from humanoid_transfer.transfer.latent.pact.descriptors import GRAVITY
    from humanoid_transfer.transfer.latent.pact.driver import PactDriver
    from humanoid_transfer.transfer.latent.pact.policy import plan_from_view

    card = json.loads((REPO / REGISTRY["nao"]["card"]).read_text("utf-8"))
    unit = REGISTRY["nao"].get("leg_length_unit")
    descriptor, contacts = describe(card, leg_length=unit), contact_joints(card)
    anchors = KinematicAnchors(card, unit)
    dims = UrmaDims(history=3, plan_steps=9, description=16, joint=32, core=64, action=32, decoder=32)
    torch.manual_seed(0)
    policy = UrmaPolicy(dims).eval()
    gen = torch.Generator().manual_seed(2)
    checkpoint = {
        "kind": "urma",
        "state": policy.state_dict(),
        "dims": vars(dims),
        "z_mean": torch.randn(27, generator=gen).numpy(),
        "z_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
        "a_mean": torch.zeros(27).numpy(),
        "a_std": torch.ones(27).numpy(),
    }
    view, n, dt, joints = 4, 3, 0.02, [5, 2, 7, 11]
    mean, std = torch.randn(27, generator=gen), torch.rand(27, generator=gen) + 0.5
    q0 = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32)
    driver = PactDriver(
        checkpoint, descriptor, contacts, anchors, DriveLoads(card, anchors), q0, joints, 0.25,
        view, mean, std, dt, n, "cpu",
    )  # fmt: skip
    j = len(q0)
    steps = []
    for k in range(4):
        g = torch.Generator().manual_seed(10 + k)
        q, qd = q0 + 0.1 * torch.randn(n, j, generator=g), torch.randn(n, j, generator=g)
        t = q + 0.05 * torch.randn(n, j, generator=g)
        grav = torch.tensor([[0.0, 0.0, -1.0]]).expand(n, -1)
        steps.append((q, qd, t, grav, 0.5 * torch.randn(n, 3, generator=g)))
    obs = torch.randn(n, 70 + view * 30 + 27, generator=gen)
    for q, qd, t, grav, w in steps:
        action = driver(obs, q, qd, t, grav, w)
    time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
    last = steps[-dims.history :]
    joint_obs = torch.stack(
        [torch.stack([q - q0, qd * time_unit, t - q0], -1) for q, qd, t, _, _ in last], dim=2
    )
    imu = torch.stack([torch.cat([gr, w * time_unit], -1) for _, _, _, gr, w in last], dim=1)
    up, est, path = plan_from_view(obs[:, -(view * 30 + 27) :], view, mean, std)
    z_mean, z_std = (torch.as_tensor(checkpoint[k]) for k in ("z_mean", "z_std"))
    with torch.no_grad():
        _, out = policy(
            stack_bodies([descriptor], [contacts]).take(torch.zeros(n, dtype=torch.long)),
            joint_obs,
            imu,
            torch.zeros(n, 27),
            (up - z_mean) / z_std,
            (up - est[:, None]) / z_std,
            (torch.arange(1, view + 1) * dt / time_unit).expand(n, -1),
            path,
            torch.ones(n, view, dtype=torch.bool),
        )
    torch.testing.assert_close(action, out[:, joints] / 0.25, atol=1e-5, rtol=1e-4)
