"""The HPT baseline: each row through its body's stem and head, padding inert, PACT's interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.hpt import HptDims, HptPolicy  # noqa: E402
from humanoid_transfer.transfer.latent.pact.model import stack_bodies  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))


def _body(name: str):
    card = json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))
    return describe(card), contact_joints(card)


def _setup():
    (g1, g1_c), (nao, nao_c) = _body("g1"), _body("nao")
    dims = HptDims(
        bodies=("g1", "nao"),
        joints=(len(g1.joint_names), len(nao.joint_names)),
        history=2,
        width=32,
        heads=4,
        trunk_layers=1,
        stem_tokens=4,
        head_hidden=16,
    )
    return dims, stack_bodies([g1, nao], [g1_c, nao_c])


def _inputs(count: int, steps: int, seed: int, batch: int = 2):
    gen = torch.Generator().manual_seed(seed)
    return dict(
        joint_obs=torch.randn(batch, count, 2, 3, generator=gen),
        imu=torch.randn(batch, 2, 6, generator=gen),
        anchor_in=torch.randn(batch, 27, generator=gen),
        plan=torch.randn(batch, steps, 27, generator=gen),
        plan_relative=torch.randn(batch, steps, 27, generator=gen),
        plan_time=torch.linspace(0.1, 0.8, steps).expand(batch, -1).clone(),
        plan_path=torch.randn(batch, steps, 3, generator=gen),
        plan_mask=torch.ones(batch, steps, dtype=torch.bool),
    )


def test_rows_go_through_their_body_and_padding_is_inert():
    dims, bodies = _setup()
    model = HptPolicy(dims).eval()
    x = _inputs(bodies.mask.shape[1], steps=8, seed=1)
    x["plan_mask"][1, 4:] = False
    ids = torch.tensor([0, 1])
    with torch.no_grad():
        learned, targets = model(bodies, **x, body_ids=ids)
        for key in ("plan", "plan_relative", "plan_path"):
            x[key][1, 4:] = 1e3
        x["joint_obs"][1, ~bodies.mask[1]] = 1e3
        x["anchor_in"] += 1e3
        _, again = model(bodies, **x, body_ids=ids)
    assert (learned == 0).all() and (targets[~bodies.mask] == 0).all()
    torch.testing.assert_close(targets, again)


def test_a_row_does_not_depend_on_the_others():
    dims, bodies = _setup()
    model = HptPolicy(dims).eval()
    count = bodies.mask.shape[1]
    x = _inputs(count, steps=8, seed=2, batch=4)
    ids = torch.tensor([0, 1, 0, 1])
    four = stack_bodies(*zip(*[_body("g1"), _body("nao")] * 2))
    with torch.no_grad():
        _, all_rows = model(four, **x, body_ids=ids)
        one = {k: v[2:3] for k, v in x.items()}
        _, alone = model(four.take(torch.tensor([2])), **one, body_ids=ids[2:3])
    torch.testing.assert_close(all_rows[2:3], alone, rtol=1e-5, atol=1e-6)


def test_gradients_reach_only_the_bodies_in_the_batch():
    dims, bodies = _setup()
    model = HptPolicy(dims)
    x = _inputs(bodies.mask.shape[1], steps=8, seed=3, batch=1)
    _, targets = model(bodies.take(torch.tensor([0])), **x, body_ids=torch.tensor([0]))
    targets.square().mean().backward()
    assert all(p.grad is not None for p in model.proprio[0].parameters())
    assert all(p.grad is None for p in model.proprio[1].parameters())
    assert all(p.grad is None for p in model.heads[1].parameters())


def test_the_driver_runs_an_hpt_checkpoint_for_its_body():
    from humanoid_transfer.transfer.latent.pact.anchors import DriveLoads, KinematicAnchors
    from humanoid_transfer.transfer.latent.pact.descriptors import GRAVITY
    from humanoid_transfer.transfer.latent.pact.driver import PactDriver
    from humanoid_transfer.transfer.latent.pact.policy import plan_from_view

    dims, _ = _setup()
    card = json.loads((REPO / REGISTRY["nao"]["card"]).read_text("utf-8"))
    unit = REGISTRY["nao"].get("leg_length_unit")
    descriptor, contacts = describe(card, leg_length=unit), contact_joints(card)
    anchors = KinematicAnchors(card, unit)
    torch.manual_seed(0)
    policy = HptPolicy(dims).eval()
    gen = torch.Generator().manual_seed(2)
    checkpoint = {
        "kind": "hpt",
        "state": policy.state_dict(),
        "dims": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(dims).items()},
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
    for k in range(3):
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
            body_ids=torch.full((n,), 1),
        )
    torch.testing.assert_close(action, out[:, joints] / 0.25, atol=1e-5, rtol=1e-4)
