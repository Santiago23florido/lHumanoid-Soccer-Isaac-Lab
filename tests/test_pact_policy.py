"""PACT as a policy: the trackers' plan views decoded to raw z*, and plans of any length."""

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
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import (  # noqa: E402
    PactPolicy,
    plan_from_view,
    view_steps,
)
from humanoid_transfer.transfer.latent.units import tracker_view  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
DIMS = PactDims(
    width=32,
    heads=4,
    history=2,
    free_tokens=2,
    body_tokens=2,
    observation=3,
    anchor_inputs=True,
    timed_plan=True,
)


def _body(name: str):
    card = json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))
    return describe(card), contact_joints(card)


def test_view_steps_reads_the_trackers_observation_widths():
    # Measured on the co-execution captures: G1 387 (body 120), NAO 210 (63), H1 360 (63).
    assert view_steps(387, 120) == 8
    assert view_steps(210, 63) == 4
    assert view_steps(360, 63) == 9
    with pytest.raises(ValueError):
        view_steps(388, 120)


def test_plan_from_view_inverts_the_trackers_view():
    gen = torch.Generator().manual_seed(0)
    view, n = 4, 5
    mean, std = torch.randn(27, generator=gen), torch.rand(27, generator=gen) + 0.5
    raw_plan = torch.randn(n, view, 27, generator=gen)
    raw_estimate = torch.randn(n, 27, generator=gen)
    observed = tracker_view(
        (raw_plan - mean) / std, (raw_estimate - mean) / std, torch.ones(n, dtype=torch.bool), 50.0
    )
    path = torch.randn(n, view * 3, generator=gen)
    tail = torch.cat([observed, path], dim=1).numpy()
    upcoming, estimate, decoded_path = plan_from_view(tail, view, mean.numpy(), std.numpy())
    np.testing.assert_allclose(upcoming, raw_plan.numpy(), atol=1e-5)
    np.testing.assert_allclose(estimate, raw_estimate.numpy(), atol=1e-5)
    np.testing.assert_allclose(decoded_path, path.numpy().reshape(n, view, 3), atol=0)


def _inputs(count: int, steps: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    return dict(
        joint_obs=torch.randn(2, count, DIMS.history, DIMS.observation, generator=gen),
        imu=torch.randn(2, DIMS.history, DIMS.imu, generator=gen),
        anchor_in=torch.randn(2, 27, generator=gen),
        plan=torch.randn(2, steps, 27, generator=gen),
        plan_relative=torch.randn(2, steps, 27, generator=gen),
        plan_time=torch.linspace(0.1, 0.8, steps).expand(2, -1).clone(),
        plan_path=torch.randn(2, steps, 3, generator=gen),
    )


def test_masked_plan_steps_do_not_reach_the_targets():
    (g1, g1_contacts), (nao, nao_contacts) = _body("g1"), _body("nao")
    bodies = stack_bodies([g1, nao], [g1_contacts, nao_contacts])
    count = bodies.mask.shape[1]
    model = PactPolicy(DIMS).eval()
    x = _inputs(count, steps=9, seed=1)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 4:] = False  # the NAO's view has 4 steps
    with torch.no_grad():
        learned, targets = model(bodies, **x, plan_mask=mask)
        for key in ("plan", "plan_relative", "plan_path"):
            x[key][1, 4:] = 1e3
        _, again = model(bodies, **x, plan_mask=mask)
    assert learned.shape == (2, 6) and targets.shape == (2, count)
    assert (targets[~bodies.mask] == 0).all()
    torch.testing.assert_close(targets, again)


def test_a_plan_step_changes_the_targets_when_unmasked():
    (g1, g1_contacts), _ = _body("g1"), None
    bodies = stack_bodies([g1, g1], [g1_contacts, g1_contacts])
    model = PactPolicy(DIMS).eval()
    x = _inputs(bodies.mask.shape[1], steps=8, seed=2)
    mask = torch.ones(2, 8, dtype=torch.bool)
    with torch.no_grad():
        _, before = model(bodies, **x, plan_mask=mask)
        x["plan"][:, -1] += 1.0
        _, after = model(bodies, **x, plan_mask=mask)
    assert (before - after).abs().max() > 1e-6


def test_gradients_reach_every_policy_parameter():
    (g1, g1_contacts), (nao, nao_contacts) = _body("g1"), _body("nao")
    bodies = stack_bodies([g1, nao], [g1_contacts, nao_contacts])
    model = PactPolicy(DIMS)
    x = _inputs(bodies.mask.shape[1], steps=6, seed=3)
    learned, targets = model(bodies, **x, plan_mask=torch.ones(2, 6, dtype=torch.bool))
    (learned.square().mean() + targets.square().mean()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_the_plan_limit_is_the_trackers():
    import re

    from humanoid_transfer.transfer.latent.pact.policy import PLAN_LIMIT

    source = (
        REPO / "source/humanoid_transfer/humanoid_transfer/transfer/tasks/tracking/command.py"
    ).read_text("utf-8")
    assert float(re.search(r"^_PLAN_LIMIT = ([0-9.]+)", source, re.M).group(1)) == PLAN_LIMIT


def test_the_no_anchor_ablation_ignores_the_anchor_inputs():
    (g1, g1_contacts), (nao, nao_contacts) = _body("g1"), _body("nao")
    bodies = stack_bodies([g1, nao], [g1_contacts, nao_contacts])
    dims = PactDims(**{**vars(DIMS), "anchor_inputs": False})
    model = PactPolicy(dims).eval()
    x = _inputs(bodies.mask.shape[1], steps=6, seed=7)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        _, before = model(bodies, **x, plan_mask=mask)
        x["anchor_in"] = x["anchor_in"] + 100.0
        _, after = model(bodies, **x, plan_mask=mask)
    torch.testing.assert_close(before, after)
    assert not any("anchor" in name for name, _ in model.named_parameters())
