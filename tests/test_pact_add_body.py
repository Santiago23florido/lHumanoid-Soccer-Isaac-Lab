"""G-N3, adding a body offline: nested probe budgets, each arm's tuned set, and the adapted
PACT checkpoint driving as the model it was trained as."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from humanoid_transfer.transfer.latent.pact.actor import BodyResidual  # noqa: E402
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.hpt import HptDims, HptPolicy  # noqa: E402
from humanoid_transfer.transfer.latent.pact.model import PactDims, stack_bodies  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import PactPolicy  # noqa: E402
from pact_gate_d import PROBES, adapt_parameters, budget_rows, driver_state  # noqa: E402

REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
DIMS = PactDims(
    width=32, heads=4, history=2, free_tokens=2, body_tokens=2, observation=3,
    anchor_inputs=True, timed_plan=True,
)  # fmt: skip


def _body(name: str):
    card = json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))
    return describe(card), contact_joints(card)


def _inputs(count: int, steps: int, seed: int, batch: int = 3):
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


def test_budgets_are_nested_training_probes_of_the_map_capture():
    envs = 3 * PROBES
    group = np.concatenate([np.repeat(np.arange(envs), 5), 1000 + np.repeat(np.arange(envs), 5)])
    test = (group % PROBES) % 4 == 0
    b = SimpleNamespace(name="body", group=group, test=test)
    rows = {k: budget_rows(b, k, seed=3) for k in (1, 4, 16, 0)}
    for small, large in ((1, 4), (4, 16), (16, 0)):
        assert set(rows[small][0]) < set(rows[large][0])
    for k, (r, probes) in rows.items():
        assert (group[r] < 1000).all() and not test[r].any()  # map capture, training probes
        assert sorted(set((group[r] % PROBES).tolist())) == probes
        assert len(probes) == (k or len(np.unique(group[~test & (group < 1000)] % PROBES)))
    with pytest.raises(ValueError):
        budget_rows(b, PROBES, seed=3)


def test_each_arm_tunes_its_own_set():
    (g1, g1_c), (nao, nao_c) = _body("g1"), _body("nao")
    table = stack_bodies([g1, nao], [g1_c, nao_c])
    one = table.take(torch.tensor([1]))
    counts = {}
    for arm in ("zero", "etune", "bitfit", "full"):
        model = PactPolicy(DIMS)
        tuned = adapt_parameters(model, "pact", arm, DIMS, one, 1)
        counts[arm] = sum(p.numel() for p in tuned)
        if arm == "etune":
            assert {id(p) for p in tuned} == {
                id(model.embodiment.delta_e), id(model.embodiment.delta_d)
            }  # fmt: skip
    assert counts["zero"] == 0 < counts["etune"] < counts["bitfit"] < counts["full"]
    hd = HptDims(
        bodies=("g1", "nao"), joints=(len(g1.joint_names), len(nao.joint_names)), history=2,
        width=32, heads=4, trunk_layers=1, stem_tokens=4, head_hidden=16,
    )  # fmt: skip
    hpt = HptPolicy(hd)
    tuned = {id(p) for p in adapt_parameters(hpt, "hpt", "stem", hd, one, 1)}
    own = {id(p) for p in [*hpt.proprio[1].parameters(), *hpt.heads[1].parameters()]}
    assert tuned == own
    with pytest.raises(ValueError):
        adapt_parameters(HptPolicy(hd), "hpt", "etune", hd, one, 1)


def test_the_adapted_checkpoint_drives_as_the_trained_model():
    """Trained on the padded table, saved on the body's own joints, loaded as the driver
    loads it (``PactDriver``'s residual branch): the same targets."""
    (g1, g1_c), (nao, nao_c) = _body("g1"), _body("nao")
    table = stack_bodies([g1, nao], [g1_c, nao_c])
    count, joints = table.mask.shape[1], len(g1.joint_names)
    assert count > joints  # the added body (the G1) is padded to the NAO's joints
    torch.manual_seed(0)
    model = PactPolicy(DIMS)
    adapt_parameters(model, "pact", "bitfit", DIMS, table.take(torch.tensor([0])), 0)
    with torch.no_grad():
        model.embodiment.delta_e.normal_()
        model.embodiment.delta_d.normal_()
    model.eval()
    x = _inputs(count, steps=4, seed=1)
    with torch.no_grad():
        _, trained = model(table.take(torch.zeros(3, dtype=torch.long)), **x)

    state, extra = driver_state(model, joints)
    loaded = PactPolicy(DIMS)
    loaded.load_state_dict(state)
    own = stack_bodies([g1], [g1_c])
    shifted = BodyResidual(loaded.embodiment, own, DIMS.body_tokens, DIMS.width)
    with torch.no_grad():
        shifted.delta_e.copy_(extra["residual"]["delta_e"])
        shifted.delta_d.copy_(extra["residual"]["delta_d"])
    loaded.embodiment = shifted
    loaded.eval()
    x_own = dict(x, joint_obs=x["joint_obs"][:, :joints])
    with torch.no_grad():
        _, driven = loaded(own.take(torch.zeros(3, dtype=torch.long)), **x_own)
    torch.testing.assert_close(driven, trained[:, :joints], rtol=1e-5, atol=1e-6)
    assert (trained[:, joints:] == 0).all()
