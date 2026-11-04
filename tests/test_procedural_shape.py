"""Checks on the family's stance-load profiles (``GaitRanges.load_shape``, skill_space.md §10.5).

A walking stance's load, the half-sine on a warped clock (``load_profile``), must keep the
half-sine's impulse, so that the biped's vertical dynamics -- and every gait it writes -- hold
for any warp; a warp away from one must lengthen the light contacts of load transfer; a run
must keep its half-sine; and the pre-registered probes, which name no warp, must stay
half-sines.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.procedural import (  # noqa: E402
    GaitRanges,
    PointMassBiped,
    feasible,
    load_profile,
    load_profile_peak,
    sample_parameters,
    standing_skill,
)

STEP = 0.07742576129865368
SHAPED = GaitRanges(load_shape=(0.5, 2.0))
LOAD = SKILL_LAYOUT.slice_of("foot_load")
VELOCITY = SKILL_LAYOUT.slice_of("com_velocity")


def _biped(n: int, ranges: GaitRanges, seed: int = 0, **kwargs) -> PointMassBiped:
    biped = PointMassBiped(
        n, ranges, body_step=STEP, substep=STEP / 4, horizon=13 * STEP, seed=seed, graph=False,
        **kwargs,
    )  # fmt: skip
    biped.set_offsets(torch.zeros(1, dtype=torch.float64))
    return biped


def _probe(biped: PointMassBiped, shapes: list[float | None], duty: float = 0.6) -> None:
    n = len(shapes)
    params = {
        "forward": torch.full((n,), 0.15),
        "lateral": torch.zeros(n),
        "turn": torch.zeros(n),
        "period": torch.full((n,), 1.2),
        "width": torch.full((n,), 0.4),
        "height": torch.full((n,), 0.94),
        "clearance": torch.full((n,), 0.05),
        "lean": torch.full((n,), 0.09),
        "duty": torch.full((n, 2), duty),
        "first": torch.tensor([i % 2 for i in range(n)]),
        "duration": torch.full((n,), 1e9),
        "offset": torch.full((n,), 0.5),
    }
    if any(s is not None for s in shapes):
        params["load_shape"] = torch.tensor([1.0 if s is None else s for s in shapes])
    ids = torch.arange(n)
    biped.set_probes(ids, params)
    biped.anchor(ids, standing_skill(n))


def _run(biped: PointMassBiped, substeps: int) -> np.ndarray:
    rows = []
    for _ in range(substeps):
        biped._step(biped.substep)
        rows.append(biped.record.clone())
    return torch.stack(rows).numpy()


def test_every_warp_keeps_the_half_sines_integral() -> None:
    s = torch.linspace(0.0, 1.0, 400001, dtype=torch.float64)
    for k in (0.5, 0.7, 1.0, 1.5, 2.0):
        warp = torch.tensor(k, dtype=torch.float64)
        integral = float(torch.trapezoid(load_profile(s, warp), s))
        assert integral == pytest.approx(2.0 / math.pi, rel=2e-3)
    # Above the floor at which the profile reads its limit, warp 1 is the half-sine.
    above = s >= 1e-4
    torch.testing.assert_close(
        load_profile(s, torch.tensor(1.0, dtype=torch.float64))[above],
        torch.sin(math.pi * s)[above].clamp(min=0.0),
    )
    assert float(load_profile_peak(torch.tensor(1.0))) == pytest.approx(1.0, abs=1e-3)
    # The load at touchdown is finite at the range's lower bound: pi/2 of the half-sine's peak.
    assert float(load_profile(torch.tensor(0.0), torch.tensor(0.5))) == pytest.approx(
        math.pi / 2, rel=1e-3
    )


def test_shaped_gaits_obey_newton_and_stay_finite() -> None:
    biped = _biped(64, SHAPED, seed=2)
    rows = _run(biped, 2400)
    assert np.isfinite(rows).all()
    live = ~biped.fallen.numpy()
    acc = np.diff(rows[..., VELOCITY.start + 2], axis=0) / biped.substep
    force = rows[1:, :, LOAD].sum(-1)
    # The vertical velocity is the heading frame's, the same in the world.
    assert np.abs(acc - (force - 1.0))[:, live].max() < 1e-3
    assert float(biped.fallen.float().mean()) <= 0.05


def test_a_warp_lengthens_the_light_contacts_and_keeps_the_impulse() -> None:
    shapes = [0.5, 1.0, 2.0]
    biped = _biped(12, SHAPED, shaped=True)
    _probe(biped, shapes * 4)
    rows = _run(biped, int(30 / biped.substep))[int(10 / biped.substep) :]
    load = rows[..., LOAD]
    light, mean = [], []
    for i in range(3):
        cols = load[:, i::3]
        light.append(float(((cols > 0.05) & (cols <= 0.25)).mean()))
        mean.append(float(cols.mean()))
    # Measured at duty 0.6: 0.097 (warp 0.5), 0.065 (the half-sine), 0.080 (warp 2).
    assert light[0] > 1.3 * light[1]
    assert light[2] > light[1]
    assert not bool(biped.fallen.any())
    # Half the weight per foot on average, whatever the profile.
    for value in mean:
        assert value == pytest.approx(0.5, abs=0.03)


def test_an_unshaped_profile_in_a_shaped_biped_is_the_half_sine() -> None:
    plain = _biped(8, GaitRanges(), shaped=False)
    shaped = _biped(8, GaitRanges(), shaped=True)
    _probe(plain, [None] * 8)
    _probe(shaped, [1.0] * 8)
    a, b = _run(plain, 1200), _run(shaped, 1200)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_probes_that_name_no_exponent_stay_half_sines() -> None:
    biped = _biped(6, SHAPED)
    _probe(biped, [None] * 6)
    _run(biped, 800)
    assert torch.equal(biped.exponent, torch.ones_like(biped.exponent))


def test_a_run_keeps_its_half_sine() -> None:
    biped = _biped(4, SHAPED, shaped=True)
    _probe(biped, [2.0] * 4, duty=0.35)
    _run(biped, int(12 / biped.substep))
    # Landed from the air, every stance carries the exponent the amplitude assumes.
    assert torch.equal(biped.exponent, torch.ones_like(biped.exponent))


def test_the_bank_spans_the_exponents_and_stays_feasible() -> None:
    g = torch.Generator().manual_seed(0)
    params = sample_parameters(4000, SHAPED, 0.12, g, "cpu")
    p = params["load_shape"]
    assert float(p.min()) >= 0.5 and float(p.max()) <= 2.0
    assert float(torch.log(p).mean()) == pytest.approx(0.0, abs=0.08)
    ok = feasible(
        params["period"], params["duty"], params["forward"], params["lateral"], params["width"],
        SHAPED, 0.12, p,
    )  # fmt: skip
    assert bool(ok.all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_shaped_biped_replays_as_a_graph_exactly_as_eagerly() -> None:
    def make(graph: bool) -> PointMassBiped:
        biped = PointMassBiped(
            64, SHAPED, body_step=STEP, substep=STEP / 4, horizon=13 * STEP, seed=9,
            device="cuda", graph=graph,
        )  # fmt: skip
        biped.set_offsets(torch.zeros(1, dtype=torch.float64))
        return biped

    a, b = make(True), make(False)
    ids = torch.arange(64, device="cuda")
    for _ in range(80):
        a.advance()
        b.advance()
    assert torch.equal(a.plan(ids)[2], b.plan(ids)[2])
    assert torch.equal(a.exponent, b.exponent)
