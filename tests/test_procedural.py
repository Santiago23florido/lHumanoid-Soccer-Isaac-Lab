"""Checks on the plans no task wrote: the point-mass biped of ``latent/procedural.py``.

The family is only worth training a tracker on if every plan in it is one a
body could follow, and if the gait asked for is the gait written. These tests
hold the biped to physics read off its own ``z*`` -- the vertical acceleration
is the load minus the weight, the centre of pressure the horizontal
acceleration implies stays under the feet, feet on the ground do not move and
feet in the air carry nothing -- and to its parameters: a gait drawn with a
period, duty and speed is measured, by the same gait signature every capture
is judged with, to have them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT, mirror  # noqa: E402
from humanoid_transfer.transfer.latent.procedural import (  # noqa: E402
    FOOT_HALF_LENGTH,
    FOOT_HALF_WIDTH,
    GaitRanges,
    PointMassBiped,
    feasible,
    sample_parameters,
    standing_skill,
)

STEP = 0.07742576129865368
"""The G1's control step in ``T``: the contract's grid."""

RANGES = GaitRanges()
VELOCITY = SKILL_LAYOUT.slice_of("com_velocity")
FEET = SKILL_LAYOUT.slice_of("foot_position")
FOOT_VELOCITY = SKILL_LAYOUT.slice_of("foot_velocity")
LOAD = SKILL_LAYOUT.slice_of("foot_load")
SKILL = SKILL_LAYOUT.size


def _biped(n: int, seed: int = 0, **kwargs) -> PointMassBiped:
    biped = PointMassBiped(
        n, RANGES, body_step=STEP, substep=STEP / 4, horizon=13 * STEP, seed=seed, **kwargs
    )
    biped.set_offsets(torch.zeros(1, dtype=torch.float64))
    return biped


def _probe(biped: PointMassBiped, gaits: list[dict]) -> None:
    n = len(gaits)
    keys = ("forward", "lateral", "turn", "period", "width", "height", "clearance", "lean")
    params = {key: torch.tensor([g[key] for g in gaits]) for key in keys}
    params["duty"] = torch.tensor([g["duty"] for g in gaits])
    params["first"] = torch.tensor([i % 2 for i in range(n)])
    params["duration"] = torch.full((n,), 1e9)
    params["offset"] = torch.full((n,), 0.5)
    ids = torch.arange(n)
    biped.set_probes(ids, params)
    biped.anchor(ids, standing_skill(n))


def _gait(forward, period, duty, lateral=0.0, turn=0.0, width=0.4) -> dict:
    return dict(
        forward=forward,
        lateral=lateral,
        turn=turn,
        period=period,
        duty=list(duty),
        width=width,
        height=0.94,
        clearance=0.05,
        lean=0.09,
    )


def _substeps(biped: PointMassBiped, count: int) -> np.ndarray:
    """``(count, n, 27 + 3)``: the biped's record after each substep."""
    rows = []
    for _ in range(count):
        biped._step(biped.substep)
        rows.append(biped.record.clone())
    return torch.stack(rows).numpy()


def _world(rows: np.ndarray) -> dict[str, np.ndarray]:
    """The record's heading-frame ``z*`` turned back into the biped's world."""
    yaw = rows[..., SKILL + 2]
    c, s = np.cos(yaw), np.sin(yaw)

    def turn(x, y):
        return c[..., None] * x - s[..., None] * y, s[..., None] * x + c[..., None] * y

    vel = rows[..., VELOCITY]
    vx, vy = turn(vel[..., :1], vel[..., 1:2])
    feet = rows[..., FEET].reshape(*rows.shape[:-1], 2, 3)
    fx, fy = turn(feet[..., 0], feet[..., 1])
    com = np.stack([rows[..., SKILL], rows[..., SKILL + 1], rows[..., 0]], axis=-1)
    foot = np.stack([fx, fy, feet[..., 2]], axis=-1) + com[..., None, :]
    return {
        "com": com,
        "vel": np.concatenate([vx, vy, vel[..., 2:3]], axis=-1),
        "foot": foot,
        "foot_vel": rows[..., FOOT_VELOCITY].reshape(*rows.shape[:-1], 2, 3),
        "load": rows[..., LOAD],
        "yaw": yaw,
    }


def test_sampled_gaits_are_feasible_and_span_the_ranges() -> None:
    g = torch.Generator().manual_seed(0)
    params = sample_parameters(4000, RANGES, 0.12, g, "cpu")
    ok = feasible(
        params["period"],
        params["duty"],
        params["forward"],
        params["lateral"],
        params["width"],
        RANGES,
        0.12,
    )
    assert bool(ok.all())
    for key in ("forward", "lateral", "turn", "width", "height", "clearance", "lean"):
        low, high = getattr(RANGES, key)
        values = params[key]
        assert float(values.min()) >= low and float(values.max()) <= high
        assert float(values.max() - values.min()) > 0.8 * (high - low), key
    # The period is a scale: log-uniform, so as many draws below the geometric middle as above.
    middle = math.sqrt(RANGES.period[0] * RANGES.period[1])
    share = float((params["period"] < middle).float().mean())
    assert 0.4 < share < 0.75  # feasibility removes some long, fast gaits
    assert float(params["duty"].min()) < 0.4 and float(params["duty"].max()) > 0.9


def test_vertical_acceleration_is_the_load_minus_the_weight() -> None:
    biped = _biped(48, seed=1)
    rows = _substeps(biped, 2400)
    world = _world(rows)
    live = ~biped.fallen.numpy()
    acc = np.diff(world["vel"][..., 2], axis=0) / biped.substep
    force = world["load"][1:].sum(-1)
    residual = np.abs(acc - (force - 1.0))[:, live]
    assert residual.max() < 1e-3


def test_the_centre_of_pressure_stays_under_the_feet() -> None:
    """The horizontal acceleration implies a CoP, ``p = c − c̈ c_z / F``: inside the support."""
    biped = _biped(48, seed=2)
    rows = _substeps(biped, 2400)
    world = _world(rows)
    live = ~biped.fallen.numpy()
    acc = np.diff(world["vel"][..., :2], axis=0) / biped.substep
    force = world["load"][1:].sum(-1)
    height = world["com"][:-1, :, 2]
    loaded = force > 0.05
    cop = world["com"][:-1, :, :2] - acc * (height / np.maximum(force, 1e-6))[..., None]
    feet = world["foot"][1:, ..., :2]
    share = world["load"][1:] / np.maximum(force, 1e-6)[..., None]
    a, b = feet[..., 0, :], feet[..., 1, :]
    # The nearest point of the support: the segment between the loaded feet, then each
    # sole's own area around it.
    u = b - a
    t = ((cop - a) * u).sum(-1) / np.maximum((u * u).sum(-1), 1e-9)
    single = share.max(-1) > 0.999
    t = np.where(single, share[..., 1], np.clip(t, 0.0, 1.0))
    nearest = a + t[..., None] * u
    gap = np.linalg.norm(cop - nearest, axis=-1)
    box = math.hypot(FOOT_HALF_LENGTH, FOOT_HALF_WIDTH) + 1e-3
    assert (gap[loaded[:, :] & live[None, :]] <= box).all()
    flying = (force < 1e-6) & live[None, :]
    # In the air nothing pushes: what is left is float32 rounding of rotated velocities.
    assert np.abs(acc[flying]).max(initial=0.0) < 1e-4


def test_feet_on_the_ground_hold_still_and_feet_in_the_air_carry_nothing() -> None:
    biped = _biped(48, seed=3)
    rows = _substeps(biped, 2000)
    world = _world(rows)
    live = ~biped.fallen.numpy()
    load = world["load"][:, live]
    foot = world["foot"][:, live]
    speed = np.linalg.norm(world["foot_vel"][:, live], axis=-1)
    assert load.min() >= 0.0
    down = load > 1e-6
    assert np.abs(foot[..., 2][down]).max() < 1e-5
    assert speed[down].max() < 1e-5
    up = foot[..., 2] > 1e-4
    assert load[up].max(initial=0.0) == 0.0


@pytest.mark.parametrize(
    "gait",
    [
        _gait(0.085, 1.87, (0.92, 0.86)),  # the walk teacher's signature
        _gait(0.19, 1.03, (0.52, 0.46)),  # quick alternating steps
        _gait(0.30, 1.00, (0.38, 0.38), width=0.3),  # a run, with flight
        _gait(0.05, 2.60, (0.85, 0.85)),  # slow, long double support
        _gait(0.12, 0.985, (0.52, 0.48), lateral=0.08, turn=0.08),
    ],
    ids=["walk", "quick", "run", "slow", "turning"],
)
def test_the_gait_measured_is_the_gait_asked_for(gait: dict) -> None:
    n = 4
    biped = _biped(n, seed=4)
    _probe(biped, [gait] * n)
    steps = int(70 / STEP)
    rows = []
    for _ in range(steps):
        biped.advance()
        rows.append(biped.plan(torch.arange(n))[2].numpy())
    z = np.stack(rows)[int(20 / STEP) :]
    done = np.zeros(z.shape[:2], dtype=bool)
    signature = gait_signature(z, done, STEP, threshold=0.01)
    assert not bool(biped.fallen.any())
    assert signature.stride_period == pytest.approx(gait["period"], rel=0.03)
    for measured, asked in zip(signature.duty_factor, gait["duty"], strict=True):
        assert measured == pytest.approx(asked, abs=0.03)
    velocity = z[..., VELOCITY].reshape(-1, 3).mean(0)
    assert velocity[0] == pytest.approx(gait["forward"], abs=0.1 * gait["forward"] + 0.005)
    assert velocity[1] == pytest.approx(gait["lateral"], abs=0.1 * abs(gait["lateral"]) + 0.005)
    height = z[..., 0].mean()
    assert height == pytest.approx(gait["height"], abs=0.01)


def test_a_symmetric_gait_is_its_own_mirror_image() -> None:
    n = 4
    biped = _biped(n, seed=5)
    _probe(biped, [_gait(0.15, 1.2, (0.7, 0.7))] * n)
    rows = []
    for _ in range(int(80 / STEP)):
        biped.advance()
        rows.append(biped.plan(torch.arange(n))[2].numpy())
    z = np.stack(rows)[int(20 / STEP) :]
    mean = z.reshape(-1, SKILL).mean(0)
    gap = np.abs(mean - mirror(mean))
    assert gap.max() < 0.01


def test_an_episode_starts_from_the_body_it_leads() -> None:
    biped = _biped(3, seed=6)
    skill = standing_skill(3)
    height = torch.tensor([0.90, 0.94, 0.98])
    skill[:, 0] = height
    skill[:, FEET.start + 1] = torch.tensor([0.15, 0.20, 0.25])
    skill[:, FEET.start + 2] = -height
    skill[:, FEET.start + 5] = -height
    biped.anchor(torch.arange(3), skill)
    now = biped.plan(torch.arange(3))[2]
    assert torch.allclose(now[:, 0], skill[:, 0], atol=1e-6)
    assert torch.allclose(now[:, FEET], skill[:, FEET], atol=1e-6)
    assert float(now[:, VELOCITY].abs().max()) == 0.0


def test_a_plans_integrals_are_the_motion_its_velocities_describe() -> None:
    n = 8
    biped = _biped(n, seed=7)
    _probe(biped, [_gait(0.2, 1.0, (0.6, 0.6), turn=0.15)] * n)
    biped.set_offsets(torch.arange(1, 14, dtype=torch.float64) * STEP)
    for _ in range(int(15 / STEP)):
        biped.advance()
    plan, integrals, now = biped.plan(torch.arange(n))
    # The velocities, turned by the plan's own heading and integrated by the trapezoid rule.
    velocity = torch.cat([now[:, None], plan], dim=1)[..., VELOCITY][..., :2]
    heading = torch.cat([torch.zeros(n, 1), integrals[..., 2]], dim=1)
    c, s = torch.cos(heading), torch.sin(heading)
    world = torch.stack(
        [c * velocity[..., 0] - s * velocity[..., 1], s * velocity[..., 0] + c * velocity[..., 1]],
        dim=-1,
    )
    rebuilt = torch.cumsum((world[:, 1:] + world[:, :-1]) / 2 * STEP, dim=1)
    error = (rebuilt - integrals[..., :2]).abs().max()
    assert float(error) < 0.01 * float(integrals[..., :2].abs().max()) + 1e-3


def test_schedules_rarely_need_rescuing_and_hardly_ever_fall() -> None:
    """Random schedules over 40 ``T``: the measured rates the documentation quotes, with margin."""
    n = 96
    biped = _biped(n, seed=8)
    for _ in range(int(40 / STEP)):
        biped.advance()
    assert bool(torch.isfinite(biped.record).all())
    assert float(biped.fallen.float().mean()) <= 0.05
    assert float((biped.mode == 2).float().mean()) > 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_captured_graph_steps_exactly_as_the_eager_code() -> None:
    a = _biped(64, seed=9, device="cuda", graph=True)
    b = _biped(64, seed=9, device="cuda", graph=False)
    ids = torch.arange(64, device="cuda")
    for _ in range(80):
        a.advance()
        b.advance()
    assert torch.equal(a.plan(ids)[2], b.plan(ids)[2])
