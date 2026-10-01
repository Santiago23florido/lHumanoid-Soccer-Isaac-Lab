"""Checks on the shared skill's memory: its clock, its windows and what it may see.

The memory is where the gait's timing is supposed to live, and also where a
body could leak back into the shared space. The tests below pin both down on
the wiring rather than on training outcomes: windows never reach into another
episode, the grid means the same body time on every body, and the memory's
only input is the physical history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from humanoid_transfer.transfer.latent.limbs import LimbDecoder, LimbLayout, g1_limbs  # noqa: E402
from humanoid_transfer.transfer.latent.memory import (  # noqa: E402
    MemoryDims,
    MemoryPolicy,
    MemorySkill,
    StateEncoder,
    TimeGrid,
    anchor_loss,
    skill_losses,
)
from humanoid_transfer.transfer.latent.model import (  # noqa: E402
    G1_OBSERVATION,
    BodyDecoder,
    LatentDims,
)
from humanoid_transfer.transfer.latent.probes import linear_probe, mlp_probe  # noqa: E402
from humanoid_transfer.transfer.latent.sequences import load_streams  # noqa: E402

G1_DT_STAR = 0.02 / np.sqrt(0.65457 / 9.81)
NAO_DT_STAR = 0.02 / np.sqrt(0.24801 / 9.81)


def _capture(path: Path, done: np.ndarray, seed: int = 0) -> Path:
    """A minimal rollout archive with the keys :func:`load_streams` reads."""
    steps, envs = done.shape
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        leg_length=0.65457,
        total_mass=32.2389,
        done=done,
        terminated=np.zeros_like(done),
        obs=rng.normal(size=(steps, envs, 123)).astype(np.float32),
        skill_state=np.zeros((steps, envs, 27), dtype=np.float32),
        command=rng.normal(size=(steps, envs, 3)).astype(np.float32),
        action_teacher=rng.normal(size=(steps, envs, 37)).astype(np.float32),
    )
    return path


# --- the clock ----------------------------------------------------------------


def test_the_grid_is_the_identity_on_the_body_it_was_built_for() -> None:
    grid = TimeGrid(step=G1_DT_STAR, samples=26)
    assert grid.history_steps(G1_DT_STAR) == 26
    np.testing.assert_allclose(grid.interpolation(G1_DT_STAR), np.eye(26))


def test_another_body_resamples_its_history_onto_the_same_instants() -> None:
    grid = TimeGrid(step=G1_DT_STAR, samples=26)
    matrix = grid.interpolation(NAO_DT_STAR)
    steps = grid.history_steps(NAO_DT_STAR)
    assert (steps - 1) * NAO_DT_STAR >= grid.span
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    # A signal linear in body time is reproduced exactly at the grid instants.
    history_time = (np.arange(steps) - (steps - 1)) * NAO_DT_STAR
    grid_time = (np.arange(26) - 25) * G1_DT_STAR
    signal = 3.0 * history_time - 0.5
    np.testing.assert_allclose(matrix @ signal, 3.0 * grid_time - 0.5, atol=1e-12)


def test_the_grid_spans_the_same_body_time_on_both_bodies() -> None:
    grid = TimeGrid(step=G1_DT_STAR, samples=26)
    for dt_star in (G1_DT_STAR, NAO_DT_STAR):
        matrix = grid.interpolation(dt_star)
        steps = grid.history_steps(dt_star)
        history_time = (np.arange(steps) - (steps - 1)) * dt_star
        np.testing.assert_allclose(matrix @ history_time, (np.arange(26) - 25) * G1_DT_STAR)


# --- the windows --------------------------------------------------------------


@pytest.fixture
def streams(tmp_path: Path):
    done = np.zeros((12, 2), dtype=bool)
    done[4, 0] = True  # env 0: episodes [0..4], [5..11]
    done[9, 1] = True  # env 1: episodes [0..9], [10..11]
    return load_streams([_capture(tmp_path / "a.npz", done)])


def test_columns_are_laid_end_to_end_with_their_episode_bounds(streams) -> None:
    assert len(streams) == 24
    np.testing.assert_array_equal(streams.start[:12].numpy(), [1] * 5 + [6] * 7)
    np.testing.assert_array_equal(streams.end[:12].numpy(), [4] * 5 + [11] * 7)
    np.testing.assert_array_equal(streams.start[12:].numpy(), [13] * 10 + [23] * 2)
    np.testing.assert_array_equal(streams.end[12:].numpy(), [21] * 10 + [23] * 2)


def test_no_window_crosses_a_reset_and_the_start_is_repeated(streams) -> None:
    windows = streams.windows(torch.tensor([7, 3]), length=4)
    np.testing.assert_array_equal(windows.numpy(), [[6, 6, 6, 7], [1, 1, 2, 3]])


def test_samples_exclude_reset_steps_and_keep_their_successors_in_the_episode(streams) -> None:
    samples = streams.samples(future=2)
    np.testing.assert_array_equal(samples.numpy(), [1, 2, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19])
    futures = streams.futures(samples, 2)
    assert bool((futures <= streams.end[samples][:, None]).all())
    assert bool((streams.start[futures] == streams.start[samples][:, None]).all())


def test_captures_are_concatenated_without_sharing_an_episode(tmp_path: Path) -> None:
    done = np.zeros((5, 1), dtype=bool)
    joined = load_streams(
        [_capture(tmp_path / "a.npz", done), _capture(tmp_path / "b.npz", done, seed=1)]
    )
    np.testing.assert_array_equal(joined.start.numpy(), [1] * 5 + [6] * 5)
    np.testing.assert_array_equal(joined.capture.numpy(), [0] * 5 + [1] * 5)
    windows = joined.windows(torch.tensor([6]), length=3)
    np.testing.assert_array_equal(windows.numpy(), [[6, 6, 6]])


# --- the probes ---------------------------------------------------------------


def test_the_probes_find_what_is_there_and_the_mlp_finds_more() -> None:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(6000, 4, generator=generator)
    targets = torch.stack([x @ torch.tensor([1.0, -2.0, 0.5, 0.0]), x[:, 0] ** 2], dim=1)
    steps = torch.arange(6000)
    train, test = steps[:5000], steps[5000:]

    def features(idx: torch.Tensor) -> torch.Tensor:
        return x[idx]

    linear = linear_probe(features, targets, train, test)
    nonlinear = mlp_probe(features, targets, train, test, epochs=40, hidden=64, batch=500)
    assert linear[0] > 0.999
    assert linear[1] < 0.05
    assert nonlinear[1] > 0.9


# --- what the memory may see -------------------------------------------------

DIMS = MemoryDims(hidden=32, channels=8)
GRID = TimeGrid(step=G1_DT_STAR, samples=26)


class _Echo(torch.nn.Module):
    """A decoder that hands back what it was given: the latent action, then the context."""

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.cat([action, context], dim=-1)


class _BaseOnly(torch.nn.Module):
    """An encoder that reads only the base terms -- so the joints cannot reach ẑ*."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Linear(9, 27)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state[:, :9])


def _policy(encoder, decoder, memory=None) -> MemoryPolicy:
    torch.manual_seed(0)
    return MemoryPolicy(
        encoder,
        MemorySkill(DIMS, GRID),
        decoder,
        G1_OBSERVATION,
        velocity_unit=2.53,
        time_unit=0.258,
        dt_star=G1_DT_STAR,
        memory=memory,
    ).eval()


def test_the_shared_skill_sees_the_body_only_through_its_physical_state() -> None:
    policy = _policy(_BaseOnly(), _Echo())
    obs = torch.randn(30, 4, 123)
    moved = obs.clone()
    moved[..., 12:123] += torch.randn(30, 4, 111)  # the joints and the last action, only
    with torch.no_grad():
        first = torch.stack([policy(o) for o in obs])
        policy.history = torch.zeros(0, policy.history_steps, 27)
        second = torch.stack([policy(o) for o in moved])
    assert torch.allclose(first[..., :16], second[..., :16])  # u: unchanged
    assert not torch.allclose(first[..., 16:], second[..., 16:])  # the decoder's context: moved


def test_no_imitation_gradient_reaches_the_encoder() -> None:
    torch.manual_seed(0)
    encoder = StateEncoder(120, 27, 32)
    skill = MemorySkill(DIMS, GRID)
    decoder = BodyDecoder(74, 37, LatentDims(action=16, hidden=32))
    streams = _random_streams(steps=60)
    steps = streams.samples(future=3)
    losses = skill_losses(skill, decoder, G1_OBSERVATION, streams, steps, torch.ones(27), 3, 0.5)
    losses["total"].backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is not None for p in skill.memory.parameters())

    skill.zero_grad(set_to_none=True)
    anchor_loss(encoder, skill, G1_OBSERVATION, streams, steps).backward()
    assert all(p.grad is None for p in skill.parameters())
    assert all(p.grad is not None for p in encoder.parameters())


def _random_streams(steps: int, envs: int = 2):
    from humanoid_transfer.transfer.latent.sequences import Streams

    generator = torch.Generator().manual_seed(1)
    total = steps * envs
    start = torch.repeat_interleave(torch.arange(envs) * steps + 1, steps)
    return Streams(
        obs=torch.randn(total, 123, generator=generator),
        skill=torch.randn(total, 27, generator=generator),
        command=torch.randn(total, 3, generator=generator),
        action=torch.randn(total, 37, generator=generator),
        falling=torch.zeros(total, dtype=torch.bool),
        start=start,
        end=start + steps - 2,
        capture=torch.zeros(total, dtype=torch.long),
        env=torch.repeat_interleave(torch.arange(envs), steps),
    )


def test_the_exported_policy_keeps_its_history_and_forgets_it_at_a_reset() -> None:
    torch.manual_seed(0)
    encoder = StateEncoder(120, 27, 32)
    decoder = BodyDecoder(74, 37, LatentDims(action=16, hidden=32))
    policy = _policy(encoder, decoder)
    scripted = torch.jit.script(_policy(encoder, decoder))

    obs = torch.randn(40, 3, 123)
    done = torch.zeros(40, 3, dtype=torch.bool)
    done[17, 1] = True
    with torch.no_grad():
        outputs = []
        for t in range(40):
            outputs.append(scripted(obs[t]))
            scripted.reset(done[t])
        outputs = torch.stack(outputs)

        # The same, from explicit windows: history since the last reset, start repeated.
        estimate = encoder(G1_OBSERVATION.encoder_input(obs))
        command = obs[..., 9:12] * policy.command_scale
        for env in range(3):
            begin = 0
            for t in range(40):
                index = torch.clamp(torch.arange(t - 25, t + 1), min=begin)
                memory = policy.skill.remember(estimate[index, env][None])
                action = policy.skill.act(estimate[t, env][None], memory, command[t, env][None])
                expected = decoder(action, obs[t, env, 12:86][None])
                assert torch.allclose(outputs[t, env], expected[0], atol=1e-5), (t, env)
                if done[t, env]:
                    begin = t + 1


def test_the_frozen_memory_policy_ignores_the_history() -> None:
    torch.manual_seed(0)
    encoder = StateEncoder(120, 27, 32)
    decoder = BodyDecoder(74, 37, LatentDims(action=16, hidden=32))
    fixed = torch.randn(16)
    policy = torch.jit.script(_policy(encoder, decoder, memory=fixed))
    obs = torch.randn(30, 2, 123)
    with torch.no_grad():
        outputs = torch.stack([policy(o) for o in obs])
        reference = _policy(encoder, decoder)
        command = obs[-1, :, 9:12] * reference.command_scale
        estimate = encoder(G1_OBSERVATION.encoder_input(obs[-1]))
        action = reference.skill.act(estimate, fixed.expand(2, -1), command)
        expected = decoder(action, obs[-1, :, 12:86])
    assert torch.allclose(outputs[-1], expected, atol=1e-5)


# --- a decoder per limb ------------------------------------------------------

G1_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint",
    "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint",
    "left_four_joint", "left_one_joint", "right_six_joint", "right_four_joint",
    "right_one_joint", "left_two_joint", "right_two_joint",
]  # fmt: skip
"""The teacher's joint order, as the captures record it."""


def test_every_g1_joint_belongs_to_exactly_one_limb() -> None:
    layout = g1_limbs(G1_JOINTS)
    joints = sorted(j for limb in layout.limbs for j in limb.joints)
    assert joints == list(range(37))
    sizes = {limb.name: len(limb.joints) for limb in layout.limbs}
    assert sizes == {"trunk": 1, "left_leg": 6, "right_leg": 6, "left_arm": 12, "right_arm": 12}
    assert layout.action == 16


def test_a_joint_in_two_limbs_or_in_none_is_refused() -> None:
    with pytest.raises(ValueError, match="matches"):
        LimbLayout.from_names(["a_joint"], {"x": ("a_.*", 1), "y": (".*joint", 1)}, 1)
    with pytest.raises(ValueError, match="no limb"):
        LimbLayout.from_names(["a_joint", "b_joint"], {"x": ("a_.*", 1)}, 1)


def test_a_limbs_share_of_the_latent_action_moves_only_that_limb() -> None:
    torch.manual_seed(0)
    layout = g1_limbs(G1_JOINTS)
    decoder = torch.jit.script(LimbDecoder(layout, hidden=16))
    action, context = torch.randn(5, 16), torch.randn(5, 74)
    left_leg = next(limb for limb in layout.limbs if limb.name == "left_leg")
    share = layout.shared + 2  # after u_0 and the trunk's two
    moved = action.clone()
    moved[:, share : share + left_leg.action] += 1.0
    with torch.no_grad():
        change = (decoder(moved, context) - decoder(action, context)).abs().amax(dim=0)
    changed = set(torch.nonzero(change > 1e-6).flatten().tolist())
    assert changed and changed <= set(left_leg.joints)


def test_a_limb_reads_only_its_own_joints() -> None:
    torch.manual_seed(0)
    layout = g1_limbs(G1_JOINTS)
    decoder = LimbDecoder(layout, hidden=16)
    action, context = torch.randn(5, 16), torch.randn(5, 74)
    right_arm = next(limb for limb in layout.limbs if limb.name == "right_arm")
    others = [j for j in range(37) if j not in right_arm.joints]
    moved = context.clone()
    moved[:, right_arm.joints] += 1.0
    moved[:, [37 + j for j in right_arm.joints]] -= 1.0
    with torch.no_grad():
        change = (decoder(action, moved) - decoder(action, context)).abs()
    assert float(change[:, others].max()) < 1e-6
    assert float(change[:, list(right_arm.joints)].max()) > 1e-6
