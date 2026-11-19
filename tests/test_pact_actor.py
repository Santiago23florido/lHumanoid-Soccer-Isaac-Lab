"""PACT as an rsl-rl actor: the same actions as the driver that runs G-D1, from inputs built
the way the observation manager builds them, and only the chosen parameters tuned."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from humanoid_transfer.transfer.latent.pact.actor import (  # noqa: E402
    PactActor,
    imu_features,
    inputs_width,
    joint_features,
)
from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.descriptors import (  # noqa: E402
    GRAVITY,
    contact_joints,
    describe,
)
from humanoid_transfer.transfer.latent.pact.driver import PactDriver, anchor_inputs  # noqa: E402
from humanoid_transfer.transfer.latent.pact.model import PactDims  # noqa: E402
from humanoid_transfer.transfer.latent.pact.policy import PactPolicy  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO / "transfer" / "bodies" / "bodies.json").read_text("utf-8"))
DIMS = PactDims(
    width=32, heads=4, history=3, free_tokens=2, body_tokens=2, observation=3,
    anchor_inputs=True, timed_plan=True,
)  # fmt: skip
VIEW, DT, SCALE = 4, 0.02, 0.25
ACTION_JOINTS = [5, 2, 7, 11]


class History:
    """The observation manager's per-term history: oldest first; the first push after a
    reset fills every slot (``isaaclab.utils.buffers.CircularBuffer``)."""

    def __init__(self, length: int) -> None:
        self.length, self.buffer = length, None

    def append(self, x: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
        if self.buffer is None:
            self.buffer = x[:, None].repeat(1, self.length, 1)
        else:
            self.buffer = torch.cat([self.buffer[:, 1:], x[:, None]], dim=1)
            self.buffer[fresh] = x[fresh][:, None].expand(-1, self.length, -1)
        return self.buffer.flatten(1)


def _card(name: str) -> dict:
    return json.loads((REPO / REGISTRY[name]["card"]).read_text("utf-8"))


def _state(card: dict, n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    j = len(card["joints"]["names"])
    q0 = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32)
    q = q0 + 0.1 * torch.randn(n, j, generator=gen)
    qd = torch.randn(n, j, generator=gen)
    g = torch.tensor([0.0, 0.0, -1.0]) + 0.1 * torch.randn(n, 3, generator=gen)
    g = g / g.norm(dim=-1, keepdim=True)
    omega = 0.5 * torch.randn(n, 3, generator=gen)
    target = q + 0.05 * torch.randn(n, j, generator=gen)
    return q, qd, g, omega, target


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    card = _card("nao")
    unit = REGISTRY["nao"].get("leg_length_unit")
    descriptor = describe(card, leg_length=unit)
    anchors = KinematicAnchors(card, unit)
    drive = DriveLoads(card, anchors)
    torch.manual_seed(0)
    policy = PactPolicy(DIMS).eval()
    gen = torch.Generator().manual_seed(2)
    checkpoint = {
        "state": policy.state_dict(),
        "dims": vars(DIMS),
        "z_mean": torch.randn(27, generator=gen).numpy(),
        "z_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
        "a_mean": torch.randn(27, generator=gen).numpy(),
        "a_std": (torch.rand(27, generator=gen) + 0.5).numpy(),
    }
    path = tmp_path_factory.mktemp("pact") / "model.pt"
    torch.save(checkpoint, path)
    skill_mean, skill_std = torch.randn(27, generator=gen), torch.rand(27, generator=gen) + 0.5
    q0 = torch.as_tensor(card["joints"]["default_pos"], dtype=torch.float32)
    return dict(
        card=card, descriptor=descriptor, anchors=anchors, drive=drive, checkpoint=checkpoint,
        path=path, skill_mean=skill_mean, skill_std=skill_std, q0=q0,
        contacts=contact_joints(card),
    )  # fmt: skip


def _actor(s, n: int, arm: str = "etune") -> PactActor:
    joints = len(s["descriptor"].joint_names)
    obs = {
        "policy": torch.zeros(n, 70 + VIEW * 30 + 27),
        "pact": torch.zeros(n, inputs_width(joints, DIMS.history)),
    }
    return PactActor(
        obs, {"actor": ["policy", "pact"]}, "actor", len(ACTION_JOINTS),
        checkpoint=str(s["path"]), repo=str(REPO), body="nao", view=VIEW,
        skill_mean=s["skill_mean"].tolist(), skill_std=s["skill_std"].tolist(), control_dt=DT,
        action_joints=ACTION_JOINTS, action_scale=SCALE, arm=arm,
    )  # fmt: skip


def test_the_actor_acts_as_the_driver_through_a_reset(setup):
    s, n = setup, 3
    driver = PactDriver(
        s["checkpoint"], s["descriptor"], s["contacts"], s["anchors"], s["drive"], s["q0"],
        ACTION_JOINTS, SCALE, VIEW, s["skill_mean"], s["skill_std"], DT, n, "cpu",
    )  # fmt: skip
    actor = _actor(s, n).eval()
    time_unit = float(np.sqrt(s["descriptor"].leg_length / GRAVITY))
    joints, imu = History(DIMS.history), History(DIMS.history)
    gen = torch.Generator().manual_seed(3)
    fresh = torch.zeros(n, dtype=torch.bool)
    for k in range(6):
        q, qd, g, w, t = _state(s["card"], n, seed=10 + k)
        policy_obs = torch.randn(n, 70 + VIEW * 30 + 27, generator=gen)
        expected = driver(policy_obs, q, qd, t, g, w)
        with torch.no_grad():
            flat = torch.cat(
                [
                    joints.append(joint_features(q, qd, t, s["q0"], time_unit), fresh),
                    imu.append(imu_features(g, w, time_unit), fresh),
                    anchor_inputs(s["anchors"], s["drive"], q, qd, g, w, t),
                ],
                dim=1,
            )
            got = actor({"policy": policy_obs, "pact": flat})
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        fresh = torch.zeros(n, dtype=torch.bool)
        if k == 2:  # env 1 ends its episode: both refill on the next reading
            driver.reset(torch.tensor([1]))
            fresh[1] = True


@pytest.mark.parametrize("arm", ["zero", "etune", "bitfit", "full"])
def test_each_arm_tunes_only_its_parameters(setup, arm):
    actor = _actor(setup, 2, arm)
    names = {n for n, p in actor.named_parameters() if p.requires_grad}
    residual = {"policy.embodiment.delta_e", "policy.embodiment.delta_d"}
    assert "distribution.std_param" in names
    if arm == "zero":
        assert names == {"distribution.std_param"}
    elif arm == "etune":
        assert names == residual | {"distribution.std_param"}
    elif arm == "bitfit":
        norms = {
            f"{m}.{p}" if m else p
            for m, module in actor.named_modules()
            if isinstance(module, torch.nn.LayerNorm)
            for p, _ in module.named_parameters(recurse=False)
        }
        biases = {n for n, _ in actor.named_parameters() if n.endswith("bias")}
        assert names == residual | norms | biases | {"distribution.std_param"}
        assert norms and biases
    else:
        assert names == {n for n, _ in actor.named_parameters()}


def test_ppo_gradients_reach_the_residuals(setup):
    s, n = setup, 4
    actor = _actor(s, n, "etune")
    joints = len(s["descriptor"].joint_names)
    gen = torch.Generator().manual_seed(5)
    obs = {
        "policy": torch.randn(n, 70 + VIEW * 30 + 27, generator=gen),
        "pact": torch.randn(n, inputs_width(joints, DIMS.history), generator=gen),
    }
    actions = actor(obs, stochastic_output=True).detach()
    actor(obs, stochastic_output=True)
    actor.get_output_log_prob(actions).sum().backward()
    for p in actor.tuned:
        assert p.grad is not None and p.grad.abs().sum() > 0
    frozen = [p for p in actor.parameters() if not p.requires_grad]
    assert frozen and all(p.grad is None for p in frozen)


def test_the_residual_moves_the_action_and_the_checkpoint_keeps_the_policy_layout(setup):
    s, n = setup, 2
    actor = _actor(s, n).eval()
    joints = len(s["descriptor"].joint_names)
    gen = torch.Generator().manual_seed(6)
    obs = {
        "policy": torch.randn(n, 70 + VIEW * 30 + 27, generator=gen),
        "pact": torch.randn(n, inputs_width(joints, DIMS.history), generator=gen),
    }
    with torch.no_grad():
        before = actor(obs)
        # A shift equal on every feature is LayerNorm's null direction; a random one is not.
        actor.policy.embodiment.delta_e.add_(
            torch.randn(DIMS.body_tokens, DIMS.width, generator=gen)
        )
        after = actor(obs)
    assert (before - after).abs().max() > 1e-4
    saved = actor.adapted_checkpoint(s["checkpoint"])
    fresh = PactPolicy(DIMS)
    fresh.load_state_dict(saved["state"])  # strict: the residual wrapper is unwrapped
    assert torch.equal(saved["residual"]["delta_e"], actor.policy.embodiment.delta_e.detach())
    assert saved["residual"]["delta_d"].shape == (joints, DIMS.width)
