"""A noise-conditional contract generator: it reads the level, and it saves and loads with it."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.latent.contract import (  # noqa: E402
    Contract,
    ContractClock,
    ContractDims,
    ContractGenerator,
    contract_loss,
    load_contract,
    save_contract,
)

CLOCK = ContractClock(step=0.0774, history=26, horizon=20, commit=13, view=8)


def _generator(noise_conditional: bool) -> ContractGenerator:
    torch.manual_seed(0)
    dims = ContractDims(
        memory=8, channels=8, flow_hidden=32, flow_steps=2, noise_conditional=noise_conditional
    )
    return ContractGenerator(dims, CLOCK).eval()


def test_a_plain_generator_is_unchanged_by_the_option() -> None:
    generator = _generator(False)
    assert not generator.noise_conditional
    assert "noise_level" not in generator.state_dict()


def test_the_level_is_read_and_defaults_to_the_deployed_one() -> None:
    generator = _generator(True)
    window = torch.randn(4, CLOCK.history, 27)
    command = torch.randn(4, 3)
    noise = torch.randn(4, generator.noise_size)
    with torch.no_grad():
        default = generator.sample(window, command, noise)
        zero = generator.sample(window, command, noise, torch.zeros(4, 1))
        high = generator.sample(window, command, noise, torch.full((4, 1), 3.0))
    torch.testing.assert_close(default, zero)
    assert not torch.allclose(default, high)
    future = torch.randn(4, CLOCK.horizon, 27)
    loss = contract_loss(generator, window, command, future, 0.1, torch.zeros(4, 1))
    assert torch.isfinite(loss["total"])


def test_a_noise_conditional_contract_saves_and_loads(tmp_path: Path) -> None:
    generator = _generator(True)
    generator.noise_level.fill_(0.5)
    dims = ContractDims(memory=8, channels=8, flow_hidden=32, flow_steps=2, noise_conditional=True)
    path = tmp_path / "generator.pt"
    save_contract(path, Contract(generator, None, torch.ones(27)), dims, CLOCK, encoder_input=120)
    loaded = load_contract(path)
    assert loaded.generator.noise_conditional
    assert float(loaded.generator.noise_level) == 0.5
