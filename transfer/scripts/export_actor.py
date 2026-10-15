"""Export an rsl-rl 5 actor checkpoint as a TorchScript driver, without a simulator.

Isaac Lab exports a policy from its ``play.py``, which keeps the simulator
running until it is closed. A tracker's checkpoint holds everything the
deterministic actor needs -- the running observation normaliser and the MLP --
so it is rebuilt here: ``(obs - mean) / (std + 0.01)`` (rsl-rl's
``EmpiricalNormalization``), then the MLP with ELU between layers, returning
the action mean. The result drives ``capture_rollouts.py --driver-policy``.

Usage::

    python transfer/scripts/export_actor.py logs/rsl_rl/g1_track/<run>/model_1499.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

_EPS = 1e-2
"""rsl-rl's ``EmpiricalNormalization`` default."""


class Actor(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, mlp: nn.Sequential) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("scale", std + _EPS)
        self.mlp = mlp

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.mlp((observation - self.mean) / self.scale)


def actor_from(state: dict) -> Actor:
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.")})
    modules: list[nn.Module] = []
    for position, index in enumerate(layers):
        weight = state[f"mlp.{index}.weight"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(state[f"mlp.{index}.bias"])
        modules.append(linear)
        if position < len(layers) - 1:
            modules.append(nn.ELU())
    if "obs_normalizer._mean" in state:
        mean = state["obs_normalizer._mean"].reshape(-1)
        std = state["obs_normalizer._std"].reshape(-1)
    else:
        width = modules[0].in_features
        mean, std = torch.zeros(width), torch.full((width,), 1.0 - _EPS)
    return Actor(mean, std, nn.Sequential(*modules)).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = actor_from(saved["actor_state_dict"])
    output = args.output or args.checkpoint.parent / "exported" / f"{args.checkpoint.stem}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(actor).save(str(output))
    probe = torch.randn(16, actor.mean.shape[0])
    with torch.no_grad():
        difference = float((actor(probe) - torch.jit.load(str(output))(probe)).abs().max())
    print(f"exported {output} (iteration {saved.get('iter')}, scripted vs eager {difference:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
