"""Widen a trained rsl_rl actor-critic for observations appended at the end of its groups.

A new observation term appended last to a group adds columns to the first layer
and entries to the input normaliser. Zero weights on the new columns, with mean
0 and variance 1 in the normaliser, leave the network computing exactly what it
computed -- the new inputs are ignored until training finds a use for them.
Adam's moments are dropped (their shapes changed) and restart; its learning rate
is kept.

Usage::

    python transfer/scripts/widen_inputs.py logs/rsl_rl/g1_track/<run>/model_1000.pt \
        --actor 24 --critic 24 --output logs/rsl_rl/g1_track/<new run>/model_1000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def widen(state: dict, extra: int) -> dict:
    state = dict(state)
    for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        name = f"obs_normalizer.{key}"
        old = state[name]
        state[name] = torch.cat([old, torch.full((*old.shape[:-1], extra), fill)], dim=-1)
    weight = state["mlp.0.weight"]
    state["mlp.0.weight"] = torch.cat([weight, weight.new_zeros(weight.shape[0], extra)], dim=1)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--actor", type=int, required=True, help="Inputs appended to the actor.")
    parser.add_argument("--critic", type=int, required=True, help="Inputs appended to the critic.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    before = saved["actor_state_dict"]["mlp.0.weight"].shape[1]
    saved["actor_state_dict"] = widen(saved["actor_state_dict"], args.actor)
    saved["critic_state_dict"] = widen(saved["critic_state_dict"], args.critic)
    saved["optimizer_state_dict"] = {
        "state": {},
        "param_groups": saved["optimizer_state_dict"]["param_groups"],
    }

    # The widened actor, fed the old inputs and anything at all in the new ones,
    # must act as the old one did.
    old_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old, new = old_state["actor_state_dict"], saved["actor_state_dict"]
    x = torch.randn(64, before)
    wide = torch.cat([x, torch.randn(64, args.actor)], dim=-1)

    def first_layer(state: dict, inputs: torch.Tensor) -> torch.Tensor:
        mean, std = state["obs_normalizer._mean"], state["obs_normalizer._std"]
        normalised = (inputs - mean) / (std + 1e-2)
        return normalised @ state["mlp.0.weight"].T + state["mlp.0.bias"]

    error = float((first_layer(old, x) - first_layer(new, wide)).abs().max())
    if error > 1e-5:
        raise RuntimeError(f"the widened actor computes something else ({error})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    print(
        f"{args.output}: actor {before} -> {before + args.actor} inputs, critic +{args.critic}, "
        f"iteration {saved['iter']} (first layer unchanged to {error:.1e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
