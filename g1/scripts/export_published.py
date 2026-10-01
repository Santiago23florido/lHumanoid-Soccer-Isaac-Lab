"""Fetch a G1 policy Isaac Lab publishes and export it as a TorchScript driver.

The transfer study may use pretrained models only from Isaac Lab's own G1
offering. Isaac Lab publishes rsl-rl checkpoints of its locomotion tasks on the
same server as its assets; this script downloads one, rebuilds the actor it
holds -- the shipped ``G1FlatPPORunnerCfg`` network, 123 -> 256 -> 128 -> 128
-> 37 with ELU, no observation normaliser -- and scripts it, so
``capture_rollouts.py --checkpoint`` can drive the robot with it like the
pinned teacher. The observation it expects is ``G1FlatEnvCfg``'s policy group,
the same 123 numbers the pinned teacher reads.

Outputs go under ``logs/published/<task>/`` (untracked): the checkpoint as
downloaded, ``policy.pt`` and a ``manifest.json`` with both hashes and the URL.

Usage::

    python g1/scripts/export_published.py --task Isaac-Velocity-Flat-G1-v0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
"""Isaac Sim 5.1's asset root: the version this repository runs."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_from(state: dict) -> nn.Sequential:
    """The rsl-rl MLP actor in ``state`` (``actor.<i>.weight``), ELU between layers."""
    layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("actor.")})
    modules: list[nn.Module] = []
    for position, index in enumerate(layers):
        weight = state[f"actor.{index}.weight"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(state[f"actor.{index}.bias"])
        modules.append(linear)
        if position < len(layers) - 1:
            modules.append(nn.ELU())
    return nn.Sequential(*modules)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0")
    args = parser.parse_args()

    out = REPO_ROOT / "logs" / "published" / args.task
    out.mkdir(parents=True, exist_ok=True)
    url = f"{ASSET_ROOT}/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/{args.task}/checkpoint.pt"
    checkpoint = out / "checkpoint.pt"
    if not checkpoint.exists():
        urllib.request.urlretrieve(url, checkpoint)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["model_state_dict"]
    if any("normalizer" in key for key in state):
        raise SystemExit("this checkpoint carries an observation normaliser; export it with it")
    actor = actor_from(state).eval()
    scripted = torch.jit.script(actor)
    policy = out / "policy.pt"
    scripted.save(str(policy))

    probe = torch.randn(64, actor[0].in_features)
    with torch.no_grad():
        difference = float((actor(probe) - torch.jit.load(str(policy))(probe)).abs().max())
    manifest = {
        "task": args.task,
        "url": url,
        "checkpoint_sha256": _sha256(checkpoint),
        "policy_sha256": _sha256(policy),
        "iteration": int(saved.get("iter", -1)),
        "observation_dim": actor[0].in_features,
        "action_dim": actor[-1].out_features,
        "parameter_count": sum(p.numel() for p in actor.parameters()),
        "export_max_abs_difference": difference,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
