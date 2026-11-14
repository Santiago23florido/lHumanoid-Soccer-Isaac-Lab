"""A deployment bundle for a training checkpoint, without Isaac Lab: for sim2sim during training.

Takes an rsl-rl checkpoint (``model_<i>.pt``) and an existing bundle
(``nao/scripts/export_runtime.py``) for everything that does not depend on the
tracker -- the body model, the plans, the contract's units -- and writes a new
bundle whose actor is the checkpoint's: its normaliser and MLP, symmetrised with
the body's measured mirror exactly as ``transfer/scripts/symmetrize_actor.py``
does, over the checkpoint's own input layout (``--history`` steps of
proprioception and actions appended, as ``NaoTrack-Sim2Real-v0`` has them).
Checked against the source bundle's actor when the layouts agree (``--check``).

Usage::

    python nao/scripts/checkpoint_bundle.py logs/rsl_rl/nao_track/<run>/model_3798.pt \
        --like nao/bundles/r1_3498_sym --history 3 --output nao/bundles/<name>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer.symmetry import (  # noqa: E402
    joint_mirror,
    joint_offsets,
    observation_mirror,
)

MIRROR = ROOT / "nao" / "body" / "mirror.json"
HISTORY_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions")
EPSILON = 1e-2
"""rsl-rl's empirical normaliser divides by its standard deviation plus this."""


def mirror_arrays(config: dict, terms: list[tuple[str, int]]) -> dict[str, np.ndarray]:
    """The Reynolds average's maps for the tracker's inputs, in the tracker's own units."""
    spec = json.loads(MIRROR.read_text(encoding="utf-8"))
    names = list(config["actuated"])
    joints = joint_mirror(names, spec)
    mean = np.asarray(config["skill_mean"])
    std = np.asarray(config["skill_std"])
    if config.get("units") is not None:
        scale, shift = (np.asarray(u) for u in config["units"])
        std, mean = std / scale, mean - shift * (std / scale)
    position = action = np.zeros(len(names))
    if "default_joint_pos" in spec:
        default = np.array([spec["default_joint_pos"][n] for n in names])
        position, _ = joint_offsets(joints, default, 1.0)
    if "action" in spec:
        recorded = spec["action"]
        index = [recorded["joints"].index(n) for n in names]
        offset = np.array(recorded["offset"])[index]
        scale = np.array(recorded["scale"])[index]
        action = joint_offsets(joints, offset, 1.0)[0] / scale
    mirror = observation_mirror(terms, joints, mean, std, position, action)
    return {
        "mirror_linear": mirror.linear,
        "mirror_offset": mirror.offset,
        "action_joints": joints,
        "action_offset": action,
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--like", type=Path, required=True, help="A bundle to take the rest from.")
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain", action="store_true", help="Do not symmetrise.")
    args = parser.parse_args()
    config = json.loads((args.like / "runtime.json").read_text(encoding="utf-8"))
    base = [(n, w) for n, w in config["policy_terms"] if not n.endswith("_history")]
    widths = dict(base)
    terms = base + [(f"{n}_history", args.history * widths[n]) for n in HISTORY_TERMS] * bool(
        args.history
    )
    config["policy_terms"] = [list(t) for t in terms]
    config["history"] = (
        {f"{n}_history": args.history for n in HISTORY_TERMS} if args.history else {}
    )
    config["policy"] = str(args.checkpoint)

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.double().numpy() for k, v in saved["actor_state_dict"].items() if v.ndim > 0}
    arrays = {
        "mean": state["obs_normalizer._mean"][0],
        "scale": state["obs_normalizer._std"][0] + EPSILON,
    }
    layers = sorted(
        int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")
    )
    for i, layer in enumerate(layers):
        arrays[f"weight_{i}"] = state[f"mlp.{layer}.weight"]
        arrays[f"bias_{i}"] = state[f"mlp.{layer}.bias"]
    width = arrays["weight_0"].shape[1]
    if width != sum(w for _, w in terms):
        raise ValueError(
            f"{args.checkpoint}: {width} inputs, the layout has {sum(w for _, w in terms)}"
        )
    if not args.plain:
        arrays.update(mirror_arrays(config, terms))

    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("body_model.json",):
        shutil.copy(args.like / name, args.output / name)
    shutil.copytree(args.like / "plans", args.output / "plans", dirs_exist_ok=True)
    (args.output / "runtime.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    np.savez(args.output / "actor.npz", **arrays)
    source = np.load(args.like / "actor.npz")
    if (
        source["mean"].shape == arrays["mean"].shape
        and "mirror_linear" in source
        and not args.plain
    ):
        gap = max(
            float(np.abs(source[k] - arrays[k]).max())
            for k in ("mirror_linear", "mirror_offset", "action_joints", "action_offset")
        )
        print(f"mirror against {args.like}: largest difference {gap:.1e}")
    print(f"wrote {args.output}: {width} inputs, history {args.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
