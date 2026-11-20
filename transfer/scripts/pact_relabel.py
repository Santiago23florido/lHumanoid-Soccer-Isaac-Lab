"""Relabel captures with another teacher, offline: the tracker's observation is recorded.

Every capture PACT trains on (``capture_contract.py --policy-obs``) holds, per step and
environment, the observation its tracker saw. A teacher chosen later
(``pact_teacher_audit_read.py``) labels the same states without simulating them again:
its action on each recorded observation, written beside the capture as
``<capture>.labels_<name>.npz`` (``action_teacher``, ``action_joints``, ``action_scale``,
``teacher``). The capture itself is not modified; ``pact_gate_d.py --labels <name>`` reads
the sidecar instead of the capture's own labels.

The action joints and scale are the body's action term's, read from one of its DAgger
captures (they record them); a capture driven by the tracker alone does not.

Usage::

    python transfer/scripts/pact_relabel.py --body g1 --name sym \\
        --teacher logs/rsl_rl/g1_track/<run>/exported/model_7193_sym_pact.pt \\
        --captures "logs/pact/coexec/g1_map.npz" "logs/pact/dagger/*/r*/g1.npz"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def sidecar(capture: Path, name: str) -> Path:
    return capture.with_name(f"{capture.stem}.labels_{name}.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body", required=True)
    parser.add_argument("--name", required=True, help="The labels' name, e.g. sym.")
    parser.add_argument(
        "--teacher", type=Path, required=True, help="An exported TorchScript actor."
    )
    parser.add_argument(
        "--captures", nargs="+", required=True, help="Paths or globs (repo-relative)."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=65536)
    args = parser.parse_args()

    paths = sorted({p for pattern in args.captures for p in REPO.glob(pattern)})
    paths = [p for p in paths if ".labels_" not in p.name]
    reference = next((p for p in sorted(REPO.glob(f"logs/pact/dagger/*/r*/{args.body}.npz"))), None)
    if reference is None:
        raise SystemExit(f"no DAgger capture of {args.body} to read its action joints from")
    with np.load(reference) as ref:
        joints = np.asarray(ref["action_joints"])
        scale = np.asarray(ref["action_scale"])
    teacher = torch.jit.load(str(args.teacher), map_location=args.device).eval()
    for path in paths:
        with np.load(path) as data:
            obs = data["policy_obs"]
        flat = obs.reshape(-1, obs.shape[-1])
        out = []
        with torch.inference_mode():
            for start in range(0, len(flat), args.chunk):
                x = torch.as_tensor(flat[start : start + args.chunk], device=args.device)
                out.append(teacher(x.float()).cpu().numpy())
        actions = np.concatenate(out).reshape(*obs.shape[:2], -1).astype(np.float32)
        if actions.shape[-1] != len(joints):
            raise SystemExit(f"{args.teacher}: {actions.shape[-1]} actions, {len(joints)} joints")
        target = sidecar(path, args.name)
        np.savez(
            target,
            action_teacher=actions,
            action_joints=joints,
            action_scale=scale,
            teacher=np.array(str(args.teacher)),
        )
        print(f"{path.relative_to(REPO)} -> {target.name}: {actions.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
