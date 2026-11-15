"""A teacher told one fixed robot, whatever robot it drives: the control for its conditioning.

Wraps a teacher's TorchScript actor (``NaoTrack-Sim2RealTeacher-Play-v0``, whose policy
group ends with ``robot_params``) so that it reads the given parameters instead of the
true ones. Evaluated on robots that differ from them (``nao/scripts/evaluate_adaptation.sh``),
it measures what being told the robot is worth, without training anything:
``nao/scripts/compare_sweeps.py`` against the same teacher told the truth.

The values are in their own units, normalised over the ranges of a bundle that reads
them (``runtime.json``'s ``robot_params``); every parameter must be given::

    python nao/scripts/constant_params.py <run>/exported/model_<n>_sym.pt \
        <run>/exported/model_<n>_sym_nominal.pt --bundle nao/bundles/<student> \
        --params '{"delay_ms": 35, "stiffness_scale": 1, "damping_scale": 1,
                   "effort_scale": 1, "armature_scale": 1}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.deploy.controller import normalise_params  # noqa: E402


def main() -> int:
    import torch
    from torch import nn

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("teacher", type=Path, help="The teacher's TorchScript actor.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle", type=Path, required=True, help="Where the ranges are read.")
    parser.add_argument("--params", required=True, help="JSON: every parameter, own units.")
    args = parser.parse_args()
    config = json.loads((args.bundle / "runtime.json").read_text(encoding="utf-8"))
    values = normalise_params(json.loads(args.params), config)
    if np.isnan(values).any():
        names = config["robot_params"]["names"]
        missing = [n for n, v in zip(names, values, strict=True) if np.isnan(v)]
        raise SystemExit(f"every parameter must be given; missing {missing}")

    class Told(nn.Module):
        def __init__(self, base: torch.jit.ScriptModule, told: torch.Tensor) -> None:
            super().__init__()
            self.base = base
            self.register_buffer("told", told)

        def forward(self, observation: torch.Tensor) -> torch.Tensor:
            count = self.told.shape[0]
            told = self.told.to(observation.dtype).expand(observation.shape[0], count)
            return self.base(torch.cat([observation[:, :-count], told], dim=-1))

    base = torch.jit.load(str(args.teacher), map_location="cpu")
    told = torch.as_tensor(values, dtype=torch.float32)
    torch.jit.script(Told(base, told)).save(str(args.output))
    print(f"wrote {args.output}: robot_params fixed at {np.round(values, 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
