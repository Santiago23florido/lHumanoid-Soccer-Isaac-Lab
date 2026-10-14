"""Make an exported tracker equivariant under the sagittal mirror, without training it.

``½ (π(o) + P π(A o + b))`` is the projection of any policy onto the policies
that answer a reflected input with the reflected action (the Reynolds operator
of the mirror's two-element group; :mod:`humanoid_transfer.transfer.symmetry`).
It is exact, needs no data, and tests two things at once: whether a tracker's
left-right bias is what keeps it from its plans, and whether the mirror maps
are right -- a wrong sign anywhere averages a policy with a body it does not
have, and the robot falls.

The input layout is the tracker's policy group (``TrackPolicyCfg``), its plan
in the units of ``--contract`` (a body's contracts share them).

Usage::

    python transfer/scripts/symmetrize_actor.py logs/rsl_rl/g1_track/<run>/exported/model_5195.pt \
        --mirror g1/body/mirror.json --contract logs/contract/walk_zs_g1_r2_5195/generator.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.symmetry import (  # noqa: E402
    joint_mirror,
    observation_mirror,
)


def tracker_terms(joints: int, view: int, imu: bool = False) -> list[tuple[str, int]]:
    """The tracker's policy group, in order: the G1's ``TrackPolicyCfg``, or with ``imu``
    the NAO's ``NaoTrackPolicyCfg``, which has no base linear velocity."""
    skill = SKILL_LAYOUT.size
    return [
        *([] if imu else [("base_lin_vel", 3)]),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", joints),
        ("joint_vel", joints),
        ("actions", joints),
        ("contract", view * skill + skill),
        ("contract_path", view * 3),
    ]


class Symmetrized(nn.Module):
    def __init__(
        self, actor: nn.Module, linear: torch.Tensor, offset: torch.Tensor, joints: torch.Tensor
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("linear", linear)
        self.register_buffer("offset", offset)
        self.register_buffer("joints", joints)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        reflected = observation @ self.linear.T + self.offset
        return 0.5 * (self.actor(observation) + self.actor(reflected) @ self.joints.T)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("actor", type=Path, help="An exported TorchScript actor.")
    parser.add_argument("--mirror", type=Path, required=True, help="The body's measure_mirror.py.")
    parser.add_argument("--contract", type=Path, required=True, help="Fixes the plan's units.")
    parser.add_argument("--view", type=int, default=None, help="Default: the contract's.")
    parser.add_argument(
        "--body",
        choices=("g1", "nao"),
        default="g1",
        help="g1: every joint, base velocity observed; nao: the actuated joints "
        "(nao_kinematics.ACTUATED_JOINTS), no base linear velocity.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    spec = json.loads(args.mirror.read_text(encoding="utf-8"))
    names = [row["name"] for row in spec["joints"]]
    if args.body == "nao":
        from humanoid_transfer.nao.assets.nao_kinematics import ACTUATED_JOINTS

        names = list(ACTUATED_JOINTS)
    contract = load_contract(args.contract)
    view = args.view or contract.generator.view
    norm = contract.generator.skill_norm
    joints = joint_mirror(names, spec)
    terms = tracker_terms(len(names), view, imu=args.body == "nao")
    mirror = observation_mirror(terms, joints, norm.mean.numpy(), norm.std.numpy())
    actor = torch.jit.load(str(args.actor), map_location="cpu").eval()
    width = mirror.offset.shape[0]
    probe = torch.zeros(1, width)
    if actor(probe).shape[-1] != len(names):
        raise ValueError(f"{args.actor}: actions are not the mirror's {len(names)} joints")
    module = Symmetrized(
        actor,
        torch.as_tensor(mirror.linear, dtype=torch.float32),
        torch.as_tensor(mirror.offset, dtype=torch.float32),
        torch.as_tensor(joints, dtype=torch.float32),
    ).eval()
    output = args.output or args.actor.with_name(args.actor.stem + "_sym.pt")
    torch.jit.script(module).save(str(output))
    x = torch.randn(64, width)
    gap = (module(x @ module.linear.T + module.offset) - module(x) @ module.joints.T).abs().max()
    print(f"{output}: {width} inputs, view {view}; equivariance gap {float(gap):.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
