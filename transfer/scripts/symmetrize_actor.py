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

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402
from humanoid_transfer.transfer.symmetry import (  # noqa: E402
    joint_mirror,
    joint_offsets,
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
        self,
        actor: nn.Module,
        linear: torch.Tensor,
        offset: torch.Tensor,
        joints: torch.Tensor,
        action_offset: torch.Tensor,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.register_buffer("linear", linear)
        self.register_buffer("offset", offset)
        self.register_buffer("joints", joints)
        self.register_buffer("action_offset", action_offset)

    def reflect_action(self, action: torch.Tensor) -> torch.Tensor:
        return action @ self.joints.T + self.action_offset

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        reflected = observation @ self.linear.T + self.offset
        return 0.5 * (self.actor(observation) + self.reflect_action(self.actor(reflected)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("actor", type=Path, help="An exported TorchScript actor.")
    parser.add_argument("--mirror", type=Path, required=True, help="The body's measure_mirror.py.")
    parser.add_argument("--contract", type=Path, required=True, help="Fixes the plan's units.")
    parser.add_argument("--view", type=int, default=None, help="Default: from the actor's width.")
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
    actor = torch.jit.load(str(args.actor), map_location="cpu").eval()
    imu = args.body == "nao"
    # The view is the tracker's, counted on its own body's grid, not the generator's: the
    # NAO's control step is longer in body time, and it sees 4 of the plan's samples where
    # the G1 sees 8. It follows from the actor's input width.
    width = int(actor.mean.shape[0])
    fixed = sum(w for _, w in tracker_terms(len(names), 0, imu=imu))
    view, rest = divmod(width - fixed, SKILL_LAYOUT.size + 3)
    if rest or (args.view is not None and args.view != view):
        raise ValueError(f"{args.actor}: {width} inputs fit no view (asked {args.view})")
    norm = contract.generator.skill_norm
    joints = joint_mirror(names, spec)
    # Offsets for a default pose that is not its own reflection (zero on the G1).
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
    terms = tracker_terms(len(names), view, imu=imu)
    mirror = observation_mirror(
        terms, joints, norm.mean.numpy(), norm.std.numpy(), position, action
    )
    probe = torch.zeros(1, width)
    if actor(probe).shape[-1] != len(names):
        raise ValueError(f"{args.actor}: actions are not the mirror's {len(names)} joints")
    module = Symmetrized(
        actor,
        torch.as_tensor(mirror.linear, dtype=torch.float32),
        torch.as_tensor(mirror.offset, dtype=torch.float32),
        torch.as_tensor(joints, dtype=torch.float32),
        torch.as_tensor(action, dtype=torch.float32),
    ).eval()
    output = args.output or args.actor.with_name(args.actor.stem + "_sym.pt")
    torch.jit.script(module).save(str(output))
    x = torch.randn(64, width)
    reflected = x @ module.linear.T + module.offset
    gap = (module(reflected) - module.reflect_action(module(x))).abs().max()
    print(f"{output}: {width} inputs, view {view}; equivariance gap {float(gap):.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
