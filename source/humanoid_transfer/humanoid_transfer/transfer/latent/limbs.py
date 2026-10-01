"""A decoder per limb: each takes its own share of the latent action.

Two humanoids differ in how many joints a limb has, not in which limbs there
are. Splitting the latent action ``u`` by limb confines that difference to
each limb's head: a leg head turns "this leg's command" into this body's leg
joints, an arm head does the same for an arm, and a limb the second body
cannot use the way the first did (the NAO's elbow, which does not straighten)
stays inside its own head instead of reshaping the legs'.

``u = (u_0, u_trunk, u_left_leg, u_right_leg, u_left_arm, u_right_arm)``:
``u_0`` is shared by every head, the rest are private. Each head also sees
its own joints' positions and velocities and nothing else. The coordination
between limbs -- balance is a whole-body affair -- is the skill policy's job,
which sees the whole physical state; the heads only translate.

The trunk head exists for a joint two legs would otherwise fight over: the
NAO's ``LHipYawPitch`` and ``RHipYawPitch`` are one motor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from .model import Normaliser, mlp


@dataclass(frozen=True)
class Limb:
    name: str
    joints: tuple[int, ...]
    """Indices into the body's joint order (the action and the context's halves)."""

    action: int
    """Width of this limb's private share of ``u``."""


@dataclass(frozen=True)
class LimbLayout:
    shared: int
    """Width of ``u_0``, read by every head."""

    limbs: tuple[Limb, ...]
    joints: int

    @property
    def action(self) -> int:
        return self.shared + sum(limb.action for limb in self.limbs)

    @classmethod
    def from_names(
        cls, names: list[str], patterns: dict[str, tuple[str, int]], shared: int
    ) -> LimbLayout:
        """Assign each joint to the one limb whose pattern matches its whole name."""
        owner: dict[int, str] = {}
        limbs = []
        for limb, (pattern, action) in patterns.items():
            joints = tuple(i for i, name in enumerate(names) if re.fullmatch(pattern, name))
            if not joints:
                raise ValueError(f"no joint matches limb {limb!r} ({pattern})")
            for joint in joints:
                if joint in owner:
                    raise ValueError(f"{names[joint]} matches {owner[joint]!r} and {limb!r}")
                owner[joint] = limb
            limbs.append(Limb(limb, joints, action))
        unassigned = [name for i, name in enumerate(names) if i not in owner]
        if unassigned:
            raise ValueError(f"joints in no limb: {unassigned}")
        return cls(shared=shared, limbs=tuple(limbs), joints=len(names))


G1_LIMB_PATTERNS: dict[str, tuple[str, int]] = {
    "trunk": (r"torso_joint", 2),
    "left_leg": (r"left_(hip|knee|ankle)_.*", 3),
    "right_leg": (r"right_(hip|knee|ankle)_.*", 3),
    "left_arm": (r"left_(shoulder|elbow|zero|one|two|three|four|five|six)_.*", 2),
    "right_arm": (r"right_(shoulder|elbow|zero|one|two|three|four|five|six)_.*", 2),
}
"""The G1's 37 joints by limb, the hand's fingers with their arm. With ``u_0``
of 4 the shares add to 16, the memoryless model's ``u``."""

G1_SHARED_ACTION = 4


def g1_limbs(joint_names: list[str]) -> LimbLayout:
    return LimbLayout.from_names(joint_names, G1_LIMB_PATTERNS, G1_SHARED_ACTION)


class LimbHead(nn.Module):
    """One limb: ``(u_0, u_limb, q_limb, q̇_limb)`` to the limb's joint targets."""

    def __init__(
        self, shared: int, start: int, stop: int, context: list[int], joints: int, hidden: int
    ) -> None:
        super().__init__()
        self.shared = shared
        self.start = start
        self.stop = stop
        self.register_buffer("context_index", torch.tensor(context, dtype=torch.long))
        self.net = mlp(shared + (stop - start) + len(context), joints, hidden)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        own = torch.cat(
            [
                action[..., : self.shared],
                action[..., self.start : self.stop],
                context.index_select(-1, self.context_index),
            ],
            dim=-1,
        )
        return self.net(own)


class LimbDecoder(nn.Module):
    """``D_r`` by limb, with the same signature as :class:`~.model.BodyDecoder`.

    ``context`` is the body's joint positions then velocities, in its joint
    order; the output is in that order too.
    """

    def __init__(self, layout: LimbLayout, hidden: int = 128) -> None:
        super().__init__()
        self.context_norm = Normaliser(2 * layout.joints)
        heads = []
        start = layout.shared
        order: list[int] = []
        for limb in layout.limbs:
            context = list(limb.joints) + [layout.joints + j for j in limb.joints]
            heads.append(
                LimbHead(
                    layout.shared, start, start + limb.action, context, len(limb.joints), hidden
                )
            )
            start += limb.action
            order.extend(limb.joints)
        self.heads = nn.ModuleList(heads)
        self.register_buffer("inverse", torch.argsort(torch.tensor(order, dtype=torch.long)))

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = self.context_norm(context)
        outputs: list[torch.Tensor] = []
        for head in self.heads:
            outputs.append(head(action, context))
        return torch.cat(outputs, dim=-1).index_select(-1, self.inverse)


__all__ = [
    "G1_LIMB_PATTERNS",
    "G1_SHARED_ACTION",
    "Limb",
    "LimbDecoder",
    "LimbHead",
    "LimbLayout",
    "g1_limbs",
]
