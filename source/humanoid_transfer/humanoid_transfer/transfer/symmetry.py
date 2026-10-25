"""The sagittal mirror of a tracker's inputs and outputs, as affine maps.

Reflecting a motion left to right turns a plan the tracker follows into another
plan it should follow, and its body's state into the mirrored body's. The
procedural family is closed under this reflection (§8.2, tested), and so is a
robot whose two sides are images of each other. The optimal tracker is then
*equivariant*: its action for the reflected input is its action reflected,
``π(A o + b) = P π(o)``. A tracker trained without being told so need not be,
and the G1's after round 2 was not: across the competence map its left stance
was 0.041 of a stride shorter than its right where every plan asked for the
same.

Two ways to impose it, both from these maps:

* train on each transition and its mirror image (Mittal et al., 2024; rsl-rl's
  symmetry augmentation), so the policy learns the equivariant function;
* average a trained policy with its reflection,
  ``½ (π(o) + P π(A o + b))`` -- the projection onto equivariant functions
  (the Reynolds operator of the two-element group), exact for any ``π``.

Every map here is its own inverse, and the tests hold them to that.

* **Joints** come from a body's measured mirror (``measure_mirror.py``): each
  joint's partner and whether a positive angle maps to a positive angle. Joint
  positions are read relative to the default pose ``q0`` and actions as offsets
  ``(q − q0) / scale``; if ``q0`` is not its own reflection (``P q0 ≠ q0``), both
  reflect with an offset, ``P x + (P q0 − q0)`` and ``P a + (P q0 − q0) / scale``
  (:func:`joint_offsets`). The G1's default pose is symmetric; the offset is
  computed anyway, for any body.
* **The skill state** is reflected by
  :func:`~humanoid_transfer.common.skill_state.mirror_matrix` in physical
  units. In a contract's normalised units ``(z − μ) / s`` that is affine:
  ``diag(1/s) M diag(s)`` plus ``(M μ − μ) / s``. A difference of two states
  (the plan relative to ``ẑ*``) drops the offset.
* **Base vectors** in the heading frame: velocities and gravity keep ``x`` and
  ``z`` and flip ``y``; angular velocity flips ``x`` and ``z``; a command
  ``(vx, vy, ωz)`` flips its last two, and the reference path ``(dx, dy, dψ)``
  too.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..common.skill_state import SKILL_LAYOUT, mirror_matrix

VECTOR = np.diag([1.0, -1.0, 1.0])
"""A polar vector in the heading frame (linear velocity, gravity)."""

AXIAL = np.diag([-1.0, 1.0, -1.0])
"""An axial vector (angular velocity)."""

PLANAR = np.diag([1.0, -1.0, -1.0])
"""``(x, y, yaw)``: a velocity command, or where the reference is and which way it faces."""


def _sides(block: np.ndarray) -> np.ndarray:
    """A per-foot quantity, left then right: the sides swap and each reflects by ``block``."""
    width = block.shape[0]
    swapped = np.zeros((2 * width, 2 * width))
    swapped[:width, width:] = block
    swapped[width:, :width] = block
    return swapped


BALANCE = np.block(
    [
        [_sides(np.diag([1.0, -1.0])), np.zeros((4, 6))],
        [np.zeros((6, 4)), _sides(VECTOR)],
    ]
)
"""Each foot's centre of pressure ``(x, y)`` then its ground force ``(x, y, z)``, heading frame
(:mod:`~humanoid_transfer.transfer.tasks.tracking.balance`)."""


def joint_mirror(names: Sequence[str], spec: dict) -> np.ndarray:
    """``(J, J)``: the reflected joint vector is ``P @ q`` (``spec`` from ``measure_mirror.py``).

    ``names`` may be a subset -- the joints a policy commands. A joint whose
    partner is not among them must have a partner that follows one of them
    (``spec["mimic"]``: follower to ``[leader, multiplier]``), as the NAO's right
    hip yaw-pitch follows its left on one motor; the reflection then lands on the
    leader, divided by the multiplier so the reflected pose keeps the constraint.
    """
    rows = {row["name"]: row for row in spec["joints"]}
    mimic = spec.get("mimic", {})
    names = list(names)
    matrix = np.zeros((len(names), len(names)))
    for i, name in enumerate(names):
        row = rows[name]
        partner, sign = row["partner"], row["sign"]
        if partner not in names:
            partner, multiplier = mimic[partner]
            sign = sign / multiplier
        matrix[names.index(partner), i] = sign
    return matrix


def load_joint_mirror(names: Sequence[str], path: Path | str) -> np.ndarray:
    return joint_mirror(names, json.loads(Path(path).read_text(encoding="utf-8")))


def joint_offsets(
    joints: np.ndarray, default: np.ndarray, scale: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Offsets reflecting positions relative to ``default`` and actions scaled by ``scale``.

    With ``P`` an involution, ``x ↦ P x + c`` is one exactly when ``P c = −c``,
    which ``c = P q0 − q0`` is.
    """
    position = joints @ default - default
    return position, position / scale


def normalised_skill_mirror(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(A, b)`` reflecting a normalised ``z*``: ``A z_n + b``."""
    m = mirror_matrix(SKILL_LAYOUT)
    linear = np.diag(1.0 / std) @ m @ np.diag(std)
    return linear, (m @ mean - mean) / std


class AffineMirror:
    """``x ↦ A x + b`` over the last dimension, built block by block."""

    def __init__(self, size: int) -> None:
        self.linear = np.zeros((size, size))
        self.offset = np.zeros(size)
        self._at = 0

    def add(self, linear: np.ndarray, offset: np.ndarray | None = None, repeat: int = 1) -> None:
        width = linear.shape[0]
        for _ in range(repeat):
            span = slice(self._at, self._at + width)
            self.linear[span, span] = linear
            if offset is not None:
                self.offset[span] = offset
            self._at += width

    def done(self) -> AffineMirror:
        if self._at != self.offset.shape[0]:
            raise ValueError(f"mirror covers {self._at} of {self.offset.shape[0]} inputs")
        return self


def observation_mirror(
    terms: Sequence[tuple[str, int]],
    joints: np.ndarray,
    skill_mean: np.ndarray,
    skill_std: np.ndarray,
    position_offset: np.ndarray | None = None,
    action_offset: np.ndarray | None = None,
) -> AffineMirror:
    """The mirror of one observation group, from its terms' names and widths, in order.

    The names are those of the tracking environments' observation terms; one this
    does not know is an error, never passed through unreflected. The offsets are
    :func:`joint_offsets`' (zero when the default pose is symmetric).
    """
    skill = SKILL_LAYOUT.size
    relative, _ = normalised_skill_mirror(skill_mean, skill_std)
    absolute = normalised_skill_mirror(skill_mean, skill_std)
    mirror = AffineMirror(sum(width for _, width in terms))
    for name, width in terms:
        if name in ("base_lin_vel", "projected_gravity"):
            mirror.add(VECTOR)
        elif name == "base_ang_vel":
            mirror.add(AXIAL)
        elif name in ("velocity_commands",):
            mirror.add(PLANAR)
        elif name == "joint_pos":
            mirror.add(joints, position_offset)
        elif name in ("joint_vel", "prior", "feedforward"):
            # Velocities, the prior's targets relative to the joints, and its feed-forward over
            # each joint's stiffness (a joint torque reflects as a joint velocity): no offset.
            mirror.add(joints)
        elif name == "actions":
            mirror.add(joints, action_offset)
        elif name == "contract":
            # The plan over the view relative to ẑ*, then ẑ*.
            mirror.add(relative, repeat=(width - skill) // skill)
            mirror.add(*absolute)
        elif name == "privileged":
            # The plan relative to the true z*, the true z*, the plan's age.
            mirror.add(relative, repeat=(width - skill - 1) // skill)
            mirror.add(*absolute)
            mirror.add(np.eye(1))
        elif name == "contract_path":
            mirror.add(PLANAR, repeat=width // 3)
        elif name == "balance":
            mirror.add(BALANCE)
        else:
            raise KeyError(f"no mirror for observation term {name!r}")
    return mirror.done()


__all__ = [
    "AXIAL",
    "AffineMirror",
    "BALANCE",
    "PLANAR",
    "VECTOR",
    "joint_mirror",
    "joint_offsets",
    "load_joint_mirror",
    "normalised_skill_mirror",
    "observation_mirror",
]
