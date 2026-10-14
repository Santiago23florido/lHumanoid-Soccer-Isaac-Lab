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
  positions are relative to the default pose, which must itself be symmetric.
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


def joint_mirror(names: Sequence[str], spec: dict) -> np.ndarray:
    """``(J, J)``: the reflected joint vector is ``P @ q`` (``spec`` from ``measure_mirror.py``)."""
    rows = {row["name"]: row for row in spec["joints"]}
    names = list(names)
    matrix = np.zeros((len(names), len(names)))
    for i, name in enumerate(names):
        row = rows[name]
        matrix[names.index(row["partner"]), i] = row["sign"]
    return matrix


def load_joint_mirror(names: Sequence[str], path: Path | str) -> np.ndarray:
    return joint_mirror(names, json.loads(Path(path).read_text(encoding="utf-8")))


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
) -> AffineMirror:
    """The mirror of one observation group, from its terms' names and widths, in order.

    The names are those of the tracking environments' observation terms; one this
    does not know is an error, never passed through unreflected.
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
        elif name in ("joint_pos", "joint_vel", "actions"):
            mirror.add(joints)
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
        else:
            raise KeyError(f"no mirror for observation term {name!r}")
    return mirror.done()


__all__ = [
    "AXIAL",
    "AffineMirror",
    "PLANAR",
    "VECTOR",
    "joint_mirror",
    "load_joint_mirror",
    "normalised_skill_mirror",
    "observation_mirror",
]
