"""The skill state: what a walking body is doing, in terms no particular body owns.

A gait recorded as joint angles describes the robot that produced it. The G1
walks with 37 joint positions, the NAO would need 19 different ones, and no
amount of renaming makes a G1 knee angle mean anything on a NAO. The quantities
that *do* mean the same thing on both are the ones the centroidal equations are
written in -- where the centre of mass is and how it moves, how the body is
tilted and turning, where the feet are, how the weight is shared between them,
and how much the body is spinning about its own centre:

======================  =====  ================  =========================
field                   size   frame             unit
======================  =====  ================  =========================
``height``              1      --                ``L``
``com_velocity``        3      heading           ``V = sqrt(g L)``
``gravity``             3      base              (unit vector)
``base_angular_vel``    3      base              ``1 / T``
``foot_position``       2x3    heading, from CoM ``L``
``foot_velocity``       2x3    heading           ``V``
``foot_load``           2      --                ``m g``
``angular_momentum``    3      heading           ``m L V``
======================  =====  ================  =========================

27 numbers, each divided by the body's own unit for its dimension
(:class:`~humanoid_transfer.common.similarity.UnitSystem`). Two consequences
are what make this a state *shared* between bodies rather than a renamed copy
of one of them, and both are tested rather than asserted:

1. **Similarity invariance.** Two dynamically similar motions of bodies of any
   size encode to the same vector. ``com_velocity`` *is* the square root of the
   Froude number; Froude matching is not a separate step any more.
2. **Gauge invariance.** Positions are taken from the centre of mass and
   vectors are expressed in the *heading frame* -- the world turned by the
   base's yaw -- so where the robot stands and which way it faces drop out.

The heading and base frames come from the **base**: the body that carries
both hip joints (the G1's pelvis, the NAO's ``base_link``). Its x-z plane is
the sagittal plane on both robots, which is what the mirror below relies on.

Contact is not a 0/1 flag here but the **load share** ``f_z / (m g)``: the
fraction of body weight each foot carries. A binary flag is derived from it by
:func:`contact_state` with a threshold that is itself a fraction of body
weight. Isaac Lab's default is 1 N, which is 0.32 % of the G1's weight and
1.9 % of the NAO's -- the same number in newtons is a different criterion on
each body, which is exactly the mistake this module exists to stop making.

The left-right mirror
---------------------
A sagittal reflection ``S = diag(1, -1, 1)`` maps a gait onto its mirror image.
Each field transforms according to what kind of quantity it is:

* scalar -- unchanged;
* polar vector (positions, velocities, gravity) -- ``v -> S v``;
* axial vector (angular velocity, angular momentum) -- ``a -> -S a``, because
  a reflection reverses the sense of rotation;
* sided fields -- left and right swap.

:func:`mirror_matrix` generates the operator from those declarations, so a
new field cannot be added without saying what kind it is.

Pure NumPy, batched over any leading dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .capturability import dcm
from .centroidal import angular_momentum, center_of_mass, com_velocity
from .frames import rotate, rotate_inverse, yaw_matrix, yaw_of
from .similarity import (
    ANGULAR_MOMENTUM,
    ANGULAR_VELOCITY,
    DIMENSIONLESS,
    FORCE,
    LENGTH,
    VELOCITY,
    Dimension,
    UnitSystem,
)

Kind = Literal["scalar", "polar", "axial"]

SIDES = ("left", "right")
_AXES = ("x", "y", "z")
_REFLECTION = np.diag([1.0, -1.0, 1.0])

DEFAULT_CONTACT_THRESHOLD = 0.05
"""Load share above which a foot counts as in contact: 5 % of body weight.

A choice, and reported as one: the gait statistics that depend on it are
published with their sensitivity to it (1 % to 10 %), not at this value alone.
"""


@dataclass(frozen=True)
class Field:
    """One named block of the skill state."""

    name: str
    kind: Kind
    dimension: Dimension
    frame: str
    """Where the components are expressed: ``heading``, ``base`` or ``-``."""

    sided: bool = False
    """True for a quantity measured once per foot, stored left then right."""

    @property
    def width(self) -> int:
        return 1 if self.kind == "scalar" else 3

    @property
    def size(self) -> int:
        return self.width * (2 if self.sided else 1)

    def mirror_block(self) -> np.ndarray:
        """The mirror on one side's components: identity, ``S`` or ``-S``."""
        if self.kind == "scalar":
            return np.eye(1)
        return _REFLECTION if self.kind == "polar" else -_REFLECTION


@dataclass(frozen=True)
class ZLayout:
    """Ordered fields and the bookkeeping to pack, unpack and mirror them."""

    fields: tuple[Field, ...]

    @property
    def size(self) -> int:
        return sum(field.size for field in self.fields)

    def slice_of(self, name: str) -> slice:
        start = 0
        for field in self.fields:
            if field.name == name:
                return slice(start, start + field.size)
            start += field.size
        raise KeyError(f"no field named {name!r}")

    def field(self, name: str) -> Field:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no field named {name!r}")

    def component_names(self) -> tuple[str, ...]:
        """One label per entry, e.g. ``foot_position.left.y``, for reports."""
        names: list[str] = []
        for field in self.fields:
            for side in SIDES if field.sided else (None,):
                prefix = f"{field.name}.{side}" if side else field.name
                if field.kind == "scalar":
                    names.append(prefix)
                else:
                    names.extend(f"{prefix}.{axis}" for axis in _AXES)
        return tuple(names)

    def pack(self, values_si: dict[str, np.ndarray], units: UnitSystem) -> np.ndarray:
        """Divide each field by its unit and concatenate, ``-> (..., size)``.

        Sided fields are given as ``(..., 2, width)`` and scalars as ``(...)``
        or ``(..., 2)`` when sided. The unit comes from the field's declared
        dimension, so there is no per-field factor to get wrong.
        """
        blocks = []
        for field in self.fields:
            value = units.to_dimensionless(values_si[field.name], field.dimension)
            if field.kind == "scalar" and not field.sided:
                value = value[..., None]
            blocks.append(value.reshape(*value.shape[: value.ndim - _trailing(field)], field.size))
        return np.concatenate(blocks, axis=-1)

    def unpack(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """The inverse of :meth:`pack`, still dimensionless."""
        z = np.asarray(z, dtype=float)
        out: dict[str, np.ndarray] = {}
        for field in self.fields:
            block = z[..., self.slice_of(field.name)]
            if field.sided:
                block = block.reshape(*block.shape[:-1], 2, field.width)
            out[field.name] = block[..., 0] if field.kind == "scalar" else block
        return out


def _trailing(field: Field) -> int:
    """How many trailing axes a field's SI value occupies before flattening."""
    if field.kind == "scalar":
        return 1
    return 2 if field.sided else 1


SKILL_LAYOUT = ZLayout(
    fields=(
        Field("height", "scalar", LENGTH, "-"),
        Field("com_velocity", "polar", VELOCITY, "heading"),
        Field("gravity", "polar", DIMENSIONLESS, "base"),
        Field("base_angular_velocity", "axial", ANGULAR_VELOCITY, "base"),
        Field("foot_position", "polar", LENGTH, "heading", sided=True),
        Field("foot_velocity", "polar", VELOCITY, "heading", sided=True),
        Field("foot_load", "scalar", FORCE, "-", sided=True),
        Field("angular_momentum", "axial", ANGULAR_MOMENTUM, "heading"),
    )
)
"""The 27-dimensional skill state every encoder in the transfer study targets."""


@dataclass(frozen=True)
class MultibodySnapshot:
    """What a simulator reports about a floating-base robot, reduced to what the encoder needs.

    Per-body arrays carry bodies on axis ``-2`` (``-3`` for matrices); any
    leading dimensions (time, environment) are kept. Everything is in the
    world frame and SI units, with gravity along ``-z`` and flat ground at
    ``ground_height``.
    """

    masses: np.ndarray
    """``(..., B)``, kg."""

    inertia_local: np.ndarray
    """``(..., B, 3, 3)``, about each body's CoM, in its link frame."""

    link_rot: np.ndarray
    """``(..., B, 3, 3)``, world-from-link rotations."""

    com_pos: np.ndarray
    """``(..., B, 3)``, each body's centre of mass."""

    com_lin_vel: np.ndarray
    """``(..., B, 3)``, velocity at each body's centre of mass."""

    ang_vel: np.ndarray
    """``(..., B, 3)``, each body's angular velocity."""

    foot_force: np.ndarray
    """``(..., 2, 3)``, net contact force on the left and right foot."""

    base_index: int
    """The body carrying both hips. Its yaw defines the heading frame."""

    foot_index: tuple[int, int]
    """Left and right foot bodies."""

    sole_from_com: np.ndarray
    """``(2, 3)``, from each foot body's CoM to its sole point, in the foot link frame."""

    ground_height: float = 0.0


def sole_points(snapshot: MultibodySnapshot) -> tuple[np.ndarray, np.ndarray]:
    """World position and velocity of both sole points, each ``(..., 2, 3)``."""
    feet = list(snapshot.foot_index)
    rotation = snapshot.link_rot[..., feet, :, :]
    offset = rotate(rotation, np.asarray(snapshot.sole_from_com, dtype=float))
    position = snapshot.com_pos[..., feet, :] + offset
    velocity = snapshot.com_lin_vel[..., feet, :] + np.cross(snapshot.ang_vel[..., feet, :], offset)
    return position, velocity


def skill_values_si(snapshot: MultibodySnapshot) -> dict[str, np.ndarray]:
    """Every field of :data:`SKILL_LAYOUT` in SI units, already in its frame.

    Separate from :func:`encode` so the dimensional and the geometric halves of
    the encoder can be checked apart: this function does all the frame work and
    no scaling, and :meth:`ZLayout.pack` does all the scaling and no geometry.
    """
    masses = np.asarray(snapshot.masses, dtype=float)
    c = center_of_mass(masses, snapshot.com_pos)
    c_dot = com_velocity(masses, snapshot.com_lin_vel)
    momentum = angular_momentum(
        masses,
        snapshot.inertia_local,
        snapshot.link_rot,
        snapshot.com_pos,
        snapshot.com_lin_vel,
        snapshot.ang_vel,
    ).total

    base_rot = snapshot.link_rot[..., snapshot.base_index, :, :]
    heading = yaw_matrix(yaw_of(base_rot))
    down = np.broadcast_to(np.array([0.0, 0.0, -1.0]), c.shape)
    sole_pos, sole_vel = sole_points(snapshot)

    return {
        "height": c[..., 2] - snapshot.ground_height,
        "com_velocity": rotate_inverse(heading, c_dot),
        "gravity": rotate_inverse(base_rot, down),
        "base_angular_velocity": rotate_inverse(
            base_rot, snapshot.ang_vel[..., snapshot.base_index, :]
        ),
        "foot_position": rotate_inverse(heading[..., None, :, :], sole_pos - c[..., None, :]),
        "foot_velocity": rotate_inverse(heading[..., None, :, :], sole_vel),
        "foot_load": np.asarray(snapshot.foot_force, dtype=float)[..., 2],
        "angular_momentum": rotate_inverse(heading, momentum),
    }


def encode(
    snapshot: MultibodySnapshot, units: UnitSystem, layout: ZLayout = SKILL_LAYOUT
) -> np.ndarray:
    """The analytic encoder: a body's state, in that body's units, ``-> (..., 27)``."""
    return layout.pack(skill_values_si(snapshot), units)


def mirror_matrix(layout: ZLayout = SKILL_LAYOUT) -> np.ndarray:
    """The sagittal mirror as a signed permutation, ``(size, size)``. It is its own inverse."""
    matrix = np.zeros((layout.size, layout.size))
    for field in layout.fields:
        start = layout.slice_of(field.name).start
        block, width = field.mirror_block(), field.width
        if field.sided:
            left, right = start, start + width
            matrix[left : left + width, right : right + width] = block
            matrix[right : right + width, left : left + width] = block
        else:
            matrix[start : start + width, start : start + width] = block
    return matrix


def mirror(z: np.ndarray, layout: ZLayout = SKILL_LAYOUT) -> np.ndarray:
    """The skill state of the mirror-image motion."""
    return np.asarray(z, dtype=float) @ mirror_matrix(layout).T


# ---------------------------------------------------------------------------
# Quantities derived from the state, never stored in it
# ---------------------------------------------------------------------------


def froude_number(z: np.ndarray, layout: ZLayout = SKILL_LAYOUT) -> np.ndarray:
    """Horizontal ``Fr = v^2 / (g L)``, read straight off the state."""
    velocity = layout.unpack(z)["com_velocity"]
    return velocity[..., 0] ** 2 + velocity[..., 1] ** 2


def dcm_offset(z: np.ndarray, layout: ZLayout = SKILL_LAYOUT) -> np.ndarray:
    """Divergent component of motion relative to the CoM, ``(xi - c)_xy / L``.

    In body units the pendulum frequency at height ``h* = c_z / L`` is
    ``1 / sqrt(h*)``, so the offset is ``v* sqrt(h*)``. Computed through
    :func:`~humanoid_transfer.common.capturability.dcm` so there is one
    definition of the DCM in the repository, not two.
    """
    values = layout.unpack(z)
    velocity = values["com_velocity"][..., :2]
    omega = 1.0 / np.sqrt(values["height"])[..., None]
    return dcm(np.zeros_like(velocity), velocity, omega)


def contact_state(
    z: np.ndarray,
    layout: ZLayout = SKILL_LAYOUT,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
) -> np.ndarray:
    """Per-foot contact, ``(..., 2)``: load share above ``threshold`` of body weight."""
    return layout.unpack(z)["foot_load"] > threshold


__all__ = [
    "DEFAULT_CONTACT_THRESHOLD",
    "Field",
    "MultibodySnapshot",
    "SIDES",
    "SKILL_LAYOUT",
    "ZLayout",
    "contact_state",
    "dcm_offset",
    "encode",
    "froude_number",
    "mirror",
    "mirror_matrix",
    "skill_values_si",
    "sole_points",
]
