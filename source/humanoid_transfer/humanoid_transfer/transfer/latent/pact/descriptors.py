"""Body descriptors made of Buckingham-Pi groups.

A body is described joint by joint, from the card that
``transfer/scripts/measure_body.py`` writes. Every feature is a dimensionless
group in the body's own units (leg length ``L``, total mass ``m``, gravity
``g``; time ``sqrt(L/g)``), so two dynamically similar bodies -- one a scaled
copy of the other, with its drives scaled as similarity requires -- have the
same descriptor, feature for feature. That property is what lets one network
read many bodies without per-body input normalisation (URMA and QWM normalise
raw SI values per feature instead).

The kinematic tree enters as two matrices over degrees of freedom: the tree
distance between their child links, and whether one is an ancestor of the
other. A graph-biased attention (GET-Zero, Graphormer) turns them into biases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GRAVITY = 9.81
"""The gravity every simulation in this repository uses (``common/scaling.py``)."""

POSITION_LIMIT_CLIP = 2.0 * np.pi
"""Joint limits beyond one turn carry no more information; infinite ones are clipped here."""

VELOCITY_LIMIT_CLIP = 100.0
"""A joint speed limit above this many radians per ``sqrt(L/g)`` never binds a gait
(the G1's tightest is 5.2, the NAO's knee 1.0); unlimited joints (H1's implicit
drives report ``inf``) are clipped here."""

EFFORT_LIMIT_CLIP = 2.0
"""A joint torque limit above this many ``m g L`` never binds a walking gait (measured peak
across all five bipeds' walking probes is 1.05 ``m g L``, and every bounded body sits at or
below 1.45 ``m g L``); unlimited drives (Digit's implicit drives report ``FLT_MAX ~ 3.4e38``)
are clipped here in both the descriptor and the commanded-torque ratio."""

JOINT_FEATURES: tuple[str, ...] = (
    "position.x",
    "position.y",
    "position.z",
    "axis.x",
    "axis.y",
    "axis.z",
    "limit.lower",
    "limit.upper",
    "default_position",
    "effort_limit",
    "velocity_limit",
    "stiffness",
    "damping",
    "armature",
    "friction",
    "subtree_mass",
    "subtree_com.x",
    "subtree_com.y",
    "subtree_com.z",
    "subtree_inertia_about_axis",
)
"""Per degree of freedom, each one a Pi group:

- ``position``: the joint's origin relative to the base body, over ``L``;
- ``axis``: its unit axis in the base frame of the zero pose;
- ``limit``, ``default_position``: radians (already dimensionless);
- ``effort_limit``: over ``m g L``, clipped at :data:`EFFORT_LIMIT_CLIP`;
- ``velocity_limit``: the tighter of the drive's and PhysX's, times ``sqrt(L/g)``,
  clipped at :data:`VELOCITY_LIMIT_CLIP`;
- ``stiffness``: over ``m g L`` per radian;
- ``damping``: over ``m g L sqrt(L/g)`` per radian per second;
- ``armature``: over ``m L^2``;
- ``friction``: PhysX's joint friction coefficient, dimensionless;
- ``subtree_mass``: the mass the joint moves, over ``m``;
- ``subtree_com``: that mass's centre relative to the joint, over ``L``;
- ``subtree_inertia_about_axis``: its inertia about the joint axis, over ``m L^2``.
"""

BODY_FEATURES: tuple[str, ...] = (
    "control_period",
    "physics_dt",
    "action_scale",
    "default_com_height",
    "foot_length",
    "foot_width",
    "foot_separation",
)
"""Per body: the control period and physics step in units of ``sqrt(L/g)``, the
action scale (radians), the default posture's CoM height above the soles, the
sole's length and width and the soles' lateral separation in the zero pose, all
over ``L``."""


@dataclass(frozen=True)
class BodyDescriptor:
    """A body as the multi-body latent reads it."""

    name: str
    joint_names: tuple[str, ...]
    joints: np.ndarray
    """``(J, len(JOINT_FEATURES))``, one row per degree of freedom, card order."""
    body: np.ndarray
    """``(len(BODY_FEATURES),)``."""
    tree_distance: np.ndarray
    """``(J, J)`` integer: links between the two joints' child links in the tree."""
    ancestor: np.ndarray
    """``(J, J)`` bool: ``ancestor[i, j]`` when joint ``i`` is on joint ``j``'s path to the root."""
    leg_length: float
    mass: float

    def index(self, names: list[str] | tuple[str, ...]) -> np.ndarray:
        """Rows of the given joints, in the given order (e.g. a task's action joints)."""
        position = {name: i for i, name in enumerate(self.joint_names)}
        missing = [name for name in names if name not in position]
        if missing:
            raise KeyError(f"{self.name}: no joints {missing}")
        return np.array([position[name] for name in names], dtype=int)


def _rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float) / np.linalg.norm(quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _link_ancestors(names: list[str], parents: list[str]) -> list[list[int]]:
    """For each link, its chain of ancestors from itself up to the root (inclusive)."""
    index = {name: i for i, name in enumerate(names)}
    chains = []
    for i in range(len(names)):
        chain, current, seen = [i], i, {i}
        while parents[current] in index:
            current = index[parents[current]]
            if current in seen:
                raise ValueError(f"link tree has a cycle at {names[current]}")
            chain.append(current)
            seen.add(current)
        chains.append(chain)
    return chains


def describe(card: dict[str, Any], *, leg_length: float | None = None) -> BodyDescriptor:
    """The descriptor of the body a card records.

    ``leg_length`` overrides the card's measured unit, so a body keeps the unit its
    trackers were trained in (the NAO's is restated in ``transfer/embodiment.py``).
    """
    length = float(leg_length if leg_length is not None else card["leg_length"])
    mass = float(card["total_mass"])
    time = np.sqrt(length / GRAVITY)
    weight_length = mass * GRAVITY * length

    links = list(card["bodies"]["names"])
    parents = list(card["bodies"]["parents"])
    link_index = {name: i for i, name in enumerate(links)}
    chains = _link_ancestors(links, parents)

    zero = card["postures"]["zero"]
    link_pos = np.asarray(zero["link_pos"], dtype=float)
    link_rot = [_rotation(q) for q in zero["link_quat_wxyz"]]
    com_world = np.asarray(zero["com_pos"], dtype=float)
    link_mass = np.asarray(card["bodies"]["mass"], dtype=float)
    inertia_link = np.asarray(card["bodies"]["inertia_link"], dtype=float)
    base = link_index[card["base_body"]]
    base_rot = link_rot[base]
    base_pos = link_pos[base]

    joint_names = list(card["joints"]["names"])
    geometry = {row["joint"]: row for row in card["joint_geometry"]}
    drives = {row["joint"]: row for row in card["drives"]}
    limits = np.asarray(card["joints"]["pos_limits"], dtype=float)
    speed_limits = np.asarray(card["joints"]["vel_limits"], dtype=float)
    default = np.asarray(card["joints"]["default_pos"], dtype=float)

    rows = []
    child_links = []
    for j, name in enumerate(joint_names):
        geo, drive = geometry[name], drives[name]
        child = link_index[geo["child_link"]]
        child_links.append(child)
        origin = np.asarray(geo["position_root"], dtype=float)
        axis = np.asarray(geo["axis_root"], dtype=float)
        position_b = base_rot.T @ (origin - base_pos) / length
        axis_b = base_rot.T @ axis

        subtree = [i for i, chain in enumerate(chains) if child in chain]
        sub_mass = link_mass[subtree].sum()
        sub_com = (link_mass[subtree, None] * com_world[subtree]).sum(axis=0) / max(sub_mass, 1e-12)
        about_axis = 0.0
        for i in subtree:
            rotation = link_rot[i]
            inertia_world = rotation @ inertia_link[i] @ rotation.T
            lever = np.cross(com_world[i] - origin, axis)
            about_axis += axis @ inertia_world @ axis + link_mass[i] * lever @ lever

        lower, upper = np.clip(limits[j], -POSITION_LIMIT_CLIP, POSITION_LIMIT_CLIP)
        rows.append(
            [
                *position_b,
                *axis_b,
                lower,
                upper,
                default[j],
                min(drive["effort_limit"] / weight_length, EFFORT_LIMIT_CLIP),
                min(min(drive["velocity_limit"], speed_limits[j]) * time, VELOCITY_LIMIT_CLIP),
                drive["stiffness"] / weight_length,
                drive["damping"] / (weight_length * time),
                (drive["armature"] or 0.0) / (mass * length**2),
                drive["friction"] or 0.0,
                sub_mass / mass,
                *(base_rot.T @ (sub_com - origin) / length),
                about_axis / (mass * length**2),
            ]
        )
    joints = np.asarray(rows, dtype=float)

    count = len(joint_names)
    distance = np.zeros((count, count), dtype=int)
    ancestor = np.zeros((count, count), dtype=bool)
    for a in range(count):
        chain_a = chains[child_links[a]]
        for b in range(count):
            chain_b = chains[child_links[b]]
            common = next(link for link in chain_a if link in set(chain_b))
            distance[a, b] = chain_a.index(common) + chain_b.index(common)
            # The degrees of freedom of one multi-axis joint (Digit's spherical rod ends,
            # ``name:0`` to ``name:2``) share their child link: one joint, no ancestry
            # between them, at distance 0.
            ancestor[a, b] = child_links[a] != child_links[b] and child_links[a] in chain_b

    control = card["control"]
    feet = card["foot_bodies"]
    foot = card["foot"][feet[0]]
    scale = control.get("action_scale", 1.0)
    body = np.array(
        [
            control["control_period"] / time,
            control["physics_dt"] / time,
            float(scale) if isinstance(scale, (int, float)) else np.nan,
            card["postures"]["default"]["com_height_above_soles"] / length,
            foot["length"] / length,
            foot["width"] / length,
            zero["sole_lateral_separation"] / length,
        ],
        dtype=float,
    )
    return BodyDescriptor(
        name=str(card.get("body", "body")),
        joint_names=tuple(joint_names),
        joints=joints,
        body=body,
        tree_distance=distance,
        ancestor=ancestor,
        leg_length=length,
        mass=mass,
    )


def contact_joints(card: dict[str, Any]) -> list[int]:
    """Per foot (card order: left, right), the joint whose child link is that foot."""
    child = {row["child_link"]: row["joint"] for row in card["joint_geometry"]}
    names = list(card["joints"]["names"])
    missing = [foot for foot in card["foot_bodies"] if foot not in child]
    if missing:
        raise KeyError(f"no joint moves the feet {missing} directly")
    return [names.index(child[foot]) for foot in card["foot_bodies"]]
