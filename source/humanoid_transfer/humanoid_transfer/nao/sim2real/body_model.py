"""The body model the robot's estimator runs on, built from the URDF in the simulator's order.

:class:`~humanoid_transfer.nao.deploy.kinematics.BodyModel` reads a plain
dictionary; this builds it. The bodies are the simulator's (fixed frames merged
into their parents, as the USD importer merges them), in the simulator's order,
so the robot's ``z*`` indexes the same bodies as the one the tracker trained on.
Masses and inertias are the merged links' from the URDF. The URDF is a robot
asset and is never committed; neither is the model derived from it, which goes
under ``nao/assets/generated/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ...common.similarity import UnitSystem
from ...common.skill_state import SKILL_LAYOUT
from ..assets import nao_kinematics as nk


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _merged(model: nk.UrdfModel, root: str, kept: set[str]) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia about it, in ``root``'s frame, of ``root`` and every
    link welded below it that is not a body of its own."""
    parts: list[tuple[float, np.ndarray, np.ndarray]] = []
    stack = [(root, np.eye(4))]
    while stack:
        name, pose = stack.pop()
        link = model.links[name]
        if link.mass > 0.0:
            rotation = pose[:3, :3] @ link.inertia_rot
            parts.append(
                (
                    link.mass,
                    (pose @ np.append(link.com, 1.0))[:3],
                    rotation @ link.inertia @ rotation.T,
                )
            )
        for joint in model.joints:
            if joint.parent == name and joint.type == "fixed" and joint.child not in kept:
                stack.append((joint.child, pose @ _transform(joint.origin_rot, joint.origin_xyz)))
    mass = sum(m for m, _, _ in parts)
    if mass == 0.0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(m * c for m, c, _ in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in parts:
        d = c - com
        inertia += i + m * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def body_model(
    body_names: Sequence[str],
    joint_names: Sequence[str],
    sole_points: Sequence[Sequence[float]],
    sole_corners: Sequence[Sequence[Sequence[float]]],
    leg_length: float,
    base_body: str = "base_link",
    foot_bodies: Sequence[str] = nk.FOOT_BODIES,
    urdf_path: Path | None = None,
) -> dict:
    """The dictionary :class:`~...deploy.kinematics.BodyModel` is built from.

    ``body_names`` and ``joint_names`` are the simulator's, in its order; the measured
    joints are those that are not a mimic of another (one motor drives both hip
    yaw-pitches; the hand drives its fingers). ``sole_points`` and ``sole_corners`` are
    in each foot's link frame, as the tracking command takes them.
    """
    model = nk.load_model(urdf_path)
    by_child = {joint.child: joint for joint in model.joints}
    by_name = {joint.name: joint for joint in model.joints}
    index = {name: i for i, name in enumerate(body_names)}
    kept = set(body_names)
    measured = [n for n in joint_names if by_name[n].type != "fixed" and by_name[n].mimic is None]
    bodies = []
    for name in body_names:
        mass, com, inertia = _merged(model, name, kept)
        if name == base_body:
            parent, origin, axis, source = -1, np.eye(4), np.zeros(3), None
        else:
            joint = by_child[name]
            if joint.type == "fixed":
                raise ValueError(f"body {name} is welded to its parent; the simulator merges it")
            origin = _transform(joint.origin_rot, joint.origin_xyz)
            above = joint.parent
            while above not in index:
                welded = by_child[above]
                if welded.type != "fixed":
                    raise ValueError(f"{above} moves but is not a body of the simulator")
                origin = _transform(welded.origin_rot, welded.origin_xyz) @ origin
                above = welded.parent
            parent, axis, source = index[above], joint.axis, joint.mimic or joint.name
        bodies.append(
            {
                "name": name,
                "parent": parent,
                "joint": source,
                "axis": [float(v) for v in axis],
                "origin": origin.tolist(),
                "mass": float(mass),
                "com": com.tolist(),
                "inertia": inertia.tolist(),
            }
        )
    feet = [index[f] for f in foot_bodies]
    foot_com = [np.array(bodies[f]["com"]) for f in feet]
    total = float(sum(b["mass"] for b in bodies))
    units = UnitSystem(leg_length, total)
    scales = []
    for field in SKILL_LAYOUT.fields:
        scales += [units.scale(field.dimension)] * field.size
    return {
        "bodies": bodies,
        "joints": measured,
        "base": index[base_body],
        "feet": feet,
        "sole_from_com": [
            (np.asarray(p) - c).tolist() for p, c in zip(sole_points, foot_com, strict=True)
        ],
        "corners": [
            (np.asarray(k) - c).tolist() for k, c in zip(sole_corners, foot_com, strict=True)
        ],
        "ground_height": 0.0,
        "skill_fields": [[field.name, field.size] for field in SKILL_LAYOUT.fields],
        "skill_scales": scales,
        "force_scale": total * units.gravity,
        "leg_length": float(leg_length),
        "total_mass": total,
    }


__all__ = ["body_model"]
