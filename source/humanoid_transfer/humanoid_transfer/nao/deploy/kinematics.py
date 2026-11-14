"""The NAO's rigid bodies and the joints between them, as the robot evaluates them.

A :class:`BodyModel` is built once from a plain dictionary (written to JSON by
:func:`humanoid_transfer.nao.sim2real.body_model.body_model`, from the URDF and
the simulator's body order) and gives every body's pose in the base frame from
the measured joint angles. Each body hangs from its parent by one revolute
joint, or is welded to it; fixed frames between them are folded into the
joint's origin. The bodies keep the simulator's order, so ``z*`` computed here
and in the simulator index the same bodies.

Every function takes a leading batch: one robot is a batch of one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def axis_rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotations by ``angle`` ``(...)`` about the unit ``axis`` ``(3,)``."""
    x, y, z = (float(v) for v in axis)
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s = np.sin(angle)[..., None, None]
    c = np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


class BodyModel:
    """Bodies in the simulator's order, their joints, masses and the feet's soles."""

    def __init__(self, spec: dict) -> None:
        bodies = spec["bodies"]
        self.names: list[str] = [b["name"] for b in bodies]
        self.parent = np.array([b["parent"] for b in bodies], dtype=np.int64)
        origin = np.array([b["origin"] for b in bodies], dtype=np.float64)
        self.origin_rot = origin[:, :3, :3]
        self.origin_pos = origin[:, :3, 3]
        axis = np.array([b["axis"] for b in bodies], dtype=np.float64)
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        # URDF axes are written to six digits (the hip yaw-pitch's is 0.707107 twice).
        self.axis = np.where(norm > 0.0, axis / np.where(norm > 0.0, norm, 1.0), axis)
        self.joint_names: list[str] = list(spec["joints"])
        """The order measured joint angles are given in."""
        self.source = np.array(
            [-1 if b["joint"] is None else self.joint_names.index(b["joint"]) for b in bodies],
            dtype=np.int64,
        )
        """Which measured angle turns each body (a mimic reads its driver's), -1 if welded."""
        self.mass = np.array([b["mass"] for b in bodies], dtype=np.float64)
        self.com = np.array([b["com"] for b in bodies], dtype=np.float64)
        """``(B, 3)`` each body's centre of mass in its link frame."""
        self.inertia = np.array([b["inertia"] for b in bodies], dtype=np.float64)
        """``(B, 3, 3)`` about each body's centre of mass, in its link frame."""
        self.base = int(spec["base"])
        self.feet = tuple(int(f) for f in spec["feet"])
        self.sole_from_com = np.array(spec["sole_from_com"], dtype=np.float64)
        """``(2, 3)`` from each foot's centre of mass to its sole point, foot link frame."""
        self.corners = np.array(spec["corners"], dtype=np.float64)
        """``(2, K, 3)`` each foot's sole corners from its centre of mass, foot link frame."""
        self.ground_height = float(spec.get("ground_height", 0.0))
        self.scales = np.array(spec["skill_scales"], dtype=np.float64)
        """``(27,)`` the unit each entry of ``z*`` is divided by."""
        self.force_scale = float(spec["force_scale"])
        """The unit of the feet's loads, newtons (``m g``)."""
        self.fields: list[tuple[str, int]] = [(str(n), int(s)) for n, s in spec["skill_fields"]]
        """``z*``'s fields and sizes, in order."""
        self.order = _evaluation_order(self.parent, self.base)

    @classmethod
    def load(cls, path: str | Path) -> BodyModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum())

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each body's link rotation ``(..., B, 3, 3)`` and origin ``(..., B, 3)`` in the base
        frame, from the measured angles ``q`` ``(..., J)`` in :attr:`joint_names`' order."""
        q = np.asarray(q, dtype=np.float64)
        lead = q.shape[:-1]
        count = len(self.names)
        rot = np.zeros((*lead, count, 3, 3))
        pos = np.zeros((*lead, count, 3))
        rot[..., self.base, :, :] = np.eye(3)
        for b in self.order:
            p = self.parent[b]
            local = np.broadcast_to(self.origin_rot[b], (*lead, 3, 3))
            if self.source[b] >= 0:
                local = local @ axis_rotation(self.axis[b], q[..., self.source[b]])
            rot[..., b, :, :] = rot[..., p, :, :] @ local
            pos[..., b, :] = pos[..., p, :] + np.einsum(
                "...ij,j->...i", rot[..., p, :, :], self.origin_pos[b]
            )
        return rot, pos

    def centres(self, rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """``(..., B, 3)`` each body's centre of mass, in the frame ``rot`` and ``pos`` are in."""
        return pos + np.einsum("...bij,bj->...bi", rot, self.com)


def _evaluation_order(parent: np.ndarray, base: int) -> list[int]:
    """Every body but the base, each after its parent."""
    order: list[int] = []
    placed = {base}
    pending = [b for b in range(len(parent)) if b != base]
    while pending:
        ready = [b for b in pending if int(parent[b]) in placed]
        if not ready:
            raise ValueError(f"bodies {pending} do not hang from the base")
        order += ready
        placed.update(ready)
        pending = [b for b in pending if b not in placed]
    return order


__all__ = ["BodyModel", "axis_rotation"]
