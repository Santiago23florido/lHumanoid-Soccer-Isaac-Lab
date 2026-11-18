"""Forward kinematics of any measured body, from its card alone.

A card (``transfer/scripts/measure_body.py``) records every joint's axis and origin
and every link's pose and centre of mass in the zero pose, with the root fixed
upright at the origin. That is all the product-of-exponentials formula needs: a
point attached to link ``i`` moves, for joint angles ``q``, by

    G_i(q) = exp(xi_1 q_1) ... exp(xi_k q_k)        (joints from the root to i)

where ``xi_j`` is the rotation about joint ``j``'s axis through its origin. So a
body measured once in the simulator can be posed offline (a tracker's crouch, a
descriptor's self-modelling target) with no URDF and no simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _screw(axis: np.ndarray, origin: np.ndarray, angle: float, kind: str) -> np.ndarray:
    """``4 x 4`` rigid motion of a joint turned (or slid) by ``angle`` from the zero pose."""
    out = np.eye(4)
    if kind == "prismatic":
        out[:3, 3] = axis * angle
        return out
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    out[:3, :3] = rotation
    out[:3, 3] = (np.eye(3) - rotation) @ origin
    return out


class CardKinematics:
    """Pose a measured body at any joint angles."""

    def __init__(self, card: dict[str, Any]) -> None:
        self.links = list(card["bodies"]["names"])
        parents = list(card["bodies"]["parents"])
        index = {name: i for i, name in enumerate(self.links)}
        self.parent = [index.get(p, -1) for p in parents]
        self.joint_names = list(card["joints"]["names"])
        joint_index = {name: i for i, name in enumerate(self.joint_names)}
        self.joint_of_link = [-1] * len(self.links)
        self.axis = np.zeros((len(self.joint_names), 3))
        self.origin = np.zeros((len(self.joint_names), 3))
        self.kind = ["revolute"] * len(self.joint_names)
        for row in card["joint_geometry"]:
            j = joint_index[row["joint"]]
            self.joint_of_link[index[row["child_link"]]] = j
            self.axis[j] = row["axis_root"]
            self.origin[j] = row["position_root"]
            self.kind[j] = row["kind"]

        zero = card["postures"]["zero"]
        self.link_pos0 = np.asarray(zero["link_pos"], dtype=float)
        self.link_rot0 = np.stack([_quat_matrix(q) for q in zero["link_quat_wxyz"]])
        self.com0 = np.asarray(zero["com_pos"], dtype=float)
        self.mass = np.asarray(card["bodies"]["mass"], dtype=float)
        self.feet = [index[f] for f in card["foot_bodies"]]
        self.sole_local = np.array(
            [card["foot"][f]["sole_point_link"] for f in card["foot_bodies"]], dtype=float
        )
        self.default = np.asarray(card["joints"]["default_pos"], dtype=float)

        # Links ordered so every parent comes before its children.
        order, placed = [], set()
        while len(order) < len(self.links):
            for i, p in enumerate(self.parent):
                if i not in placed and (p == -1 or p in placed):
                    order.append(i)
                    placed.add(i)
        self.order = order

    def motions(self, q: np.ndarray) -> np.ndarray:
        """``(links, 4, 4)``: the rigid motion of every link from the zero pose."""
        q = np.asarray(q, dtype=float)
        motion = np.zeros((len(self.links), 4, 4))
        for i in self.order:
            base = np.eye(4) if self.parent[i] == -1 else motion[self.parent[i]]
            j = self.joint_of_link[i]
            if j >= 0:
                base = base @ _screw(self.axis[j], self.origin[j], q[j], self.kind[j])
            motion[i] = base
        return motion

    def _move(self, motion: np.ndarray, point: np.ndarray) -> np.ndarray:
        return motion[:3, :3] @ point + motion[:3, 3]

    def link_positions(self, q: np.ndarray) -> np.ndarray:
        motions = self.motions(q)
        return np.stack([self._move(m, p) for m, p in zip(motions, self.link_pos0, strict=True)])

    def com(self, q: np.ndarray) -> np.ndarray:
        motions = self.motions(q)
        coms = np.stack([self._move(m, c) for m, c in zip(motions, self.com0, strict=True)])
        return (self.mass[:, None] * coms).sum(axis=0) / self.mass.sum()

    def soles(self, q: np.ndarray) -> np.ndarray:
        """``(2, 3)``: each foot's sole point."""
        motions = self.motions(q)
        out = []
        for foot, local in zip(self.feet, self.sole_local, strict=True):
            zero = self.link_rot0[foot] @ local + self.link_pos0[foot]
            out.append(self._move(motions[foot], zero))
        return np.stack(out)

    def com_height_above_soles(self, q: np.ndarray) -> float:
        return float(self.com(q)[2] - self.soles(q)[:, 2].min())


def sagittal_crouch(
    card: dict[str, Any],
    chains: list[tuple[str, str, str]],
    target: float,
    leg_length: float | None = None,
    kinematics: CardKinematics | None = None,
) -> dict[str, float]:
    """Joint angles that lower the CoM to ``target`` leg lengths above the soles.

    Each ``(hip, knee, ankle)`` pitch chain bends with hip and ankle at minus half the
    knee about parallel axes, which keeps the torso upright and the foot flat (the
    rule the NAO's crouch follows). The other joints stay at the card's default. The
    knee angle is found by bisection within every joint's limits.
    """
    kinematics = kinematics or CardKinematics(card)
    length = float(leg_length or card["leg_length"])
    names = kinematics.joint_names
    limits = np.asarray(card["joints"]["pos_limits"], dtype=float)
    sign = {
        name: float(np.sign(kinematics.axis[names.index(name)][1])) for c in chains for name in c
    }

    def pose(knee: float) -> np.ndarray:
        q = kinematics.default.copy()
        for hip_name, knee_name, ankle_name in chains:
            bend = knee * sign[knee_name]
            q[names.index(knee_name)] = knee
            q[names.index(hip_name)] = -bend / 2 * sign[hip_name]
            q[names.index(ankle_name)] = -bend / 2 * sign[ankle_name]
        return q

    def feasible(knee: float) -> bool:
        q = pose(knee)
        return bool(np.all((q >= limits[:, 0] - 1e-9) & (q <= limits[:, 1] + 1e-9)))

    def height(knee: float) -> float:
        return kinematics.com_height_above_soles(pose(knee)) / length

    low, high = 0.0, 0.0
    while feasible(high + 0.01) and high < np.pi:
        high += 0.01
    if height(low) < target or height(high) > target:
        raise ValueError(
            f"target {target:.3f} L is outside the reachable {height(high):.3f}-{height(low):.3f} L"
        )
    for _ in range(50):
        middle = (low + high) / 2
        low, high = (middle, high) if height(middle) > target else (low, middle)
    q = pose((low + high) / 2)
    return {name: float(q[names.index(name)]) for c in chains for name in c}
