"""The parts of ``z*`` a body's own sensors and measured kinematics give exactly.

From the card (joint axes and origins, link masses, inertias and centres in the
zero pose), the joint angles and speeds, the IMU's projected gravity and base
angular velocity, these follow without any learning:

- ``gravity`` and ``base_angular_velocity``: the IMU itself;
- ``foot_position``: each sole relative to the centre of mass, in the heading frame;
- ``foot_velocity`` relative to the centre of mass's velocity, in the heading frame;
- ``angular_momentum`` about the centre of mass, in the heading frame: it depends on
  velocities relative to the centre of mass only, so the base's unknown linear
  velocity cancels;
- ``height`` above the lowest sole, which is the height above the ground whenever a
  foot touches it.

What remains for a learned estimator is the centre of mass's velocity (and from it
each foot's absolute velocity), the feet's loads, and the height in flight. That is
the split PACT's anchors use: physics where it is exact, learning where it is not.

The heading frame is the base's yaw (ZYX), as in
:func:`~humanoid_transfer.transfer.latent.skill_torch.skill_state`; with the yaw
removed, a base-frame vector reaches it through the roll-pitch rotation, which the
projected gravity alone determines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from humanoid_transfer.common.similarity import UnitSystem
from humanoid_transfer.common.skill_state import SKILL_LAYOUT

EXACT_FIELDS: tuple[str, ...] = (
    "gravity",
    "base_angular_velocity",
    "foot_position",
    "angular_momentum",
)
"""Fields equal to the true ``z*`` up to the card's accuracy, in any state."""


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def tilt_from_gravity(gravity_b: torch.Tensor) -> torch.Tensor:
    """``(N, 3) -> (N, 3, 3)``: the roll-pitch rotation (ZYX, zero yaw) of the base.

    Its third row is minus the projected gravity: ``sin(pitch) = g_x``,
    ``tan(roll) = g_y / g_z``.
    """
    g = gravity_b / gravity_b.norm(dim=-1, keepdim=True)
    pitch = torch.asin(g[:, 0].clamp(-1.0, 1.0))
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    cp, sp, cr, sr = torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)
    zero = torch.zeros_like(cp)
    r_y = torch.stack(
        [
            torch.stack([cp, zero, sp], -1),
            torch.stack([zero, torch.ones_like(cp), zero], -1),
            torch.stack([-sp, zero, cp], -1),
        ],
        -2,
    )
    r_x = torch.stack(
        [
            torch.stack([torch.ones_like(cr), zero, zero], -1),
            torch.stack([zero, cr, -sr], -1),
            torch.stack([zero, sr, cr], -1),
        ],
        -2,
    )
    return r_y @ r_x


class KinematicAnchors(torch.nn.Module):
    """Exact ``z*`` fields of one body from its card and its sensors (no parameters)."""

    def __init__(
        self,
        card: dict[str, Any],
        leg_length: float | None = None,
        sole_points: list[list[float]] | None = None,
    ) -> None:
        super().__init__()
        links = list(card["bodies"]["names"])
        if links[0] != card["base_body"]:
            raise ValueError(f"the root link {links[0]} is not the base {card['base_body']}")
        index = {name: i for i, name in enumerate(links)}
        parent = [index.get(p, -1) for p in card["bodies"]["parents"]]
        joints = list(card["joints"]["names"])
        # A link moved by several degrees of freedom (Digit's spherical rod ends, ``name:0``
        # to ``name:2``) has them all, applied in card order as a serial chain at one origin.
        joints_of: list[list[int]] = [[] for _ in links]
        axis = np.zeros((len(joints), 3))
        origin = np.zeros((len(joints), 3))
        for row in card["joint_geometry"]:
            j = joints.index(row["joint"])
            if row["kind"] != "revolute":
                raise ValueError(f"{row['joint']}: only revolute joints are supported")
            joints_of[index[row["child_link"]]].append(j)
            axis[j], origin[j] = row["axis_root"], row["position_root"]
        order, placed = [], set()
        while len(order) < len(links):
            for i, p in enumerate(parent):
                if i not in placed and (p == -1 or p in placed):
                    order.append(i)
                    placed.add(i)
        self.order, self.parent, self.joints_of = order, parent, joints_of

        zero = card["postures"]["zero"]
        rot0 = np.stack([_quat_matrix(q) for q in zero["link_quat_wxyz"]])
        feet = [index[f] for f in card["foot_bodies"]]
        soles = sole_points or [card["foot"][f]["sole_point_link"] for f in card["foot_bodies"]]
        sole0 = np.stack(
            [
                rot0[f] @ np.asarray(s) + np.asarray(zero["link_pos"][f])
                for f, s in zip(feet, soles, strict=True)
            ]
        )
        f32 = torch.float32
        self.register_buffer("axis", torch.as_tensor(axis, dtype=f32))
        self.register_buffer("origin", torch.as_tensor(origin, dtype=f32))
        self.register_buffer("rot0", torch.as_tensor(rot0, dtype=f32))
        self.register_buffer("com0", torch.as_tensor(np.asarray(zero["com_pos"]), dtype=f32))
        self.register_buffer("sole0", torch.as_tensor(sole0, dtype=f32))
        self.register_buffer("mass", torch.as_tensor(card["bodies"]["mass"], dtype=f32))
        self.register_buffer(
            "inertia", torch.as_tensor(np.asarray(card["bodies"]["inertia_link"]), dtype=f32)
        )
        self.feet = feet
        self.units = UnitSystem(float(leg_length or card["leg_length"]), float(card["total_mass"]))

    def _rodrigues(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """``axis (N, 3)``, ``angle (N,)`` -> ``(N, 3, 3)``."""
        x, y, z = axis.unbind(-1)
        zero = torch.zeros_like(x)
        k = torch.stack(
            [
                torch.stack([zero, -z, y], -1),
                torch.stack([z, zero, -x], -1),
                torch.stack([-y, x, zero], -1),
            ],
            -2,
        )
        s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
        eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(k)
        return eye + s * k + (1 - c) * (k @ k)

    def chain(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward kinematics in the base frame: per link its motion from the zero pose
        (``rot``, ``trans``), its angular velocity relative to the base (``omega``) and the
        velocity of the base-frame origin attached to it (``v0``); per joint its axis and a
        point on it in the current pose (``axis``, ``origin``, ``(N, J, 3)``)."""
        n = joint_pos.shape[0]
        device, dtype = joint_pos.device, joint_pos.dtype
        links = len(self.parent)
        rot = [None] * links  # rotation of each link's motion from the zero pose
        trans = [None] * links
        omega = [None] * links  # angular velocity relative to the base (base frame)
        v0 = [None] * links  # velocity of the base-frame origin as if attached to the link
        eye = torch.eye(3, device=device, dtype=dtype).expand(n, 3, 3)
        zero3 = torch.zeros(n, 3, device=device, dtype=dtype)
        axes: list[torch.Tensor | None] = [None] * self.axis.shape[0]
        origins: list[torch.Tensor | None] = [None] * self.axis.shape[0]
        for i in self.order:
            p = self.parent[i]
            r_p, t_p = (eye, zero3) if p == -1 else (rot[p], trans[p])
            w_p, v_p = (zero3, zero3) if p == -1 else (omega[p], v0[p])
            for j in self.joints_of[i]:
                a = torch.einsum("nij,j->ni", r_p, self.axis[j])
                o = torch.einsum("nij,j->ni", r_p, self.origin[j]) + t_p
                spin = self._rodrigues(a, joint_pos[:, j])
                r_p = spin @ r_p
                t_p = torch.einsum("nij,nj->ni", spin, t_p - o) + o
                rate = joint_vel[:, j : j + 1]
                w_p = w_p + a * rate
                v_p = v_p - torch.linalg.cross(a, o, dim=-1) * rate
                axes[j], origins[j] = a, o
            rot[i], trans[i], omega[i], v0[i] = r_p, t_p, w_p, v_p
        zero_j = torch.zeros(n, 3, device=device, dtype=dtype)
        return {
            "rot": torch.stack(rot, 1),  # (N, links, 3, 3)
            "trans": torch.stack(trans, 1),
            "omega": torch.stack(omega, 1),
            "v0": torch.stack(v0, 1),
            "axis": torch.stack([zero_j if a is None else a for a in axes], 1),
            "origin": torch.stack([zero_j if o is None else o for o in origins], 1),
        }

    def forward(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        gravity_b: torch.Tensor,
        ang_vel_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Dimensionless fields; ``foot_velocity_relative`` is each sole's velocity minus
        the centre of mass's, in the heading frame (add the CoM velocity for ``z*``'s)."""
        kin = self.chain(joint_pos, joint_vel)
        rot_t, trans_t, omega_t, v0_t = kin["rot"], kin["trans"], kin["omega"], kin["v0"]

        com_links = torch.einsum("nlij,lj->nli", rot_t, self.com0) + trans_t
        total = self.mass.sum()
        com = torch.einsum("l,nli->ni", self.mass, com_links) / total
        vel_links = v0_t + torch.linalg.cross(omega_t, com_links, dim=-1)
        # The base's rotation moves every point too: add omega_b x r (its linear velocity cancels).
        vel_links = vel_links + torch.linalg.cross(
            ang_vel_b[:, None].expand_as(com_links), com_links, dim=-1
        )
        com_vel = torch.einsum("l,nli->ni", self.mass, vel_links) / total
        spin_total = omega_t + ang_vel_b[:, None]
        link_rot = rot_t @ self.rot0
        inertia = link_rot @ self.inertia @ link_rot.transpose(-1, -2)
        momentum = torch.einsum("nlij,nlj->ni", inertia, spin_total) + torch.linalg.cross(
            com_links - com[:, None],
            self.mass[None, :, None] * (vel_links - com_vel[:, None]),
            dim=-1,
        ).sum(1)

        feet = self.feet
        soles = torch.einsum("nfij,fj->nfi", rot_t[:, feet], self.sole0) + trans_t[:, feet]
        sole_vel = v0_t[:, feet] + torch.linalg.cross(
            omega_t[:, feet] + ang_vel_b[:, None], soles, dim=-1
        )
        tilt = tilt_from_gravity(gravity_b)
        to_heading = lambda v: torch.einsum("nij,n...j->n...i", tilt, v)  # noqa: E731

        u = self.units

        def dimensionless(value: torch.Tensor, field: str) -> torch.Tensor:
            scale = u.to_dimensionless(np.ones(1), SKILL_LAYOUT.field(field).dimension)[0]
            return value * float(scale)

        feet_h = to_heading(soles - com[:, None])
        com_h = to_heading(com)
        soles_h = to_heading(soles)
        return {
            "height": dimensionless(
                (com_h[:, 2] - soles_h[..., 2].min(dim=1).values)[:, None], "height"
            ),
            "gravity": gravity_b / gravity_b.norm(dim=-1, keepdim=True),
            "base_angular_velocity": dimensionless(ang_vel_b, "base_angular_velocity"),
            "foot_position": dimensionless(feet_h, "foot_position"),
            "foot_velocity_relative": dimensionless(
                to_heading(sole_vel - com_vel[:, None]), "foot_velocity"
            ),
            "angular_momentum": dimensionless(to_heading(momentum), "angular_momentum"),
        }


class DriveLoads(torch.nn.Module):
    """Each foot's load from the drives' torques: the static balance of each leg.

    The card gives every joint's drive (an implicit PD: stiffness, damping, effort limit)
    and the action's scale and offset, so the torque each joint applies follows from the
    commanded action and the joint's state. Below a joint, the leg is held by that torque,
    its own links' weights and the ground's wrench on the foot (inertia neglected):

        a_j . [ sum_l (c_l - o_j) x m_l g + (p - o_j) x F + M ] = -tau_j,

    for every driven joint ``j`` on the chain from the base to the foot, with ``F`` and ``M``
    the ground's force and moment at the sole point ``p``. A joint without a drive (zero
    stiffness and damping: a mimic joint, as the NAO's ``RHipYawPitch``, or a closed loop's
    passive joint, as Digit's tarsus and toes) is held by a constraint whose torque the drives
    do not give, so its balance is not used. A ridge least-squares solve per foot gives the
    wrench; the load in ``z*`` is ``F``'s vertical part over ``m g``. No
    parameter is learned: it is as good as the drive model and the statics.
    """

    def __init__(
        self,
        card: dict[str, Any],
        anchors: KinematicAnchors,
        ridge: float = 1e-6,
        moment_prior: float = 1e-2,
        action_joints: list[str] | None = None,
    ) -> None:
        """``action_joints``: the joints the action drives, in its order (all by default);
        the others hold their default position. ``ridge`` keeps the force solve defined;
        ``moment_prior`` pulls the ground's moment towards zero (the centre of pressure
        towards the sole point), so a leg with fewer than six joints still has one wrench."""
        super().__init__()
        self.anchors = anchors
        names = list(card["joints"]["names"])
        drives = {d["joint"]: d for d in card["drives"]}
        f32 = torch.float32
        column = lambda key: torch.as_tensor(  # noqa: E731
            [float(drives[name][key]) for name in names], dtype=f32
        )
        self.register_buffer("stiffness", column("stiffness"))
        self.register_buffer("damping", column("damping"))
        self.register_buffer("effort", column("effort_limit"))
        self.register_buffer("default", torch.as_tensor(card["joints"]["default_pos"], dtype=f32))
        control = card["control"]
        self.scale = float(control["action_scale"])
        self.offset = bool(control.get("action_use_default_offset", True))
        self.ridge = ridge
        self.moment_prior = moment_prior
        acted = action_joints or names
        self.register_buffer(
            "action_index", torch.as_tensor([names.index(j) for j in acted], dtype=torch.long)
        )
        links, parent, joints_of = len(anchors.parent), anchors.parent, anchors.joints_of
        # below[j, l]: link l moves with joint j (it is in the subtree of j's child link)
        below = np.zeros((len(names), links), dtype=np.float32)
        for leaf in range(links):
            node = leaf
            while node != -1:
                for j in joints_of[node]:
                    below[j, leaf] = 1.0
                node = parent[node]
        self.register_buffer("below", torch.as_tensor(below))
        chains = []
        for foot in anchors.feet:
            on = np.zeros(len(names), dtype=np.float32)
            node = foot
            while node != -1:
                for j in joints_of[node]:
                    on[j] = 1.0
                node = parent[node]
            chains.append(on)
        self.register_buffer("on_chain", torch.as_tensor(np.stack(chains)))  # (2, J)

    def set_drives(
        self, stiffness: torch.Tensor, damping: torch.Tensor, effort: torch.Tensor
    ) -> None:
        """Use the drives' gains and limits as a capture recorded them (the task's own)."""
        for name, value in (("stiffness", stiffness), ("damping", damping), ("effort", effort)):
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def drive_ratio(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """The torque each target commands, as a fraction of the joint's limit and softly
        saturated: ``tanh((kp (T - q) - kd qd) / tau_max)``, 0 for a joint without a drive
        (``pact_gate_d.drive_ratio``, what the shared models were trained on)."""
        active = (self.stiffness > 0) & (self.effort > 0)
        limit = torch.where(active, self.effort, torch.ones_like(self.effort))
        ratio = (self.stiffness * (target - joint_pos) - self.damping * joint_vel) / limit
        return torch.where(active, torch.tanh(ratio), torch.zeros_like(ratio))

    def torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        action: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(N, J)``: the torque each implicit PD drive applies, from the joints' PD targets
        (``target``, as recorded) or from the action that set them."""
        if target is None:
            command = torch.zeros_like(joint_pos)
            command[:, self.action_index] = self.scale * action
            target = command + (self.default if self.offset else 0.0)
        tau = self.stiffness * (target - joint_pos) - self.damping * joint_vel
        return torch.maximum(torch.minimum(tau, self.effort), -self.effort)

    def balance(self, joint_pos: torch.Tensor, gravity_b: torch.Tensor) -> dict[str, torch.Tensor]:
        """The legs' statics in this pose: per joint the torque its carried links' weight
        exerts (``gravity_tau``, ``(N, J)``), per foot the map from the ground's wrench at the
        sole to joint torques (``wrench_map``, ``(N, 2, J, 6)``, zero off the foot's chain)."""
        anchors = self.anchors
        kin = anchors.chain(joint_pos, torch.zeros_like(joint_pos))
        rot, trans = kin["rot"], kin["trans"]
        axis, origin = kin["axis"], kin["origin"]  # (N, J, 3)
        com = torch.einsum("nlij,lj->nli", rot, anchors.com0) + trans  # (N, L, 3)
        g = 9.81 * gravity_b / gravity_b.norm(dim=-1, keepdim=True)  # (N, 3), base frame
        weight = anchors.mass[None, :, None] * g[:, None, :]  # (N, L, 3)
        arm = com[:, None, :, :] - origin[:, :, None, :]  # (N, J, L, 3)
        moment = torch.linalg.cross(arm, weight[:, None].expand_as(arm), dim=-1)
        gravity_tau = torch.einsum("njli,jl,nji->nj", moment, self.below, axis)
        soles = (
            torch.einsum("nfij,fj->nfi", rot[:, anchors.feet], anchors.sole0)
            + trans[:, anchors.feet]
        )
        maps = []
        for f in range(len(anchors.feet)):
            lever = torch.linalg.cross(axis, soles[:, f, None, :] - origin, dim=-1)
            maps.append(torch.cat([lever, axis], dim=-1) * self.on_chain[f][None, :, None])
        return {"gravity_tau": gravity_tau, "wrench_map": torch.stack(maps, 1), "down": g / 9.81}

    def forward(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        gravity_b: torch.Tensor,
        action: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``loads (N, 2)`` in ``m g``; ``force`` and ``moment`` ``(N, 2, 3)``, the ground's
        wrench at each sole point in the base frame (N, N m).

        ``target``: each joint's PD target as the drives hold it when the state is read
        (the previous action's); or ``action``, the action that set them.
        """
        statics = self.balance(joint_pos, gravity_b)
        tau = self.torque(joint_pos, joint_vel, action, target)
        driven = ((self.stiffness > 0) | (self.damping > 0)).to(tau.dtype)
        forces, moments, loads = [], [], []
        for f in range(statics["wrench_map"].shape[1]):
            rows = self.on_chain[f] * driven  # (J,)
            a_matrix = statics["wrench_map"][:, f] * rows[None, :, None]  # (N, J, 6)
            b = -(tau + statics["gravity_tau"]) * rows[None, :]
            gram = a_matrix.transpose(1, 2) @ a_matrix
            diagonal = gram.diagonal(dim1=1, dim2=2)
            force_scale = diagonal[:, :3].mean(-1).clamp_min(1e-9)
            moment_scale = diagonal[:, 3:].mean(-1).clamp_min(1e-9)
            weights = torch.cat(
                [
                    (self.ridge * force_scale)[:, None].expand(-1, 3),
                    (self.moment_prior * moment_scale)[:, None].expand(-1, 3),
                ],
                dim=-1,
            )
            reg = torch.diag_embed(weights)
            wrench = torch.linalg.solve(gram + reg, a_matrix.transpose(1, 2) @ b[..., None])
            force = wrench[:, :3, 0]
            forces.append(force)
            moments.append(wrench[:, 3:, 0])
            weight = 9.81 * self.anchors.mass.sum()
            loads.append(-(force * statics["down"]).sum(-1) / weight)
        return {
            "loads": torch.stack(loads, -1),
            "force": torch.stack(forces, 1),
            "moment": torch.stack(moments, 1),
        }
