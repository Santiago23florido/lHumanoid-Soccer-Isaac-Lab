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
from humanoid_transfer.transfer.latent.pact.descriptors import EFFORT_LIMIT_CLIP, GRAVITY

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
        # Loop closures (``card_loops.py``): the point each joins on its two links, in the zero
        # pose; empty for a body whose legs are trees.
        loops = card.get("loops", [])
        ends = np.zeros((len(loops), 2, 3))
        loop_links = np.zeros((len(loops), 2), dtype=np.int64)
        for k, loop in enumerate(loops):
            for side in (0, 1):
                link = index[loop[f"body{side}"]]
                loop_links[k, side] = link
                ends[k, side] = rot0[link] @ np.asarray(loop[f"pos{side}"]) + np.asarray(
                    zero["link_pos"][link]
                )
        self.loop_links = loop_links
        f32 = torch.float32
        self.register_buffer("loop_ends0", torch.as_tensor(ends, dtype=f32))
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
    stiffness and damping: a mimic joint, as the NAO's ``RHipYawPitch``) is held by a
    constraint whose torque the drives do not give, so its balance is not used. A ridge
    least-squares solve per foot gives the wrench; the load in ``z*`` is ``F``'s vertical part
    over ``m g``. No parameter is learned: it is as good as the drive model and the statics.

    A card with loop closures (``card_loops.py``: Digit's achilles and toe rods) adds one
    unknown force per loop at its closing point, acting on its two links with opposite signs;
    the undriven joints on a loop's path (the rods' joints, the tarsus and toes) then balance
    zero, and both feet and every loop are solved together.
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
        weight_length = float(anchors.units.mass * GRAVITY * anchors.units.length)
        self.register_buffer("effort_cap", torch.tensor(EFFORT_LIMIT_CLIP * weight_length, dtype=f32))
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
        # A loop's force acts on its two links: +f on the first, -f on the second. A joint
        # carries it when exactly one of them moves with it (sign +1 or -1): those joints are
        # the loop's path, whose undriven ones are held by that force alone (torque 0).
        signs = np.zeros((len(names), len(anchors.loop_links)), dtype=np.float32)
        for k, (first, second) in enumerate(anchors.loop_links):
            signs[:, k] = below[:, first] - below[:, second]
        self.register_buffer("loop_sign", torch.as_tensor(signs))  # (J, K)
        self.register_buffer("on_loop", torch.as_tensor((signs != 0).any(1).astype(np.float32)))

    def set_drives(
        self, stiffness: torch.Tensor, damping: torch.Tensor, effort: torch.Tensor
    ) -> None:
        """Use the drives' gains and limits as a capture recorded them (the task's own)."""
        for name, value in (("stiffness", stiffness), ("damping", damping), ("effort", effort)):
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def effective_effort(self) -> torch.Tensor:
        """Per-joint effort scale for ``drive_ratio``: capped at ``EFFORT_LIMIT_CLIP * m g L``
        so unlimited drives (e.g. Digit's ``FLT_MAX``) stay on a physical torque scale."""
        cap = getattr(self, "effort_cap", None)
        return self.effort if cap is None else torch.minimum(self.effort, cap)

    def drive_ratio(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """The torque each target commands, as a fraction of the joint's effective limit and
        softly saturated: ``tanh((kp (T - q) - kd qd) / tau_eff)``, 0 for a joint without a
        drive (``pact_gate_d.drive_ratio``, what the shared models were trained on)."""
        effort = self.effective_effort()
        active = (self.stiffness > 0) & (effort > 0)
        limit = torch.where(active, effort, torch.ones_like(effort))
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
        out = {"gravity_tau": gravity_tau, "wrench_map": torch.stack(maps, 1), "down": g / 9.81}
        if self.loop_sign.shape[1]:
            links = anchors.loop_links
            ends = [
                torch.einsum("nkij,kj->nki", rot[:, links[:, side]], anchors.loop_ends0[:, side])
                + trans[:, links[:, side]]
                for side in (0, 1)
            ]
            point = 0.5 * (ends[0] + ends[1])  # (N, K, 3): where each loop closes
            lever = torch.linalg.cross(
                axis[:, :, None, :].expand(-1, -1, point.shape[1], -1),
                point[:, None, :, :] - origin[:, :, None, :],
                dim=-1,
            )  # (N, J, K, 3)
            out["loop_map"] = lever * self.loop_sign[None, :, :, None]
        return out

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
        if "loop_map" in statics:
            return self._closed(statics, tau, driven)
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

    def _closed(
        self, statics: dict[str, torch.Tensor], tau: torch.Tensor, driven: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The legs with loop closures, solved together: both feet's wrenches and every loop's
        force (unknown 3-vectors at the closing points). Rows: each driven joint balances its
        drive's torque; each undriven joint on a loop's path balances zero (the loop holds it);
        other undriven joints (mimics) are left out, as in the tree's solve."""
        free = (1.0 - driven) * self.on_loop
        rows = (driven + free).clamp(max=1.0)  # (J,)
        wrench_map, loop_map = statics["wrench_map"], statics["loop_map"]
        n, joints = loop_map.shape[:2]
        feet = wrench_map.shape[1]
        a_matrix = torch.cat(
            [wrench_map[:, f] for f in range(feet)] + [loop_map.reshape(n, joints, -1)], dim=-1
        ) * rows[None, :, None]  # (N, J, 6 * feet + 3K)
        b = -(tau * driven + statics["gravity_tau"]) * rows[None, :]
        gram = a_matrix.transpose(1, 2) @ a_matrix
        diagonal = gram.diagonal(dim1=1, dim2=2)
        weights = []
        for f in range(feet):
            force_scale = diagonal[:, 6 * f : 6 * f + 3].mean(-1).clamp_min(1e-9)
            moment_scale = diagonal[:, 6 * f + 3 : 6 * f + 6].mean(-1).clamp_min(1e-9)
            weights += [
                (self.ridge * force_scale)[:, None].expand(-1, 3),
                (self.moment_prior * moment_scale)[:, None].expand(-1, 3),
            ]
        loop_scale = diagonal[:, 6 * feet :].mean(-1).clamp_min(1e-9)
        weights.append((self.ridge * loop_scale)[:, None].expand(-1, diagonal.shape[1] - 6 * feet))
        reg = torch.diag_embed(torch.cat(weights, dim=-1))
        solution = torch.linalg.solve(gram + reg, a_matrix.transpose(1, 2) @ b[..., None])[..., 0]
        weight = 9.81 * self.anchors.mass.sum()
        forces = torch.stack([solution[:, 6 * f : 6 * f + 3] for f in range(feet)], 1)
        moments = torch.stack([solution[:, 6 * f + 3 : 6 * f + 6] for f in range(feet)], 1)
        loads = -(forces * statics["down"][:, None, :]).sum(-1) / weight
        return {"loads": loads, "force": forces, "moment": moments}

    def wrench_anchor(
        self, joint_pos: torch.Tensor, gravity_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dimensionless static inverse-dynamics anchor in the heading frame (in ``m g L`` units).

        Returns ``(tau_g, j_eff)`` with shapes ``(N, J)`` and ``(N, J, 12)`` such that for a
        heading-frame ground wrench ``w = [F_L/(mg), M_L/(mgL), F_R/(mg), M_R/(mgL)]``,
        static equilibrium satisfies:

            tau / (m g L) = -tau_g - j_eff @ w

        On closed-chain legs (e.g. Digit's achilles and toe rods), internal loop forces are
        eliminated analytically through the undriven loop joints' zero-torque equations, so
        ``j_eff`` maps foot wrenches directly onto driven joint torques.
        """
        statics = self.balance(joint_pos, gravity_b)
        length = float(self.anchors.units.length)
        mg_l = float(GRAVITY * self.anchors.mass.sum() * length)
        tilt = tilt_from_gravity(gravity_b)  # (N, 3, 3): base -> heading
        wmap = statics["wrench_map"]  # (N, 2, J, 6)
        lever_h = torch.einsum("nab,nfjb->nfja", tilt, wmap[..., :3]) / length
        axis_h = torch.einsum("nab,nfjb->nfja", tilt, wmap[..., 3:])
        j_h = torch.cat([lever_h, axis_h], dim=-1)  # (N, 2, J, 6)
        n, joints = joint_pos.shape
        j_mat = j_h.permute(0, 2, 1, 3).reshape(n, joints, 12)
        tau_g = statics["gravity_tau"] / mg_l
        if "loop_map" not in statics:
            return tau_g, j_mat
        driven = ((self.stiffness > 0) | (self.damping > 0)).to(joint_pos.dtype)
        free = (1.0 - driven) * self.on_loop  # (J,)
        lmap = statics["loop_map"].reshape(n, joints, -1) / length  # (N, J, 3K)
        l_free = lmap * free[None, :, None]
        gram = l_free.transpose(1, 2) @ l_free
        diag = gram.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-9)
        reg = torch.diag_embed((1e-5 * diag)[:, None].expand(-1, lmap.shape[2]))
        rhs = torch.cat([tau_g[:, :, None], j_mat], dim=-1) * free[None, :, None]
        sol = torch.linalg.solve(gram + reg, l_free.transpose(1, 2) @ rhs)
        corr = lmap @ sol
        return tau_g - corr[..., 0], j_mat - corr[..., 1:]

    def leg_mask(self) -> torch.Tensor:
        """``(J,)`` bool: active driven joints on either foot's kinematic chain or closed loop."""
        active = (self.stiffness > 0) & (self.effective_effort() > 0)
        return ((self.on_chain.sum(0) > 0) | (self.on_loop > 0)) & active

    def operational_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Static dimensionless drive parameters ``(tau_eff_dimless, k_eff, leg_mask)`` each ``(J,)``.

        ``k_eff`` is the actuator-bounded effective stiffness ``(kp * 1.5 * tau_eff) / (kp + 1.5 * tau_eff)``
        in ``m g L`` units: prevents high-stiffness/weak-effort drives (e.g. NAO) from saturating
        under operational-space velocity commands while staying linear for compliant drives.
        """
        mg_l = float(GRAVITY * self.anchors.mass.sum() * self.anchors.units.length)
        effort = self.effective_effort()
        active = (self.stiffness > 0) & (effort > 0)
        tau_eff = torch.where(active, effort / mg_l, torch.ones_like(effort))
        kp_dimless = torch.where(active, self.stiffness / mg_l, torch.zeros_like(self.stiffness))
        k_eff = (kp_dimless * 1.5 * tau_eff) / (kp_dimless + 1.5 * tau_eff).clamp_min(1e-6)
        return tau_eff, k_eff, self.leg_mask()

    def operational_anchor_from_ratio(
        self, joint_pos: torch.Tensor, gravity_b: torch.Tensor, cmd_ratio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Operational-space force+motion anchor terms from current pose and commanded torque ratio.

        Returns ``tau_g (N, J)``, ``j_eff (N, J, 12)``, ``j_pinv (N, J, 12)``, ``w_t (N, 12)``,
        ``k_eff (J,)``, ``tau_eff_dimless (J,)`` -- all dimensionless in ``m g L`` / heading frame.
        """
        tau_g, j_eff = self.wrench_anchor(joint_pos, gravity_b)
        tau_eff, k_eff, leg = self.operational_params()
        leg_f = leg.to(j_eff.dtype)
        jl = j_eff * leg_f[None, :, None]
        gram = jl.transpose(1, 2) @ jl
        diag = gram.diagonal(dim1=1, dim2=2)
        # Damped operational-space pseudoinverse J_pinv = J_eff (J_eff^T J_eff + reg)^{-1}
        p_prior = j_eff.new_tensor([2e-2, 2e-2, 2e-2, 1e-1, 1e-1, 1e-1] * 2)
        p_reg = torch.diag_embed(p_prior[None, :] * diag.clamp_min(1e-3) + 1e-2)
        j_pinv = torch.linalg.solve(gram + p_reg, jl.transpose(1, 2)).transpose(1, 2)
        # Current heading-frame foot wrench w_t from current commanded torque ratio
        tau_t = torch.atanh(cmd_ratio.clamp(-0.92, 0.92)) * tau_eff[None, :]
        rhs = -(tau_t + tau_g) * leg_f[None, :]
        w_prior = j_eff.new_tensor([1e-3, 1e-3, 1e-3, 5e-2, 5e-2, 5e-2] * 2)
        w_reg = torch.diag_embed(w_prior[None, :] * diag.clamp_min(1e-4) + 1e-3)
        w_t = torch.linalg.solve(gram + w_reg, jl.transpose(1, 2) @ rhs[..., None])[..., 0].clamp(
            -2.5, 2.5
        )
        return {
            "tau_g": tau_g,
            "j_eff": j_eff,
            "j_pinv": j_pinv,
            "w_t": w_t,
            "k_eff": k_eff,
            "tau_eff_dimless": tau_eff,
        }

    def operational_anchor(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        gravity_b: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Same as :meth:`operational_anchor_from_ratio`, reading ``cmd_ratio`` from ``target``."""
        return self.operational_anchor_from_ratio(
            joint_pos, gravity_b, self.drive_ratio(joint_pos, joint_vel, target)
        )

    def targets_from_torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        tau_dimless: torch.Tensor,
        clip_ratio: float = 3.0,
    ) -> torch.Tensor:
        """Exact analytic PD inversion: dimensionless torque ``tau / (m g L)`` -> joint position
        targets relative to ``default`` (rad), soft-bounded to ``±clip_ratio * tau_eff``."""
        effort = self.effective_effort()
        active = (self.stiffness > 0) & (effort > 0)
        mg_l = float(GRAVITY * self.anchors.mass.sum() * self.anchors.units.length)
        tau_nm = torch.clamp(tau_dimless * mg_l, -clip_ratio * effort, clip_ratio * effort)
        kp_safe = torch.where(active, self.stiffness, torch.ones_like(self.stiffness))
        rel_target = (joint_pos - self.default) + (self.damping * joint_vel + tau_nm) / kp_safe
        return torch.where(active, rel_target, torch.zeros_like(rel_target))
