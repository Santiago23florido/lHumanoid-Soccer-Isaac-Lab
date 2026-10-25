"""How a body is held up, measured: each foot's centre of pressure and ground force.

``z*`` carries each foot's vertical load and nothing of where on the foot it
acts or how hard the ground pushes sideways -- what a body's balance is made
of. A simulator reports neither directly (its contact sensor gives the normal
force only), so both are reconstructed from the wrench the leg transmits to the
foot through the ankle (:func:`~...latent.whole_body.ground_reaction`), as the
NAO's balance controller does.

In the body's own units and its heading frame, per foot, left then right:

* the centre of pressure relative to the sole point, ``/ L``, ``(x, y)`` -- zero
  while the foot carries less than :data:`LOADED` of the body's weight;
* the ground force, ``/ m g``, ``(x, y, z)``.

What a critic and a tracker's physics heads read (decoder v2, skill_space.md §9);
it is not part of the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

from ...latent.skill_torch import quat_to_matrix
from ...latent.whole_body import centre_of_pressure, ground_reaction, yaw_rotation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

LOADED = 0.05
"""Share of the body's weight below which a foot's centre of pressure is not defined."""

GRAVITY = 9.81
SIZE = 10


def foot_reactions(term) -> dict[str, torch.Tensor]:
    """Each foot's ground reaction in the world frame, from a tracking command's body.

    ``force`` ``(N, 2, 3)`` newtons; ``cop`` ``(N, 2, 2)`` metres; ``soles`` ``(N, 2, 3)``;
    ``loaded`` ``(N, 2)``, the feet carrying at least :data:`LOADED` of the body's weight.
    """
    data, body = term.robot.data, term.body
    feet = list(body.foot_index)
    quat = data.body_link_quat_w[:, feet]
    wrench = data.body_incoming_joint_wrench_b[:, feet]
    joint_force = quat_apply(quat, wrench[..., :3])
    joint_moment = quat_apply(quat, wrench[..., 3:])
    joint_pos = data.body_link_pos_w[:, feet]
    foot_com = data.body_com_pos_w[:, feet]
    force, moment = ground_reaction(
        joint_force, joint_moment, joint_pos, foot_com, body.masses[feet], GRAVITY
    )
    foot_rot = quat_to_matrix(quat)
    soles = foot_com + torch.einsum("nfij,fj->nfi", foot_rot, body.sole_from_com)
    ground = soles[..., 2].min(dim=1).values
    cop = centre_of_pressure(force, moment, joint_pos, ground)
    weight = float(body.masses.sum()) * GRAVITY
    return {"force": force, "cop": cop, "soles": soles, "loaded": force[..., 2] > LOADED * weight}


def balance_state(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """``(N, 10)``: per foot, the centre of pressure ``/ L`` then the ground force ``/ m g``."""
    term = env.command_manager.get_term(command_name)
    data, body = term.robot.data, term.body
    reaction = foot_reactions(term)
    force, cop, soles = reaction["force"], reaction["cop"], reaction["soles"]
    heading = yaw_rotation(quat_to_matrix(data.body_link_quat_w[:, body.base_index]))
    to_heading = heading.transpose(-1, -2)[:, None]
    weight = float(body.masses.sum()) * GRAVITY
    offset = torch.cat([cop - soles[..., :2], torch.zeros_like(cop[..., :1])], dim=-1)
    offset = (to_heading @ offset[..., None])[..., 0][..., :2] / body.units.length
    offset = torch.where(reaction["loaded"][..., None], offset, 0.0)
    ground_force = (to_heading @ force[..., None])[..., 0] / weight
    state = torch.cat([offset.flatten(1), ground_force.flatten(1)], dim=-1)
    return torch.nan_to_num(state, nan=0.0).clamp(-10.0, 10.0)


__all__ = ["LOADED", "SIZE", "balance_state", "foot_reactions"]
