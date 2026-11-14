"""Randomisation of the NAO's drives, drawn once per environment at startup.

* :func:`scale_effort_limits` -- each joint's torque ceiling as a multiple of the
  URDF's. The URDF's 3.02 N m at the pitch joints is below the motor's ideal stall
  torque through its gearbox (8.9 N m) and above its nominal (2.11 N m); what the
  board's current limit and the gearbox's efficiency leave is unknown until measured.
* :func:`set_reflected_inertia` -- the rotor's inertia seen at the joint through
  the gearbox, ``J_r r^2`` (the simulated joints have none): with ``J_r`` about
  3e-7 kg m^2 for a 22 mm motor (Portescap's 22N-series datasheets: 2.0--3.5e-7)
  and ``r`` = 130.85 or 201.3 (``nao/body/motors.json``), 5e-3 to 1.2e-2 kg m^2,
  more than a swinging shank's own inertia about the knee.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from isaaclab.managers import SceneEntityCfg

MOTORS = Path(__file__).resolve().parents[5] / "nao" / "body" / "motors.json"
ROTOR_INERTIA = 3.0e-7
"""A leg motor's rotor inertia, kg m^2 (see the module's docstring)."""


def reflected_inertia(joint_names: list[str], rotor_inertia: float = ROTOR_INERTIA) -> torch.Tensor:
    """``(J,)`` each joint's reflected rotor inertia ``J_r r^2``; 0 for joints without a motor
    listed (the mimic joints, which share their driver's)."""
    spec = json.loads(MOTORS.read_text(encoding="utf-8"))
    values = []
    for name in joint_names:
        key = name[1:] if name[:1] in "LR" and name[1:] in spec["joints"] else name
        entry = spec["joints"].get(key)
        if entry is None:
            values.append(0.0)
            continue
        motor, gear = entry
        values.append(rotor_inertia * spec["motors"][motor]["reduction"][gear] ** 2)
    return torch.tensor(values)


def _draw(count: int, width: int, low: float, high: float, device) -> torch.Tensor:
    return low + (high - low) * torch.rand(count, width, device=device)


def scale_effort_limits(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, scale_range: tuple[float, float]
) -> None:
    """Each selected joint's torque ceiling times a factor drawn per environment and joint.

    A startup event: the ceilings it scales are the asset's, read before any change.
    """
    robot = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    joints = torch.as_tensor(asset_cfg.joint_ids, device=env.device)
    default = robot.data.joint_effort_limits[ids][:, joints].clone()
    limits = default * _draw(len(ids), len(joints), *scale_range, env.device)
    robot.write_joint_effort_limit_to_sim(limits, joint_ids=joints, env_ids=ids)
    # The implicit drives clip the torque they report with their own copy.
    for actuator in robot.actuators.values():
        indices = actuator.joint_indices
        if isinstance(indices, slice):
            indices = torch.arange(robot.num_joints, device=env.device)[indices]
        mask = torch.isin(indices.to(env.device), joints)
        if mask.any():
            columns = mask.nonzero().squeeze(-1)
            limit = actuator.effort_limit
            if not isinstance(limit, torch.Tensor):
                continue
            limit[ids[:, None], columns] = robot.data.joint_effort_limits[ids][:, indices[columns]]


def set_reflected_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
    rotor_inertia: float = ROTOR_INERTIA,
) -> None:
    """Each selected joint's armature: its reflected rotor inertia times a factor drawn per
    environment and joint."""
    robot = env.scene[asset_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    joints = list(asset_cfg.joint_ids)
    names = [robot.data.joint_names[j] for j in joints]
    nominal = reflected_inertia(names, rotor_inertia).to(env.device)
    armature = nominal * _draw(len(ids), len(joints), *scale_range, env.device)
    robot.write_joint_armature_to_sim(armature, joint_ids=joints, env_ids=ids)


__all__ = ["ROTOR_INERTIA", "reflected_inertia", "scale_effort_limits", "set_reflected_inertia"]
