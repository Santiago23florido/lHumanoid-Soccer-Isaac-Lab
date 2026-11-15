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
* :func:`scale_drive_gains` -- the drives' stiffness and damping as multiples of the
  derived gains.

Each draws, per environment, either one factor per joint (``spread=None``) or one
factor for the whole robot with every joint scattered ``±spread`` around it. The
second makes the robot's factor a property of the robot -- all its motors stronger
or weaker together, as with wear, heat or battery -- that a policy can be told or
can infer (:mod:`.params`, which reads it from :func:`~.params.robot_draws`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from isaaclab.actuators import ImplicitActuator
from isaaclab.managers import SceneEntityCfg

from .params import draw_factors, robot_draws

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


def _ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids


def _actuator_columns(actuator, num_joints: int, device) -> torch.Tensor:
    indices = actuator.joint_indices
    if isinstance(indices, slice):
        indices = torch.arange(num_joints, device=device)[indices]
    return indices.to(device)


def scale_effort_limits(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
    spread: float | None = None,
) -> None:
    """Each selected joint's torque ceiling times a factor (see the module).

    A startup event: the ceilings it scales are the asset's, read before any change.
    """
    robot = env.scene[asset_cfg.name]
    ids = _ids(env, env_ids)
    joints = torch.as_tensor(asset_cfg.joint_ids, device=env.device)
    default = robot.data.joint_effort_limits[ids][:, joints].clone()
    factors, own = draw_factors(len(ids), len(joints), scale_range, spread, env.device)
    robot.write_joint_effort_limit_to_sim(default * factors, joint_ids=joints, env_ids=ids)
    robot_draws(env)["effort_scale"][ids] = own
    # The implicit drives clip the torque they report with their own copy.
    for actuator in robot.actuators.values():
        indices = _actuator_columns(actuator, robot.num_joints, env.device)
        mask = torch.isin(indices, joints)
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
    spread: float | None = None,
) -> None:
    """Each selected joint's armature: its reflected rotor inertia times a factor (see the
    module)."""
    robot = env.scene[asset_cfg.name]
    ids = _ids(env, env_ids)
    joints = list(asset_cfg.joint_ids)
    names = [robot.data.joint_names[j] for j in joints]
    nominal = reflected_inertia(names, rotor_inertia).to(env.device)
    factors, own = draw_factors(len(ids), len(joints), scale_range, spread, env.device)
    robot.write_joint_armature_to_sim(nominal * factors, joint_ids=joints, env_ids=ids)
    robot_draws(env)["armature_scale"][ids] = own


def scale_drive_gains(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_range: tuple[float, float] | None = None,
    damping_range: tuple[float, float] | None = None,
    spread: float | None = None,
) -> None:
    """The selected joints' drive stiffness and damping times factors of the derived gains
    (see the module), as Isaac Lab's ``randomize_actuator_gains`` writes them."""
    robot = env.scene[asset_cfg.name]
    ids = _ids(env, env_ids)
    joints = torch.as_tensor(asset_cfg.joint_ids, device=env.device)
    for key, scale_range, default, attribute, write in (
        ("stiffness_scale", stiffness_range, robot.data.default_joint_stiffness, "stiffness",
         robot.write_joint_stiffness_to_sim),
        ("damping_scale", damping_range, robot.data.default_joint_damping, "damping",
         robot.write_joint_damping_to_sim),
    ):  # fmt: skip
        if scale_range is None:
            continue
        factors, own = draw_factors(len(ids), len(joints), scale_range, spread, env.device)
        values = default[ids].clone()
        values[:, joints] = values[:, joints] * factors
        for actuator in robot.actuators.values():
            columns = _actuator_columns(actuator, robot.num_joints, env.device)
            getattr(actuator, attribute)[ids] = values[:, columns]
            if isinstance(actuator, ImplicitActuator):
                write(values[:, columns], joint_ids=columns, env_ids=ids)
        robot_draws(env)[key][ids] = own


__all__ = [
    "ROTOR_INERTIA",
    "reflected_inertia",
    "scale_drive_gains",
    "scale_effort_limits",
    "set_reflected_inertia",
]
