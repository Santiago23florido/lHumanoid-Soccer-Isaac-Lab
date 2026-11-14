"""A body cost for the real robot's motors: the heat their windings make.

A brushed DC motor's copper loss is ``R i^2`` and its torque is ``k i``, so the
heat a joint makes is proportional to its squared torque. The NAO's motor
boards cut a joint's stiffness when it overheats, and its knees carry the
crouch on a long lever (``nao/results/joint_load_*.json``: the knee's RMS torque
equals its motor's nominal on the adopted tracker). :func:`motor_heat` charges
each leg joint's squared torque in units of its motor's nominal torque through
its gearbox, so a joint at its nominal rating costs 1. It prices what the body
spends, not how it walks: no gait, posture or timing appears in it.
"""

from __future__ import annotations

import json

import torch
from isaaclab.managers import SceneEntityCfg

from .events import MOTORS


def nominal_joint_torque(joint_names: list[str]) -> torch.Tensor:
    """``(J,)`` each joint's nominal motor torque through its gearbox, N m (ideal gearbox)."""
    spec = json.loads(MOTORS.read_text(encoding="utf-8"))
    values = []
    for name in joint_names:
        key = name[1:] if name[:1] in "LR" and name[1:] in spec["joints"] else name
        motor, gear = spec["joints"][key]
        entry = spec["motors"][motor]
        values.append(entry["nominal_torque_mNm"] * 1e-3 * entry["reduction"][gear])
    return torch.tensor(values)


def motor_heat(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """``(N,)`` the sum over ``asset_cfg``'s joints of (torque / nominal torque)^2."""
    robot = env.scene[asset_cfg.name]
    cache = getattr(env, "_nominal_torque", None)
    if cache is None:
        names = [robot.data.joint_names[j] for j in asset_cfg.joint_ids]
        cache = nominal_joint_torque(names).to(env.device)
        env._nominal_torque = cache
    torque = robot.data.applied_torque[:, asset_cfg.joint_ids]
    return ((torque / cache) ** 2).sum(dim=-1)


__all__ = ["motor_heat", "nominal_joint_torque"]
