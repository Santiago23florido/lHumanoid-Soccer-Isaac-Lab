"""A simulated body's joint drives, read off its articulation (no Isaac Lab import).

PACT's drive loads (:class:`~.anchors.DriveLoads`) need each joint's stiffness, damping
and effort limit as the simulation holds them; captures record the same numbers.
"""

from __future__ import annotations

import numpy as np
import torch

DRIVE_KEYS = ("drive_stiffness", "drive_damping", "drive_effort_limit")


def drive_parameters(robot) -> dict | None:
    """Per joint, the drives' stiffness, damping and effort limit as the simulation holds
    them, and the default joint positions; None if any cannot be read."""
    try:
        count = len(robot.joint_names)
        out = {key: np.full(count, np.nan, dtype=np.float32) for key in DRIVE_KEYS}
        for actuator in robot.actuators.values():
            index = actuator.joint_indices
            if isinstance(index, slice):
                index = list(range(count))[index]
            index = [int(i) for i in (index.tolist() if hasattr(index, "tolist") else index)]
            for key, value in zip(
                DRIVE_KEYS,
                (actuator.stiffness, actuator.damping, actuator.effort_limit),
                strict=True,
            ):
                row = value[0] if isinstance(value, torch.Tensor) and value.ndim == 2 else value
                row = row.detach().cpu().numpy() if isinstance(row, torch.Tensor) else row
                out[key][index] = np.broadcast_to(np.asarray(row, dtype=np.float32), len(index))
        out["default_joint_pos"] = robot.data.default_joint_pos[0].cpu().numpy()
        if any(np.isnan(v).any() for v in out.values()):
            raise ValueError("a joint without an actuator model")
        return out
    except (AttributeError, ValueError, TypeError) as error:
        print(f"[drives] drive parameters not read: {error}")
        return None
