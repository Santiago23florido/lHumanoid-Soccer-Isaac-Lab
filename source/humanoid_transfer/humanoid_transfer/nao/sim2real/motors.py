"""The NAO's leg motors as the simulation sees them: each joint's reflected rotor inertia.

``J_r r^2``: the rotor's inertia seen at the joint through the gearbox, with ``J_r`` about
3e-7 kg m^2 for a 22 mm motor (Portescap's 22N-series datasheets: 2.0--3.5e-7) and ``r``
the joint's reduction (``nao/body/motors.json``). Without Isaac Lab, so the drive fit
(``nao/scripts/fit_actuators.py``) runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

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


__all__ = ["MOTORS", "ROTOR_INERTIA", "reflected_inertia"]
