"""Source-robot articulation configuration.

The NAO package derives every number from a URDF because it had to: the model
arrived as an XML file and nothing else. The source robot is the opposite case.
Isaac Lab ships a validated articulation, tuned actuator groups and a working
locomotion task, so this module's job is to *select* one and record what it is,
not to rebuild it.

Selecting rather than hardcoding matters for the transfer study. The interesting
variable is how large the embodiment gap is, and the cleanest way to vary it is
to change which robot plays teacher. ``SOURCE_ROBOT`` is that switch.

Isaac Lab's assets download from its Nucleus server on first use, so the first
run needs network access. Nothing is vendored here and nothing is licensed the
way the NAO meshes are -- which is the other reason this robot is convenient.
"""

from __future__ import annotations

from typing import Any

SOURCE_ROBOT = "g1_29dof"
"""Which shipped humanoid backs the source embodiment.

============  ====  ======  =============================================
value         DOF   mass    notes
============  ====  ======  =============================================
``g1_29dof``  29    ~35 kg  Default. Most joints, most capable, the widest
                            gap to the NAO and therefore the most
                            demanding transfer.
``g1``        23    ~35 kg  Same platform, reduced joint set.
``h1``        19    ~52 kg  Larger, simpler, matches the NAO's 19
                            commanded joints exactly -- useful as a
                            control condition where the joint counts agree
                            and only the scale differs.
============  ====  ======  =============================================

Changing this value changes which teacher is trained. It does not change the
student, which is always the NAO.
"""

_CFG_NAMES = {
    "g1_29dof": "G1_29DOF_CFG",
    "g1": "G1_CFG",
    "h1": "H1_CFG",
}

NOMINAL_BASE_HEIGHT = {
    "g1_29dof": 0.74,
    "g1": 0.74,
    "h1": 1.05,
}
"""Approximate standing base height in metres, as the shipped configs spawn it.

Used to size the teacher's command ranges against the student's. These are the
spawn heights from the Isaac Lab configurations, not measurements of a settled
pose, and they are quoted to two decimals for that reason.
"""


def get_source_robot_cfg(robot: str | None = None) -> Any:
    """Return the ``ArticulationCfg`` for the selected source robot.

    Imported lazily. ``isaaclab_assets`` pulls in Isaac Lab, which pulls in the
    Isaac Sim Kit application, and the tests import this package on machines
    where that is not available.

    Args:
        robot: Override for :data:`SOURCE_ROBOT`.

    Returns:
        The Isaac Lab articulation configuration, unmodified. Callers that need
        a different prim path should use ``.replace(prim_path=...)``.
    """
    name = robot or SOURCE_ROBOT
    if name not in _CFG_NAMES:
        raise ValueError(
            f"unknown source robot {name!r}; choose one of {sorted(_CFG_NAMES)}"
        )

    import isaaclab_assets.robots.unitree as unitree

    return getattr(unitree, _CFG_NAMES[name])


def describe_source_robot(robot: str | None = None) -> dict[str, Any]:
    """Static facts about the selected source robot, without loading Isaac Lab.

    Enough to reason about the embodiment gap in tests and documentation.
    """
    name = robot or SOURCE_ROBOT
    if name not in _CFG_NAMES:
        raise ValueError(
            f"unknown source robot {name!r}; choose one of {sorted(_CFG_NAMES)}"
        )
    dof = {"g1_29dof": 29, "g1": 23, "h1": 19}[name]
    return {
        "name": name,
        "config": _CFG_NAMES[name],
        "actuated_dof": dof,
        "nominal_base_height": NOMINAL_BASE_HEIGHT[name],
    }


__all__ = [
    "NOMINAL_BASE_HEIGHT",
    "SOURCE_ROBOT",
    "describe_source_robot",
    "get_source_robot_cfg",
]
