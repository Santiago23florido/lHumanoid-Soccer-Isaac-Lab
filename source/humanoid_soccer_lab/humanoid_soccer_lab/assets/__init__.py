"""Robot and scene asset configurations.

:mod:`nao_paths` and :mod:`nao_kinematics` are free of Isaac Lab and safe to
import anywhere; the kinematics module needs only NumPy. The other modules
require Isaac Lab, which in turn needs the Isaac Sim Kit application, so they
are imported lazily to keep this package usable on machines without it.
"""

from .nao_kinematics import (
    ACTUATED_JOINTS,
    ARM_ACTUATED_JOINTS,
    FOOT_BODIES,
    HELD_JOINTS,
    LEG_ACTUATED_JOINTS,
    MIMIC_DRIVEN_JOINTS,
    NOMINAL_STAND_JOINT_POS,
    capturable_com_velocity,
    center_of_mass,
    com_height_above_soles,
    lipm_omega,
    support_polygon_double_stance,
)
from .nao_paths import (
    DERIVED_URDF_PATH,
    NAO_ASSET_DIR,
    NAO_URDF_PATH,
    NAO_USD_PATH,
    meshes_are_available,
)

_LAZY = {
    "NAO_CFG",
    "NAO_STAND_CFG",
    "get_nao_cfg",
    "get_nao_stand_cfg",
    "NAO_EXPECTED_JOINTS",
    "NAO_SPAWN_HEIGHT",
    "NAO_STAND_BASE_HEIGHT",
    "NAO_STAND_SPAWN_HEIGHT",
    "NAO_STAND_COM_HEIGHT",
    "NAO_STAND_LIPM_OMEGA",
    "NAO_STAND_JOINT_POS",
    "NAO_STIFFNESS",
    "NAO_DAMPING",
    "NAO_SOFT_JOINT_POS_LIMIT_FACTOR",
}
"""Names that live in :mod:`nao`, which cannot be imported without Isaac Lab."""

__all__ = [
    # paths
    "DERIVED_URDF_PATH",
    "NAO_ASSET_DIR",
    "NAO_URDF_PATH",
    "NAO_USD_PATH",
    "meshes_are_available",
    # kinematics
    "ACTUATED_JOINTS",
    "ARM_ACTUATED_JOINTS",
    "FOOT_BODIES",
    "HELD_JOINTS",
    "LEG_ACTUATED_JOINTS",
    "MIMIC_DRIVEN_JOINTS",
    "NOMINAL_STAND_JOINT_POS",
    "capturable_com_velocity",
    "center_of_mass",
    "com_height_above_soles",
    "lipm_omega",
    "support_polygon_double_stance",
    *sorted(_LAZY),
]


def __getattr__(name: str):
    """Expose the Isaac Lab dependent symbols only when they are asked for."""
    if name in _LAZY:
        from . import nao

        return getattr(nao, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
