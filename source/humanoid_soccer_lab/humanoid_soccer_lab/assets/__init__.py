"""Robot and scene asset configurations.

:mod:`nao_paths` is pure standard library and safe to import anywhere. The
other modules require Isaac Lab, so they are imported lazily to keep this
package usable on machines without Isaac Sim.
"""

from .nao_paths import (
    DERIVED_URDF_PATH,
    NAO_ASSET_DIR,
    NAO_URDF_PATH,
    NAO_USD_PATH,
    meshes_are_available,
)

__all__ = [
    "DERIVED_URDF_PATH",
    "NAO_ASSET_DIR",
    "NAO_URDF_PATH",
    "NAO_USD_PATH",
    "meshes_are_available",
    "NAO_CFG",
    "get_nao_cfg",
    "NAO_EXPECTED_JOINTS",
    "NAO_SPAWN_HEIGHT",
]


def __getattr__(name: str):
    """Expose the Isaac Lab dependent symbols only when they are asked for."""
    if name in {"NAO_CFG", "get_nao_cfg", "NAO_EXPECTED_JOINTS", "NAO_SPAWN_HEIGHT"}:
        from . import nao

        return getattr(nao, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
