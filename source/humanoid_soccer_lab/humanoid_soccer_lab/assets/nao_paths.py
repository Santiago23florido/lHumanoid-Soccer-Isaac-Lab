"""Filesystem layout of the NAO asset.

Pure standard library on purpose: this module is imported by the mesh
bootstrap script and by the tests, both of which must work on a machine that
has neither Isaac Sim nor Isaac Lab installed.

See ``third_party/nao/README.md`` for provenance and licensing of the files
these paths point at.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "NAO_ASSET_DIR",
    "NAO_URDF_PATH",
    "NAO_MESH_DIR",
    "NAO_MESH_V40_DIR",
    "NAO_TEXTURE_DIR",
    "NAO_TEXTURE_PATH",
    "GENERATED_NAO_DIR",
    "DERIVED_URDF_PATH",
    "NAO_USD_PATH",
    "ASSET_CACHE_DIR",
    "URDF_MESH_PACKAGE_PREFIX",
    "JOINT_LIMIT_MARGIN_RAD",
    "required_mesh_files",
    "missing_mesh_files",
    "meshes_are_available",
    "describe_missing_meshes",
    "joint_limits_from_urdf",
    "default_joint_positions",
    "urdf_robot_name",
]


def _resolve_repo_root() -> Path:
    """Locate the repository root.

    ``NAO_ASSET_ROOT`` wins if set, which keeps the package usable when it is
    installed non-editable or when the assets live outside the checkout.
    """
    override = os.environ.get("NAO_ASSET_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # .../source/humanoid_soccer_lab/humanoid_soccer_lab/assets/nao_paths.py
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _resolve_repo_root()

NAO_ASSET_DIR = REPO_ROOT / "assets" / "robots" / "nao"
"""Root of the NAO asset tree."""

NAO_URDF_PATH = NAO_ASSET_DIR / "urdf" / "nao.urdf"
"""Verbatim upstream URDF (BSD 3-Clause). Tracked in git."""

NAO_MESH_DIR = NAO_ASSET_DIR / "meshes"
NAO_MESH_V40_DIR = NAO_MESH_DIR / "V40"
"""Geometry directory. Untracked; populated by ``scripts/fetch_nao_meshes.py``."""

NAO_TEXTURE_DIR = NAO_ASSET_DIR / "texture"
NAO_TEXTURE_PATH = NAO_TEXTURE_DIR / "textureNAO.png"
"""Texture directory.

Must sit beside ``meshes/`` because the upstream COLLADA files reference the
texture as ``../../texture/textureNAO.png``.
"""

GENERATED_NAO_DIR = REPO_ROOT / "assets" / "generated" / "nao"
"""Build artifacts derived from the sources above. Untracked."""

DERIVED_URDF_PATH = GENERATED_NAO_DIR / "nao_isaac.urdf"
"""URDF with repository-relative mesh paths, generated for Isaac Sim."""

NAO_USD_PATH = GENERATED_NAO_DIR / "usd" / "nao.usd"
"""USD produced by the Isaac Lab URDF converter."""

ASSET_CACHE_DIR = REPO_ROOT / ".asset_cache"
"""Download cache for third-party archives. Untracked."""

URDF_MESH_PACKAGE_PREFIX = "package://nao_meshes/"
"""ROS-style prefix used by the upstream URDF for every mesh reference."""

JOINT_LIMIT_MARGIN_RAD = 1.0e-3
"""Numerical guard, in radians, kept between a default joint angle and its limit.

Not a physical parameter: the URDF stores limits in double precision while the
simulation compares them in float32, so a value placed exactly on a limit can
round to the wrong side of it and fail Isaac Lab's default-state validation.
"""


def required_mesh_files(urdf_path: Path | None = None) -> list[str]:
    """Return the mesh paths the URDF needs, relative to ``NAO_ASSET_DIR``.

    Parsed from the URDF rather than hard-coded so the list cannot drift from
    the model.
    """
    urdf_path = urdf_path or NAO_URDF_PATH
    root = ET.parse(urdf_path).getroot()
    found: list[str] = []
    seen: set[str] = set()
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith(URDF_MESH_PACKAGE_PREFIX):
            continue
        relative = filename[len(URDF_MESH_PACKAGE_PREFIX) :]
        if relative not in seen:
            seen.add(relative)
            found.append(relative)
    return sorted(found)


def missing_mesh_files(urdf_path: Path | None = None) -> list[str]:
    """Return the required mesh paths that are not present on disk."""
    return [rel for rel in required_mesh_files(urdf_path) if not (NAO_ASSET_DIR / rel).is_file()]


def meshes_are_available(urdf_path: Path | None = None) -> bool:
    """True when every mesh referenced by the URDF resolves on disk."""
    if not NAO_MESH_V40_DIR.is_dir():
        return False
    return not missing_mesh_files(urdf_path)


def urdf_robot_name(urdf_path: Path | None = None) -> str:
    """Return the ``name`` attribute of the URDF ``<robot>`` element."""
    urdf_path = urdf_path or NAO_URDF_PATH
    return ET.parse(urdf_path).getroot().get("name", "unknown")


def joint_limits_from_urdf(urdf_path: Path | None = None) -> dict[str, tuple[float, float]]:
    """Return ``{joint_name: (lower, upper)}`` for every limited joint in the URDF.

    Continuous joints carry no positional limit and are omitted.
    """
    urdf_path = urdf_path or NAO_URDF_PATH
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        lower, upper = limit.get("lower"), limit.get("upper")
        if lower is None or upper is None:
            continue
        limits[name] = (float(lower), float(upper))
    return limits


def default_joint_positions(urdf_path: Path | None = None) -> dict[str, float]:
    """Return the URDF zero pose clamped into the joint limits.

    The NAO zero pose is not reachable: ``LElbowRoll`` is limited to
    ``[-1.545, -0.035]`` and ``RElbowRoll`` to ``[0.035, 1.545]``, because a NAO
    elbow cannot fully straighten. Loading the raw zero pose therefore fails
    Isaac Lab's validation with "default positions out of the limits".

    Only joints whose zero is actually out of range appear in the result; every
    other joint keeps the upstream zero. The returned angle is the closest valid
    one, pushed :data:`JOINT_LIMIT_MARGIN_RAD` inside the limit. No posture is
    invented — the values come from the URDF's own limits.
    """
    adjusted: dict[str, float] = {}
    for name, (lower, upper) in joint_limits_from_urdf(urdf_path).items():
        if lower <= 0.0 <= upper:
            continue
        margin = min(JOINT_LIMIT_MARGIN_RAD, (upper - lower) / 100.0)
        adjusted[name] = lower + margin if lower > 0.0 else upper - margin
    return adjusted


def describe_missing_meshes(urdf_path: Path | None = None) -> str:
    """Build the operator-facing message shown when geometry is absent."""
    missing = missing_mesh_files(urdf_path)
    lines = [
        "NAO geometry is missing from this checkout.",
        "",
        f"  expected under : {NAO_ASSET_DIR}",
        f"  missing files  : {len(missing)}",
        "",
        "The NAO meshes are licensed CC BY-NC-ND 4.0 and cannot be redistributed",
        "in this repository, so they are fetched locally after you accept the",
        "upstream license. Run this once per checkout:",
        "",
        "  python scripts/fetch_nao_meshes.py",
        "",
        "See third_party/nao/README.md for the licensing details.",
    ]
    return "\n".join(lines)
