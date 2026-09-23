"""NAO asset integration checks.

Deliberately free of Isaac Sim and rendering so they run in any Python
environment. Tests that need the geometry skip themselves when the mesh tree
has not been fetched, because the meshes cannot be committed (see
``third_party/nao/README.md``).
"""

from __future__ import annotations

import hashlib
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_soccer_lab"))

from humanoid_soccer_lab.nao.assets import nao_paths  # noqa: E402

# Digest of nao_robot @67476469a1371b00b17538eb6ea336367ece7d44,
# nao_description/urdf/naoV50_generated_urdf/nao.urdf, copied verbatim.
UPSTREAM_URDF_SHA256 = "50da7a565da1766473fe8403859954c67d217628c143d95c0ada350cc5403dc0"

NAO_ROBOT_COMMIT = "67476469a1371b00b17538eb6ea336367ece7d44"
NAO_MESHES_COMMIT = "7c5b9f3a880fe38552459214157dcf1969812f4c"

MAJOR_JOINTS = (
    "HeadYaw",
    "HeadPitch",
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    "RHipYawPitch",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "RHand",
)

requires_meshes = pytest.mark.skipif(
    not nao_paths.meshes_are_available(),
    reason="NAO meshes not fetched; run scripts/fetch_nao_meshes.py",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


# --------------------------------------------------------------------------
# Expected files
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "assets/robots/nao/urdf/nao.urdf",
        "assets/robots/nao/README.md",
        "third_party/nao/README.md",
        "third_party/nao/LICENSE.nao_robot.txt",
        "third_party/nao/LICENSE.nao_meshes.txt",
        "scripts/view_nao.py",
        "scripts/fetch_nao_meshes.py",
        "source/humanoid_soccer_lab/humanoid_soccer_lab/nao/assets/nao.py",
        "source/humanoid_soccer_lab/humanoid_soccer_lab/nao/assets/nao_usd.py",
        "source/humanoid_soccer_lab/humanoid_soccer_lab/nao/assets/nao_paths.py",
    ],
)
def test_expected_asset_files_exist(relative: str) -> None:
    assert (ROOT / relative).is_file(), f"missing {relative}"


# --------------------------------------------------------------------------
# URDF validity and fidelity
# --------------------------------------------------------------------------


def test_urdf_is_valid_xml_and_is_nao_h25_v50() -> None:
    root = ET.parse(nao_paths.NAO_URDF_PATH).getroot()
    assert root.tag == "robot"
    assert root.get("name") == "NaoH25V50"


def test_urdf_is_byte_identical_to_upstream() -> None:
    digest = hashlib.sha256(nao_paths.NAO_URDF_PATH.read_bytes()).hexdigest()
    assert digest == UPSTREAM_URDF_SHA256, (
        "The vendored URDF must stay a verbatim copy of the upstream BSD file. "
        "Modifications belong in the generated derived URDF instead."
    )


def test_urdf_exposes_every_major_joint() -> None:
    root = ET.parse(nao_paths.NAO_URDF_PATH).getroot()
    names = {joint.get("name") for joint in root.findall("joint")}
    missing = [name for name in MAJOR_JOINTS if name not in names]
    assert missing == [], f"URDF is missing joints: {missing}"


def test_urdf_preserves_the_upstream_mesh_scale() -> None:
    """A missing scale would render the robot ten times too large."""
    root = ET.parse(nao_paths.NAO_URDF_PATH).getroot()
    meshes = list(root.iter("mesh"))
    assert meshes, "no mesh references found"
    for mesh in meshes:
        assert mesh.get("scale") == "0.1 0.1 0.1", f"unexpected scale on {mesh.get('filename')}"


def test_default_joint_positions_lie_inside_the_urdf_limits() -> None:
    limits = nao_paths.joint_limits_from_urdf()
    defaults = nao_paths.default_joint_positions()
    # The zero pose is unreachable only for the two elbow roll joints.
    assert set(defaults) == {"LElbowRoll", "RElbowRoll"}
    for name, value in defaults.items():
        lower, upper = limits[name]
        assert lower <= value <= upper, f"{name}={value} outside [{lower}, {upper}]"


# --------------------------------------------------------------------------
# Mesh resolution
# --------------------------------------------------------------------------


def test_urdf_mesh_references_are_all_ros_package_paths() -> None:
    required = nao_paths.required_mesh_files()
    assert len(required) == 78, f"expected 78 mesh references, found {len(required)}"
    assert all(name.startswith("meshes/V40/") for name in required)


@requires_meshes
def test_every_referenced_mesh_resolves_on_disk() -> None:
    assert nao_paths.missing_mesh_files() == []


@requires_meshes
def test_texture_sits_where_the_collada_files_expect_it() -> None:
    """The .dae files reference ../../texture/textureNAO.png from meshes/V40."""
    assert nao_paths.NAO_TEXTURE_PATH.is_file()
    dae = nao_paths.NAO_MESH_V40_DIR / "HeadYaw.dae"
    resolved = (dae.parent / "../../texture/textureNAO.png").resolve()
    assert resolved == nao_paths.NAO_TEXTURE_PATH.resolve()


def _stl_extent(path: Path) -> tuple[float, float, float]:
    """Return the bounding-box extent of a binary STL, in file units."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    offset = 84
    for _ in range(count):
        values = struct.unpack("<12f", data[offset : offset + 48])
        for vertex in range(3):
            for axis in range(3):
                coordinate = values[3 + vertex * 3 + axis]
                lo[axis] = min(lo[axis], coordinate)
                hi[axis] = max(hi[axis], coordinate)
        offset += 50
    return (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])


@requires_meshes
def test_foot_mesh_is_real_world_sized_once_the_scale_is_applied() -> None:
    """Independent proof that scale=0.1 is correct rather than arbitrary.

    A real NAO foot is roughly 16 cm long and 9 cm wide. Dropping the scale
    would make it 1.6 m, which this catches.
    """
    extent = _stl_extent(nao_paths.NAO_MESH_V40_DIR / "RAnkleRoll_0.10.stl")
    length, width, _ = (value * 0.1 for value in extent)
    assert 0.14 <= length <= 0.19, f"foot length {length:.4f} m is not NAO sized"
    assert 0.06 <= width <= 0.12, f"foot width {width:.4f} m is not NAO sized"


# --------------------------------------------------------------------------
# Derived URDF
# --------------------------------------------------------------------------


@requires_meshes
def test_derived_urdf_resolves_meshes_without_ros_and_drops_gazebo() -> None:
    from humanoid_soccer_lab.nao.assets import nao_usd

    path = nao_usd.build_derived_urdf(force=True)
    root = ET.parse(path).getroot()

    assert root.get("name") == "NaoH25V50"
    assert root.findall("gazebo") == []
    assert root.findall("transmission") == []

    source_root = ET.parse(nao_paths.NAO_URDF_PATH).getroot()
    assert len(root.findall("link")) == len(source_root.findall("link"))
    assert len(root.findall("joint")) == len(source_root.findall("joint"))

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        assert not filename.startswith("package://"), "ROS package path survived"
        assert mesh.get("scale") == "0.1 0.1 0.1"
        assert (path.parent / filename).resolve().is_file(), f"unresolved mesh {filename}"


# --------------------------------------------------------------------------
# Licensing and attribution
# --------------------------------------------------------------------------


def test_licenses_are_present_and_are_the_right_licenses() -> None:
    bsd = (ROOT / "third_party/nao/LICENSE.nao_robot.txt").read_text(encoding="utf-8")
    assert "Redistribution and use in source and binary forms" in bsd
    assert "University of Freiburg" in bsd

    meshes = (ROOT / "third_party/nao/LICENSE.nao_meshes.txt").read_text(encoding="utf-8")
    assert "Attribution-NonCommercial-NoDerivatives 4.0" in meshes


def test_attribution_records_upstream_projects_and_commits() -> None:
    readme = (ROOT / "third_party/nao/README.md").read_text(encoding="utf-8")
    for token in (
        "ros-naoqi/nao_robot",
        "ros-naoqi/nao_meshes",
        NAO_ROBOT_COMMIT,
        NAO_MESHES_COMMIT,
        "Armin Hornung",
        "BSD 3-Clause",
        "CC BY-NC-ND 4.0",
    ):
        assert token in readme, f"attribution is missing {token!r}"


def test_third_party_assets_are_not_relicensed_under_the_project_license() -> None:
    readme = (ROOT / "third_party/nao/README.md").read_text(encoding="utf-8")
    assert "does **not** apply to, and does not relicense" in readme


# --------------------------------------------------------------------------
# Only the required third-party files were vendored
# --------------------------------------------------------------------------


def test_no_ros_stack_was_copied_into_the_repository() -> None:
    forbidden = (
        "nao_apps",
        "nao_bringup",
        "naoqi_driver",
        "CMakeLists.txt",
        "package.xml",
        ".rosinstall",
        "naoGazebo",
        "naoTransmission",
    )
    tracked = _tracked_files()
    offenders = [f for f in tracked for token in forbidden if token in f]
    assert offenders == [], f"upstream ROS stack files were vendored: {offenders}"


def test_only_the_urdf_was_vendored_from_nao_robot() -> None:
    tracked = _tracked_files()
    nao_asset_files = [f for f in tracked if f.startswith("assets/robots/nao/")]
    assert sorted(nao_asset_files) == [
        "assets/robots/nao/README.md",
        "assets/robots/nao/urdf/nao.urdf",
    ]


def test_no_launch_or_xacro_files_were_vendored() -> None:
    tracked = _tracked_files()
    offenders = [f for f in tracked if f.endswith((".launch", ".xacro", ".rst"))]
    assert offenders == []


def test_meshes_and_generated_artifacts_are_untracked() -> None:
    """CC BY-NC-ND geometry and USD derived from it must never be committed."""
    tracked = _tracked_files()
    offenders = [
        f
        for f in tracked
        if f.startswith(("assets/robots/nao/meshes/", "assets/robots/nao/texture/"))
        or f.startswith("assets/generated/")
        or f.endswith((".dae", ".stl", ".usd", ".usda", ".usdc"))
    ]
    assert offenders == [], f"licensed or generated binaries are tracked: {offenders}"


def test_gitignore_excludes_the_licensed_and_generated_paths() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "assets/robots/nao/meshes/",
        "assets/robots/nao/texture/",
        "assets/generated/",
    ):
        assert pattern in gitignore, f".gitignore is missing {pattern}"


# --------------------------------------------------------------------------
# Phase 1 scope
# --------------------------------------------------------------------------


RL_MARKERS = (
    r"\bdef _get_rewards\b",
    r"\bdef _get_observations\b",
    r"\bdef _get_dones\b",
    r"\bDirectRLEnv\b",
    r"\bRewardTermCfg\b",
    r"\bObservationTermCfg\b",
    r"\bActionTermCfg\b",
    r"\bimport\s+(rsl_rl|skrl|stable_baselines3)\b",
    r"\bPPO\b",
)


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/view_nao.py",
        "source/humanoid_soccer_lab/humanoid_soccer_lab/nao/assets/nao.py",
        "source/humanoid_soccer_lab/humanoid_soccer_lab/nao/assets/nao_usd.py",
        "source/humanoid_soccer_lab/humanoid_soccer_lab/nao/assets/nao_paths.py",
    ],
)
def test_no_reinforcement_learning_was_implemented_for_the_nao(relative: str) -> None:
    """Phase 1 is asset integration only."""
    text = (ROOT / relative).read_text(encoding="utf-8")
    for marker in RL_MARKERS:
        assert re.search(marker, text) is None, f"{relative} contains RL code ({marker})"
