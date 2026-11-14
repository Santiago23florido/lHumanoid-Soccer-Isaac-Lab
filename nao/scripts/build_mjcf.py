"""Build a MuJoCo model of the NAO from its URDF, for sim2sim tests of the robot's runtime.

The bodies, joints, masses and inertias are the URDF's (the derived one Isaac
Lab imports, ``nao/assets/generated/nao/nao_isaac.urdf``); collision shapes are
its decimated meshes, colliding with the floor only (as in Isaac Lab, no self
collisions). The right hip yaw-pitch and the fingers follow their drivers by
equality constraints. Every measured joint gets a position actuator with the
simulator's gains and torque ceiling, read from a deployment bundle's
``runtime.json`` -- so the same drive model runs in a second physics engine.

The model is derived from CC BY-NC-ND meshes and is written under
``nao/assets/generated/`` (ignored by git). Runs without Isaac Lab::

    python nao/scripts/build_mjcf.py --bundle nao/bundles/<name>
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402
from humanoid_transfer.nao.assets.nao_paths import DERIVED_URDF_PATH  # noqa: E402

OUTPUT = ROOT / "nao" / "assets" / "generated" / "nao_mujoco.xml"


def quat_of(rotation: np.ndarray) -> list[float]:
    """w, x, y, z of a rotation matrix."""
    m = rotation
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return list(q / np.linalg.norm(q))


def fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def collisions(urdf: Path) -> dict[str, list[tuple[str, list[float], np.ndarray, np.ndarray]]]:
    """Each link's collision meshes: file, scale, origin rotation and position."""
    out: dict[str, list] = {}
    for link in ET.parse(urdf).getroot().findall("link"):
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            origin = collision.find("origin")
            xyz = (
                [0.0, 0.0, 0.0]
                if origin is None
                else [float(v) for v in origin.get("xyz", "0 0 0").split()]
            )
            rpy = (
                [0.0, 0.0, 0.0]
                if origin is None
                else [float(v) for v in origin.get("rpy", "0 0 0").split()]
            )
            scale = [float(v) for v in mesh.get("scale", "1 1 1").split()]
            rotation = nk._rpy_to_matrix(*rpy)
            out.setdefault(link.get("name"), []).append(
                (
                    str((urdf.parent / mesh.get("filename")).resolve()),
                    scale,
                    rotation,
                    np.array(xyz),
                )
            )
    return out


def build(drives: dict, friction: float, timestep: float, urdf: Path = DERIVED_URDF_PATH) -> str:
    model = nk.load_model(urdf)
    shapes = collisions(urdf)
    by_parent: dict[str, list] = {}
    for joint in model.joints:
        by_parent.setdefault(joint.parent, []).append(joint)

    mujoco = ET.Element("mujoco", model="nao")
    ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mujoco, "option", timestep=f"{timestep}", integrator="implicitfast")
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(
        default,
        "geom",
        contype="0",
        conaffinity="1",
        condim="3",
        friction=f"{friction} 0.005 0.0001",
    )
    asset = ET.SubElement(mujoco, "asset")
    meshes: dict[tuple, str] = {}
    world = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 1",
        contype="1",
        conaffinity="1",
        friction=f"{friction} 0.005 0.0001",
    )
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1")

    def add_body(parent: ET.Element, link: str, rotation: np.ndarray, position: np.ndarray, joint):
        body = ET.SubElement(
            parent, "body", name=link, pos=fmt(position), quat=fmt(quat_of(rotation))
        )
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.type != "fixed":
            drive = drives.get(joint.name, {})
            lower = drive.get("lower", joint.lower)
            upper = drive.get("upper", joint.upper)
            limits = {} if lower is None or upper is None else {"range": fmt([lower, upper])}
            ET.SubElement(
                body, "joint", name=joint.name, type="hinge", axis=fmt(joint.axis), **limits
            )
        info = model.links[link]
        if info.mass > 0.0:
            # The inertia in the link frame (MuJoCo takes a full tensor without a rotation).
            i = info.inertia_rot @ info.inertia @ info.inertia_rot.T
            ET.SubElement(
                body,
                "inertial",
                pos=fmt(info.com),
                mass=f"{info.mass:.9g}",
                fullinertia=fmt([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]]),
            )
        for k, (file, scale, rot, pos) in enumerate(shapes.get(link, [])):
            key = (file, tuple(scale))
            if key not in meshes:
                meshes[key] = f"mesh{len(meshes)}"
                ET.SubElement(asset, "mesh", name=meshes[key], file=file, scale=fmt(scale))
            ET.SubElement(
                body,
                "geom",
                name=f"{link}_{k}",
                type="mesh",
                mesh=meshes[key],
                pos=fmt(pos),
                quat=fmt(quat_of(rot)),
            )
        for child in by_parent.get(link, []):
            sub = model.links[child.child]
            if child.type == "fixed" and sub.mass == 0.0 and not by_parent.get(child.child):
                continue  # a sensor frame: nothing to simulate
            add_body(body, child.child, child.origin_rot, child.origin_xyz, child)
        return body

    add_body(world, "base_link", np.eye(3), np.zeros(3), None)
    ET.SubElement(world.find("body"), "site", name="imu", pos="0 0 0")

    equality = ET.SubElement(mujoco, "equality")
    for joint in model.joints:
        if joint.mimic is not None and joint.type != "fixed":
            ET.SubElement(
                equality, "joint", joint1=joint.name, joint2=joint.mimic, polycoef="0 1 0 0 0"
            )
    actuator = ET.SubElement(mujoco, "actuator")
    for joint in model.joints:
        drive = drives.get(joint.name)
        if joint.type == "fixed" or joint.mimic is not None or drive is None:
            continue
        if drive["stiffness"] <= 0.0:
            continue
        ET.SubElement(
            actuator,
            "position",
            name=joint.name,
            joint=joint.name,
            kp=f"{drive['stiffness']:.9g}",
            kv=f"{drive['damping']:.9g}",
            forcerange=fmt([-drive["effort"], drive["effort"]]),
            forcelimited="true",
        )
    ET.indent(mujoco)
    return ET.tostring(mujoco, encoding="unicode")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--friction", type=float, default=0.64)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    drives = json.loads((args.bundle / "runtime.json").read_text(encoding="utf-8"))["drives"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(drives, args.friction, args.timestep), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
