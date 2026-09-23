"""Rigid-body kinematics of the NAO, computed from the vendored URDF.

Every geometric and inertial quantity the balance task needs is derived here
from ``assets/robots/nao/urdf/nao.urdf`` rather than copied from a datasheet.
That matters: the controller gains, the reward scales and the termination
thresholds are all sized from these numbers, so they have to come from the same
model the simulator loads. A datasheet value that disagreed with the URDF would
silently detune everything downstream.

Only NumPy is used, never Isaac Lab, so the tests can check these values on a
machine with no Isaac Sim installed. The functions are written for clarity on a
single pose, not for batched use inside an RL loop: the environment computes the
centre of mass on the GPU from the simulator's own body poses instead.

See ``nao/assets/licenses/README.md`` for the model's provenance and licensing.
"""

from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .nao_paths import NAO_ASSET_DIR, NAO_URDF_PATH, URDF_MESH_PACKAGE_PREFIX

__all__ = [
    "GRAVITY",
    "Link",
    "Joint",
    "UrdfModel",
    "load_model",
    "forward_kinematics",
    "center_of_mass",
    "com_height_above_soles",
    "sole_height_below_base",
    "lipm_omega",
    "composite_inertia_about_joint",
    "whole_body_inertia_about_ankle",
    "foot_collision_extent",
    "NOMINAL_STAND_JOINT_POS",
    "LEG_ACTUATED_JOINTS",
    "ARM_ACTUATED_JOINTS",
    "ACTUATED_JOINTS",
    "HELD_JOINTS",
    "MIMIC_DRIVEN_JOINTS",
    "FOOT_BODIES",
    "SOLE_FRAMES",
    "SUPPORT_POLYGON_X",
    "SUPPORT_POLYGON_Y_SINGLE",
    "STANCE_HALF_WIDTH",
    "support_polygon_double_stance",
    "capturable_com_velocity",
]

GRAVITY = 9.81
"""Gravitational acceleration in m/s^2, matching the Isaac Lab simulation default."""


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Link:
    """One URDF ``<link>`` with its inertial properties in the link frame."""

    name: str
    mass: float
    com: np.ndarray
    """Centre of mass offset from the link origin, in metres."""
    inertia: np.ndarray
    """3x3 inertia tensor about the centre of mass, in the inertial frame."""
    inertia_rot: np.ndarray
    """Rotation from the inertial frame to the link frame."""


@dataclass(frozen=True)
class Joint:
    """One URDF ``<joint>`` with its fixed origin and motion axis."""

    name: str
    type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    mimic: str | None


@dataclass(frozen=True)
class UrdfModel:
    """Parsed URDF: links by name, joints in document order, plus a child index."""

    name: str
    links: dict[str, Link]
    joints: tuple[Joint, ...]
    children: dict[str, tuple[str, ...]]

    @property
    def total_mass(self) -> float:
        return sum(link.mass for link in self.links.values())


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the URDF fixed-axis roll-pitch-yaw rotation ``R_z R_y R_x``."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``angle`` radians about a unit ``axis``."""
    unit = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def _origin_of(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Read a URDF ``<origin>``; an absent origin is the identity, per the spec."""
    if element is None:
        return np.eye(3), np.zeros(3)
    xyz = element.get("xyz")
    rpy = element.get("rpy")
    translation = np.array([float(v) for v in xyz.split()]) if xyz else np.zeros(3)
    angles = [float(v) for v in rpy.split()] if rpy else [0.0, 0.0, 0.0]
    return _rpy_to_matrix(*angles), translation


def _float_or_none(element: ET.Element | None, key: str) -> float | None:
    """Read a numeric URDF attribute, tolerating both absent element and key."""
    if element is None:
        return None
    raw = element.get(key)
    return float(raw) if raw is not None else None


@lru_cache(maxsize=4)
def load_model(urdf_path: Path | None = None) -> UrdfModel:
    """Parse the URDF into links and joints. Cached: the file never changes."""
    path = urdf_path or NAO_URDF_PATH
    root = ET.parse(path).getroot()

    links: dict[str, Link] = {}
    for element in root.findall("link"):
        name = element.get("name", "")
        inertial = element.find("inertial")
        if inertial is None:
            # 27 of the 36 merged sensor frames carry no <inertial> at all.
            links[name] = Link(name, 0.0, np.zeros(3), np.zeros((3, 3)), np.eye(3))
            continue
        mass = float(inertial.find("mass").get("value"))
        rotation, translation = _origin_of(inertial.find("origin"))
        tensor = inertial.find("inertia")
        ixx = _float_or_none(tensor, "ixx") or 0.0
        ixy = _float_or_none(tensor, "ixy") or 0.0
        ixz = _float_or_none(tensor, "ixz") or 0.0
        iyy = _float_or_none(tensor, "iyy") or 0.0
        iyz = _float_or_none(tensor, "iyz") or 0.0
        izz = _float_or_none(tensor, "izz") or 0.0
        inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        links[name] = Link(name, mass, translation, inertia, rotation)

    joints: list[Joint] = []
    children: dict[str, list[str]] = {}
    for element in root.findall("joint"):
        rotation, translation = _origin_of(element.find("origin"))
        axis_element = element.find("axis")
        if axis_element is not None and axis_element.get("xyz"):
            axis = np.array([float(v) for v in axis_element.get("xyz").split()])
        else:
            axis = np.array([0.0, 0.0, 1.0])
        limit = element.find("limit")
        mimic = element.find("mimic")
        parent = element.find("parent").get("link")
        child = element.find("child").get("link")
        joints.append(
            Joint(
                name=element.get("name", ""),
                type=element.get("type", "fixed"),
                parent=parent,
                child=child,
                origin_xyz=translation,
                origin_rot=rotation,
                axis=axis,
                lower=_float_or_none(limit, "lower"),
                upper=_float_or_none(limit, "upper"),
                effort=_float_or_none(limit, "effort"),
                velocity=_float_or_none(limit, "velocity"),
                mimic=mimic.get("joint") if mimic is not None else None,
            )
        )
        children.setdefault(parent, []).append(child)

    return UrdfModel(
        name=root.get("name", "unknown"),
        links=links,
        joints=tuple(joints),
        children={key: tuple(value) for key, value in children.items()},
    )


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def forward_kinematics(
    joint_pos: dict[str, float] | None = None, urdf_path: Path | None = None
) -> dict[str, np.ndarray]:
    """Return every link's 4x4 pose in the ``base_link`` frame.

    Args:
        joint_pos: Joint angles in radians. Omitted joints are held at zero.
        urdf_path: Override for the URDF location.

    Returns:
        ``{link_name: 4x4 transform}``. ``base_link`` is the identity.
    """
    model = load_model(urdf_path)
    angles = joint_pos or {}
    poses: dict[str, np.ndarray] = {"base_link": np.eye(4)}

    # The URDF does not list joints parent-before-child, so sweep until nothing
    # new resolves. The tree has 79 links; this converges in a few passes.
    progressed = True
    while progressed:
        progressed = False
        for joint in model.joints:
            if joint.parent not in poses or joint.child in poses:
                continue
            local = _transform(joint.origin_rot, joint.origin_xyz)
            if joint.type != "fixed":
                # A mimic joint follows its driver one-to-one on this model:
                # every <mimic> tag in the NAO URDF omits multiplier and offset.
                source = joint.mimic or joint.name
                angle = angles.get(joint.name, angles.get(source, 0.0))
                local = local @ _transform(_axis_rotation(joint.axis, angle), np.zeros(3))
            poses[joint.child] = poses[joint.parent] @ local
            progressed = True
    return poses


def center_of_mass(
    joint_pos: dict[str, float] | None = None, urdf_path: Path | None = None
) -> np.ndarray:
    """Return the whole-body centre of mass in the ``base_link`` frame, in metres."""
    model = load_model(urdf_path)
    poses = forward_kinematics(joint_pos, urdf_path)
    total = 0.0
    weighted = np.zeros(3)
    for name, link in model.links.items():
        if link.mass == 0.0 or name not in poses:
            continue
        world = (poses[name] @ np.append(link.com, 1.0))[:3]
        weighted += link.mass * world
        total += link.mass
    return weighted / total


def sole_height_below_base(
    joint_pos: dict[str, float] | None = None, urdf_path: Path | None = None
) -> float:
    """Return how far the lower sole sits below ``base_link``, in metres."""
    poses = forward_kinematics(joint_pos, urdf_path)
    return float(-min(poses["l_sole"][2, 3], poses["r_sole"][2, 3]))


def com_height_above_soles(
    joint_pos: dict[str, float] | None = None, urdf_path: Path | None = None
) -> float:
    """Return ``z_c``, the centre-of-mass height above the soles, in metres.

    This is the length scale of the linear inverted pendulum, so it sets the
    natural frequency of the entire balance problem.
    """
    poses = forward_kinematics(joint_pos, urdf_path)
    sole_z = min(poses["l_sole"][2, 3], poses["r_sole"][2, 3])
    return float(center_of_mass(joint_pos, urdf_path)[2] - sole_z)


def lipm_omega(com_height: float) -> float:
    r"""Return the linear inverted pendulum frequency :math:`\omega_0=\sqrt{g/z_c}`.

    The LIPM divergent mode grows as :math:`e^{\omega_0 t}`, so ``1/omega`` is the
    time constant the controller has to beat and ``ln(2)/omega`` is how long the
    robot takes to double a balance error.
    """
    return math.sqrt(GRAVITY / com_height)


# ---------------------------------------------------------------------------
# Inertia
# ---------------------------------------------------------------------------


def _subtree(model: UrdfModel, root_link: str) -> list[str]:
    """Return ``root_link`` and every link below it in the kinematic tree."""
    found = [root_link]
    for child in model.children.get(root_link, ()):
        found.extend(_subtree(model, child))
    return found


def _inertia_about_axis(
    model: UrdfModel,
    poses: dict[str, np.ndarray],
    link_names: list[str],
    point: np.ndarray,
    axis: np.ndarray,
) -> float:
    """Composite inertia of ``link_names`` about the axis through ``point``.

    Parallel-axis (Steiner) sum of every link's inertia, projected onto the axis.
    """
    total = np.zeros((3, 3))
    for name in link_names:
        link = model.links.get(name)
        if link is None or link.mass == 0.0 or name not in poses:
            continue
        rotation = poses[name][:3, :3]
        com_world = (poses[name] @ np.append(link.com, 1.0))[:3]
        frame = rotation @ link.inertia_rot
        inertia_world = frame @ link.inertia @ frame.T
        offset = com_world - point
        total += inertia_world + link.mass * (
            offset @ offset * np.eye(3) - np.outer(offset, offset)
        )
    unit = axis / np.linalg.norm(axis)
    return float(unit @ total @ unit)


def composite_inertia_about_joint(
    joint_name: str,
    joint_pos: dict[str, float] | None = None,
    urdf_path: Path | None = None,
) -> float:
    """Inertia of the *distal* subtree about ``joint_name``'s axis, in kg m^2.

    .. warning::
        This is the open-chain load, which is **not** what a standing robot's
        ankle feels. With the foot on the ground the chain inverts and the ankle
        carries the whole body; see :func:`whole_body_inertia_about_ankle`, which
        is larger by a factor of roughly 650. Sizing ankle gains from the value
        returned here would under-shoot by two orders of magnitude.
    """
    model = load_model(urdf_path)
    poses = forward_kinematics(joint_pos, urdf_path)
    joint = next(j for j in model.joints if j.name == joint_name)
    frame = poses[joint.child]
    return _inertia_about_axis(
        model,
        poses,
        _subtree(model, joint.child),
        frame[:3, 3],
        frame[:3, :3] @ joint.axis,
    )


def whole_body_inertia_about_ankle(
    joint_name: str = "LAnklePitch",
    joint_pos: dict[str, float] | None = None,
    urdf_path: Path | None = None,
) -> tuple[float, float]:
    """Return ``(inertia, lever_arm)`` of the whole body about an ankle axis.

    With both feet flat the robot behaves as a single rigid body pivoting about
    the ankle axis, so this is the inertia the balance controller actually sees.
    The lever arm is the distance from that axis to the centre of mass, which
    turns into the destabilising gravitational stiffness ``M g l``.

    Returns:
        Inertia in kg m^2 and lever arm in metres.
    """
    model = load_model(urdf_path)
    poses = forward_kinematics(joint_pos, urdf_path)
    joint = next(j for j in model.joints if j.name == joint_name)
    frame = poses[joint.child]
    point = frame[:3, 3]
    inertia = _inertia_about_axis(
        model, poses, list(model.links), point, frame[:3, :3] @ joint.axis
    )
    lever = float(np.linalg.norm(center_of_mass(joint_pos, urdf_path) - point))
    return inertia, lever


# ---------------------------------------------------------------------------
# Joint groups
# ---------------------------------------------------------------------------

MIMIC_DRIVEN_JOINTS: tuple[str, ...] = (
    "RHipYawPitch",
    "LFinger11",
    "LFinger12",
    "LFinger13",
    "LFinger21",
    "LFinger22",
    "LFinger23",
    "LThumb1",
    "LThumb2",
    "RFinger11",
    "RFinger12",
    "RFinger13",
    "RFinger21",
    "RFinger22",
    "RFinger23",
    "RThumb1",
    "RThumb2",
)
"""Joints the URDF marks ``<mimic>``, so they must never be commanded directly.

``RHipYawPitch`` is the load-bearing one: on the real NAO a *single* motor drives
both hip yaw-pitch joints, and the URDF encodes that as a mimic of
``LHipYawPitch``. Commanding it independently would fight the PhysX mimic
constraint. The sixteen finger and thumb joints follow ``LHand`` and ``RHand``.
"""

LEG_ACTUATED_JOINTS: tuple[str, ...] = (
    "LHipYawPitch",
    "LHipRoll",
    "LHipPitch",
    "LKneePitch",
    "LAnklePitch",
    "LAnkleRoll",
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
)
"""The eleven independent leg degrees of freedom.

Eleven rather than twelve because of the shared hip yaw-pitch motor above. These
carry the ankle and hip balance strategies.
"""

ARM_ACTUATED_JOINTS: tuple[str, ...] = (
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
)
"""Shoulder and elbow joints, which enable the angular-momentum strategy.

When the centre of pressure saturates at the edge of the support polygon the
only remaining way to generate a restoring moment is to change centroidal
angular momentum, which in practice means swinging the arms.
"""

ACTUATED_JOINTS: tuple[str, ...] = LEG_ACTUATED_JOINTS + ARM_ACTUATED_JOINTS
"""The 19 joints the balance policy commands."""

HELD_JOINTS: tuple[str, ...] = (
    "HeadYaw",
    "HeadPitch",
    "LWristYaw",
    "RWristYaw",
    "LHand",
    "RHand",
)
"""Joints held at their nominal angle by a stiff PD, not exposed to the policy.

They contribute almost nothing to balance -- the head is 0.68 kg close to the
body axis, and the wrists and hands are lighter still -- so commanding them
would only add exploration noise.
"""

FOOT_BODIES: tuple[str, ...] = ("l_ankle", "r_ankle")
"""Bodies that carry the foot collision geometry after the fixed frames merge.

``l_sole`` and ``r_sole`` carry no inertial, so the URDF-to-USD conversion merges
them into their parent ankle links. Contact sensing therefore targets these.
"""

SOLE_FRAMES: tuple[str, ...] = ("l_sole", "r_sole")
"""Sole frames of the URDF, used for kinematics here but absent from the USD."""


# ---------------------------------------------------------------------------
# Nominal standing posture
# ---------------------------------------------------------------------------

_CROUCH_HIP = -0.35
_CROUCH_KNEE = 0.70
_CROUCH_ANKLE = -0.35

NOMINAL_STAND_JOINT_POS: dict[str, float] = {
    # Sagittal leg chain. The three pitch angles sum to zero, which is the
    # condition for the torso to stay vertical and the foot flat while the knee
    # is bent: theta_hip + theta_ankle = -theta_knee.
    "LHipPitch": _CROUCH_HIP,
    "LKneePitch": _CROUCH_KNEE,
    "LAnklePitch": _CROUCH_ANKLE,
    "RHipPitch": _CROUCH_HIP,
    "RKneePitch": _CROUCH_KNEE,
    "RAnklePitch": _CROUCH_ANKLE,
    # Arms hanging down and slightly out, well inside their limits. The elbow
    # roll sign is forced by the URDF: a NAO elbow cannot straighten, so
    # LElbowRoll is confined to [-1.545, -0.035] and RElbowRoll to [0.035, 1.545].
    "LShoulderPitch": 1.40,
    "LShoulderRoll": 0.15,
    "LElbowYaw": -1.20,
    "LElbowRoll": -0.50,
    "RShoulderPitch": 1.40,
    "RShoulderRoll": -0.15,
    "RElbowYaw": 1.20,
    "RElbowRoll": 0.50,
}
"""Nominal standing posture: a shallow crouch with the arms down.

The crouch is a *kinematic conditioning* choice, not a stability one. At full
knee extension the leg Jacobian loses rank in the vertical direction, so the
controller cannot move the centre of mass up or down and its authority over the
sagittal plane collapses. Bending the knee by 0.70 rad moves the configuration
well away from that singularity at a cost of only 12 mm of standing height.

Joints not listed keep the URDF zero, except the elbow rolls above whose limits
exclude zero (see :func:`~humanoid_transfer.nao.assets.nao_paths.default_joint_positions`).
"""


# ---------------------------------------------------------------------------
# Support polygon
# ---------------------------------------------------------------------------

SUPPORT_POLYGON_X: tuple[float, float] = (-0.0607, 0.1033)
"""Foot extent along +x (forward), in metres, relative to the sole frame.

Measured from the extent of the upstream collision mesh
``meshes/V40/RAnkleRoll_0.10.stl`` with the URDF's own ``scale="0.1 0.1 0.1"``
applied: the foot reaches 60.7 mm behind and 103.3 mm ahead of the ankle axis,
for a total length of 164 mm. ``tests/test_nao_kinematics.py`` re-derives this
from the mesh whenever the geometry has been fetched, so the constant cannot
drift from the model.
"""

SUPPORT_POLYGON_Y_SINGLE: tuple[float, float] = (-0.0526, 0.0397)
"""One foot's extent along +y (left), in metres, in that foot's sole frame.

Stated for the right foot; the left foot mirrors it. Total width 92.3 mm. The
asymmetry about the ankle axis is real: a NAO foot is wider on its outer edge.
"""

STANCE_HALF_WIDTH = 0.05
"""Lateral offset of each sole frame from the sagittal plane, in metres.

Read from the URDF: both soles sit exactly 50 mm either side of ``base_link``.
"""


def support_polygon_double_stance() -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ``((x_min, x_max), (y_min, y_max))`` of the double-stance polygon.

    The convex hull of two flat feet bridges the gap between them, so the
    lateral extent runs from the outer edge of one foot to the outer edge of the
    other. Coordinates are relative to the midpoint between the two sole frames.
    """
    outer = STANCE_HALF_WIDTH - SUPPORT_POLYGON_Y_SINGLE[0]
    return SUPPORT_POLYGON_X, (-outer, outer)


def capturable_com_velocity(
    joint_pos: dict[str, float] | None = None, urdf_path: Path | None = None
) -> dict[str, float]:
    r"""Largest centre-of-mass velocity still recoverable without stepping.

    From the divergent component of motion :math:`\xi = x + \dot{x}/\omega_0`,
    the robot can come to rest without moving a foot exactly when :math:`\xi`
    can be placed inside the support polygon. Since the centre of pressure is
    bounded by the polygon edge at distance :math:`d`, the bound on velocity is

    .. math:: \dot{x}_{max} = \omega_0 \, d

    This is the zero-step capturability limit of Pratt et al. It is what sizes
    the perturbation curriculum: pushes beyond it are not recoverable by an
    ankle strategy alone and would teach the policy nothing but how to fall.

    Returns:
        Bounds in m/s for the forward, backward and lateral directions, the
        centre-of-mass height and LIPM frequency they were derived from, and the
        forward impulse in N s that produces the forward bound.
    """
    pose = NOMINAL_STAND_JOINT_POS if joint_pos is None else joint_pos
    model = load_model(urdf_path)
    height = com_height_above_soles(pose, urdf_path)
    omega = lipm_omega(height)
    poses = forward_kinematics(pose, urdf_path)
    sole_midpoint = 0.5 * (poses["l_sole"][:3, 3] + poses["r_sole"][:3, 3])
    # The support bounds are defined about the soles, not about base_link.
    com = center_of_mass(pose, urdf_path) - sole_midpoint

    x_min, x_max = SUPPORT_POLYGON_X
    _, (_, y_max) = support_polygon_double_stance()
    forward_margin = x_max - com[0]
    backward_margin = com[0] - x_min
    lateral_margin = y_max - abs(com[1])

    return {
        "omega": omega,
        "com_height": height,
        "forward": omega * forward_margin,
        "backward": omega * backward_margin,
        "lateral": omega * lateral_margin,
        "forward_impulse": model.total_mass * omega * forward_margin,
    }


# ---------------------------------------------------------------------------
# Mesh measurement (used by the tests to validate the constants above)
# ---------------------------------------------------------------------------


def foot_collision_extent(
    mesh_name: str = "meshes/V40/RAnkleRoll_0.10.stl", urdf_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the foot collision mesh AABB in the sole frame, in metres.

    Reads the binary STL directly and applies the URDF's mesh scale and the
    ankle-to-sole offset, so the result is directly comparable with
    :data:`SUPPORT_POLYGON_X` and :data:`SUPPORT_POLYGON_Y_SINGLE`.

    Raises:
        FileNotFoundError: if the CC BY-NC-ND geometry has not been fetched.
    """
    path = NAO_ASSET_DIR / mesh_name
    if not path.is_file():
        raise FileNotFoundError(
            f"NAO collision mesh not found: {path}. Run scripts/fetch_nao_meshes.py."
        )

    model = load_model(urdf_path)
    scale = np.array([0.1, 0.1, 0.1])
    for mesh in ET.parse(urdf_path or NAO_URDF_PATH).getroot().iter("mesh"):
        if mesh.get("filename", "") != f"{URDF_MESH_PACKAGE_PREFIX}{mesh_name}":
            continue
        raw = mesh.get("scale")
        if raw:
            scale = np.array([float(v) for v in raw.split()])
        break

    data = path.read_bytes()
    triangle_count = struct.unpack("<I", data[80:84])[0]
    vertices = np.empty((triangle_count * 3, 3))
    offset = 84
    for index in range(triangle_count):
        values = struct.unpack("<12f", data[offset : offset + 48])
        vertices[3 * index : 3 * index + 3] = np.array(values[3:12]).reshape(3, 3)
        offset += 50
    vertices *= scale

    # The sole frame sits a fixed distance below the ankle link origin.
    sole_joint = next(j for j in model.joints if j.child == "r_sole")
    vertices = vertices - sole_joint.origin_xyz
    return vertices.min(axis=0), vertices.max(axis=0)


def capturable_velocity_in_direction(
    angle: float | np.ndarray,
    joint_pos: dict[str, float] | None = None,
    urdf_path: Path | None = None,
) -> float | np.ndarray:
    r"""Zero-step capturable speed for a push along ``angle``, in m/s.

    The bound :math:`\dot{x}_{max} = \omega_0 d` needs ``d``, the distance from
    the centre of mass to the edge of the support polygon **in the direction the
    push is driving it**. That distance is strongly direction dependent for a
    biped: the NAO's centre of mass sits 12.6 mm ahead of the sole midpoint, and
    the heel is 60.7 mm behind the ankle against 103.3 mm of toe, so backward is
    the tightest direction by a wide margin.

    Using a single number for every direction is what makes a curriculum
    unlearnable. Set to the forward bound of 0.548 m/s, roughly half the pushes
    land in the backward hemisphere where much of that magnitude cannot be
    recovered by any zero-step controller, and the policy spends most of its
    experience on episodes it could not have won.

    The polygon is treated as the axis-aligned rectangle the two flat feet span,
    so ``d`` is an exact ray-box distance rather than an interpolation between
    three measured directions.

    Args:
        angle: Push heading in radians, 0 forward and pi/2 to the robot's left.
            Accepts an array.
        joint_pos: Posture to evaluate at. Defaults to the nominal stand.
        urdf_path: Override for the URDF location.

    Returns:
        Capturable speed in m/s, matching the shape of ``angle``.
    """
    pose = NOMINAL_STAND_JOINT_POS if joint_pos is None else joint_pos
    omega = lipm_omega(com_height_above_soles(pose, urdf_path))

    poses = forward_kinematics(pose, urdf_path)
    sole_midpoint = 0.5 * (poses["l_sole"][:3, 3] + poses["r_sole"][:3, 3])
    com = (center_of_mass(pose, urdf_path) - sole_midpoint)[:2]

    (x_min, x_max), (y_min, y_max) = support_polygon_double_stance()

    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    # Distance along the ray to each of the four edges, keeping the nearest
    # positive one. A component of zero never reaches its pair of edges, so it
    # is pushed to infinity rather than dividing by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        to_x = np.where(
            direction[..., 0] > 0,
            (x_max - com[0]) / direction[..., 0],
            np.where(direction[..., 0] < 0, (x_min - com[0]) / direction[..., 0], np.inf),
        )
        to_y = np.where(
            direction[..., 1] > 0,
            (y_max - com[1]) / direction[..., 1],
            np.where(direction[..., 1] < 0, (y_min - com[1]) / direction[..., 1], np.inf),
        )
    return omega * np.minimum(to_x, to_y)
