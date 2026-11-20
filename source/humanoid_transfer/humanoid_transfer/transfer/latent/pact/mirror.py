"""The mirror image of a PACT training sample: the body's sagittal symmetry as a prior.

Every body PACT drives is mirror-symmetric, and so is the task family; the trackers were
trained toward their mirror image (mirror loss), yet a tracker can still favour one side,
and a student imitating it inherits the bias. Training PACT on each sample and its mirror
image (labels reflected the same way) fits it to the symmetric problem instead.

The reflections are :mod:`humanoid_transfer.transfer.symmetry`'s:

- joints: the body's measured mirror (a signed permutation ``M``, ``transfer/bodies/<b>/``
  or ``<b>/body/mirror.json``); positions relative to the default pose gain
  ``c = M q0 − q0`` (zero for a symmetric default pose), speeds do not;
- IMU: projected gravity flips ``y`` (a polar vector), angular velocity flips ``x`` and
  ``z`` (an axial one);
- ``z*`` (raw): :func:`~humanoid_transfer.common.skill_state.mirror_matrix`; the anchor
  inputs, the plan and its difference to the estimate all are ``z*``-shaped;
- the path ``(dx, dy, dψ)``: flips its last two;
- the encoder's six learned targets (CoM velocity, height correction, the two loads): the
  rows of the ``z*`` mirror for those fields.

NumPy only; ``tests/test_pact_mirror.py`` checks the involution and that the anchors of a
reflected joint state are the reflected anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from humanoid_transfer.common.skill_state import SKILL_LAYOUT, mirror_matrix
from humanoid_transfer.transfer.symmetry import AXIAL, PLANAR, VECTOR, joint_mirror

LEARNED_FIELDS = ("com_velocity", "height", "foot_load")
"""The encoder's learned outputs, in ``pact_gate_a2.targets_of``'s order."""


def mirror_file(repo: Path, body: str) -> Path:
    for path in (
        repo / "transfer" / "bodies" / body / "mirror.json",
        repo / body / "body" / "mirror.json",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(f"no measured mirror for {body}")


class SampleMirror:
    """Reflects one body's samples (any leading batch shape)."""

    def __init__(self, joint_names, mirror: dict, q0: np.ndarray) -> None:
        self.joints = joint_mirror(tuple(joint_names), mirror).astype(np.float32)
        q0 = np.asarray(q0, dtype=np.float32)
        self.offset = (self.joints @ q0 - q0).astype(np.float32)
        self.z = mirror_matrix(SKILL_LAYOUT).astype(np.float32)
        index = np.concatenate(
            [np.arange(SKILL_LAYOUT.size)[SKILL_LAYOUT.slice_of(f)] for f in LEARNED_FIELDS]
        )
        self.learned_matrix = self.z[np.ix_(index, index)]
        self.imu_matrix = np.zeros((6, 6), np.float32)
        self.imu_matrix[:3, :3] = VECTOR
        self.imu_matrix[3:, 3:] = AXIAL

    @classmethod
    def for_body(cls, repo: Path, body: str, joint_names, q0) -> SampleMirror:
        mirror = json.loads(mirror_file(repo, body).read_text(encoding="utf-8"))
        return cls(joint_names, mirror, q0)

    # Joint quantities: (..., J) along the last axis.
    def positions(self, x: np.ndarray) -> np.ndarray:
        """Joint positions relative to the default pose (or targets relative to it)."""
        return x @ self.joints.T + self.offset

    def speeds(self, x: np.ndarray) -> np.ndarray:
        return x @ self.joints.T

    def joint_obs(self, obs: np.ndarray) -> np.ndarray:
        """``(N, J, H, 3)``: ``q − q0``, ``q̇ sqrt(L/g)``, ``target − q0``."""
        moved = np.moveaxis(obs, 1, -1)  # (N, H, 3, J)
        out = np.empty_like(moved)
        out[..., 0, :] = self.positions(moved[..., 0, :])
        out[..., 1, :] = self.speeds(moved[..., 1, :])
        out[..., 2, :] = self.positions(moved[..., 2, :])
        return np.moveaxis(out, -1, 1)

    def imu(self, x: np.ndarray) -> np.ndarray:
        """``(..., 6)``: projected gravity, then base angular velocity (any time scaling)."""
        return x @ self.imu_matrix.T

    def skill(self, z: np.ndarray) -> np.ndarray:
        """Raw ``z*`` (or a difference of two), ``(..., 27)``."""
        return z @ self.z.T

    def path(self, x: np.ndarray) -> np.ndarray:
        """``(..., 3)``: ``(dx, dy, dψ)``."""
        return x @ PLANAR.T.astype(np.float32)

    def learned(self, x: np.ndarray) -> np.ndarray:
        """``(..., 6)``: CoM velocity, height correction, loads."""
        return x @ self.learned_matrix.T
