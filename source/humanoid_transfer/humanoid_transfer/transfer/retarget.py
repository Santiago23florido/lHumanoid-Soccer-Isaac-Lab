"""The joint-target baseline (B3): the G1's joint trajectories, as NAO joint targets.

The classic way to move a gait between robots is to copy its joint trajectories
through a correspondence (:mod:`correspondence`) and train the second robot to
imitate them. The zero-shot study argues the joint signal is the wrong one to send;
this module builds it, as well as it can be built, so the argument is a measurement.

**Values.** :mod:`correspondence` maps the names and the signs and deliberately leaves
the values open: an angle does not transfer as a number, and the candidate maps are
to be measured against each other. Three are built here, each clamped to the
student's limits:

``copy``
    ``q_s = σ q_t``: the angle itself.
``offset``
    ``q_s = q_s⁰ + σ (q_t − q_t⁰)``: the excursion from each robot's own nominal
    posture, which keeps the student's crouch.
``range``
    The teacher's angle at the same fraction of its range (in the student's sign
    convention) as the student's: ``q_s = lo_s + (hi_s − lo_s)(σ q_t − lo_t')/(hi_t' − lo_t')``.

**Time.** Dynamic similarity fixes the student's clock: a motion that lasts ``D`` on the
teacher lasts ``D T_s/T_t`` on the student (``T = √(L/g)``). The student's control
step ``k`` reads the teacher's motion at time ``k Δt_s T_t/T_s``, interpolated, inside
one episode of the teacher's capture (a reset is never interpolated across).

**Root.** The reference velocity of the base is the teacher's centre-of-mass velocity
in its heading frame (``z*``'s ``com_velocity``, already in ``V``) and its yaw rate, in
the student's units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correspondence import signed_channels

MODES = ("copy", "offset", "range")


@dataclass(frozen=True)
class JointRecord:
    """Joint names, each joint's nominal angle and its limits, in one robot's order."""

    names: tuple[str, ...]
    default: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def index(self, name: str) -> int:
        return self.names.index(name)


def retarget_angles(
    teacher_angles: np.ndarray, teacher: JointRecord, student: JointRecord, mode: str
) -> np.ndarray:
    """``(..., J_t) -> (..., J_s)``: the teacher's angles as the student's, clamped.

    A student joint without a teacher counterpart holds its nominal angle.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    shape = teacher_angles.shape[:-1]
    out = np.broadcast_to(student.default, (*shape, len(student.names))).copy()
    for teacher_joint, student_joint, sign in signed_channels():
        t, s = teacher.index(teacher_joint), student.index(student_joint)
        q = teacher_angles[..., t]
        if mode == "copy":
            value = sign * q
        elif mode == "offset":
            value = student.default[s] + sign * (q - teacher.default[t])
        else:
            low, high = sorted((sign * teacher.lower[t], sign * teacher.upper[t]))
            fraction = (sign * q - low) / (high - low)
            value = student.lower[s] + (student.upper[s] - student.lower[s]) * fraction
        out[..., s] = np.clip(value, student.lower[s], student.upper[s])
    return out


def episodes(done: np.ndarray) -> list[tuple[int, int, int]]:
    """``(env, first, last)`` of each uninterrupted stretch of a ``(T, N)`` capture.

    ``done[t]`` means the step taken from row ``t`` ended an episode, so row ``t + 1``
    begins a new one.
    """
    stretches = []
    for env in range(done.shape[1]):
        first = 0
        for t in np.flatnonzero(done[:, env]):
            stretches.append((env, first, int(t)))
            first = int(t) + 1
        if first < done.shape[0]:
            stretches.append((env, first, done.shape[0] - 1))
    return stretches


def resample(series: np.ndarray, factor: float) -> np.ndarray:
    """``(T, ...) -> (T', ...)``: row ``k`` of the result is ``series`` at row ``k factor``.

    ``factor`` is how many of the source's rows one row of the result spans.
    """
    source = np.arange(series.shape[0], dtype=float)
    target = np.arange(0.0, series.shape[0] - 1 + 1e-9, factor)
    flat = series.reshape(series.shape[0], -1)
    out = np.stack([np.interp(target, source, flat[:, k]) for k in range(flat.shape[1])], -1)
    return out.reshape(len(target), *series.shape[1:])


__all__ = ["MODES", "JointRecord", "episodes", "resample", "retarget_angles"]
