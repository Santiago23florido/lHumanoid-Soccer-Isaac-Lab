"""A gait's dimensionless signature, read off a sequence of skill states.

The dynamic similarity hypothesis (Alexander and Jayes, 1983) says that bodies
of different size, moving at the same Froude number, walk with the same
*dimensionless* stride length, stride period and duty factor. That turns those
few numbers into the most compact description of a gait that should survive a
change of body, and it is what a student's gait gets compared against: not
"does the NAO move its knee like the G1" but "at the same Froude number, does
it take strides of the same length in leg lengths, at the same rate in pendulum
times".

Every quantity here comes from the skill state of
:mod:`~humanoid_transfer.common.skill_state`, so it is already in body units --
there is no length or mass anywhere in this module.

Sequences are ``(T, N, ...)``: time first, then independent rollouts. A
``done`` flag at step ``t`` means the rollout was reset after it, so step
``t + 1`` starts a new episode and no event or interval may span the two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .skill_state import (
    DEFAULT_CONTACT_THRESHOLD,
    SIDES,
    SKILL_LAYOUT,
    ZLayout,
    contact_state,
    dcm_offset,
    mirror,
)


def episode_start(done: np.ndarray) -> np.ndarray:
    """``(T, N)`` True where a step is the first of an episode (``t = 0`` or after a reset)."""
    start = np.zeros_like(done, dtype=bool)
    start[0] = True
    start[1:] = done[:-1]
    return start


def touchdowns(contact: np.ndarray, done: np.ndarray) -> np.ndarray:
    """``(T, N, 2)`` True where a foot goes from air to ground.

    A contact already present at the start of an episode is not a touchdown:
    nothing landed, the episode just began there.
    """
    landing = np.zeros_like(contact, dtype=bool)
    landing[1:] = contact[1:] & ~contact[:-1]
    landing[episode_start(done)] = False
    return landing


def stride_periods(landing: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Intervals, in steps, between consecutive touchdowns of the same foot, within episodes."""
    episode = np.cumsum(episode_start(done), axis=0)
    intervals: list[int] = []
    for env in range(landing.shape[1]):
        for side in range(landing.shape[2]):
            times = np.flatnonzero(landing[:, env, side])
            if times.size < 2:
                continue
            same = episode[times[1:], env] == episode[times[:-1], env]
            intervals.extend(np.diff(times)[same].tolist())
    return np.asarray(intervals, dtype=float)


@dataclass(frozen=True)
class GaitSignature:
    """The dimensionless numbers dynamic similarity says a body change should preserve."""

    speed: float
    """Mean horizontal CoM speed in ``V = sqrt(g L)``; its square is the Froude number."""

    froude: float
    stride_period: float
    """Same-foot touchdown interval, in pendulum times ``T = sqrt(L / g)``."""

    stride_length: float
    """``speed x stride_period``, in leg lengths."""

    duty_factor: tuple[float, float]
    """Fraction of time each foot carries more than the contact threshold."""

    dcm_at_touchdown: tuple[tuple[float, float], tuple[float, float]]
    """Per foot, ``(xi - p_foot)_xy / L`` at the instant it lands, heading frame."""

    asymmetry: float
    """``|E z - M E z| / (|E z| + |M E z|)``: 0 for a gait identical to its mirror image."""

    touchdowns: int
    contact_threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def gait_signature(
    z: np.ndarray,
    done: np.ndarray,
    dt_star: float,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
    layout: ZLayout = SKILL_LAYOUT,
) -> GaitSignature:
    """Summarise a ``(T, N, 27)`` skill-state sequence sampled every ``dt_star`` body times."""
    fields = layout.unpack(z)
    contact = contact_state(z, layout, threshold)
    landing = touchdowns(contact, done)
    periods = stride_periods(landing, done) * dt_star

    speed = float(np.linalg.norm(fields["com_velocity"][..., :2], axis=-1).mean())
    period = float(periods.mean()) if periods.size else float("nan")

    offset = dcm_offset(z, layout)[..., None, :] - fields["foot_position"][..., :2]
    at_landing = tuple(
        tuple(float(v) for v in offset[..., side, :][landing[..., side]].mean(axis=0))
        if landing[..., side].any()
        else (float("nan"), float("nan"))
        for side in range(len(SIDES))
    )

    mean = z.reshape(-1, z.shape[-1]).mean(axis=0)
    mirrored = mirror(mean, layout)
    asymmetry = float(
        np.linalg.norm(mean - mirrored) / (np.linalg.norm(mean) + np.linalg.norm(mirrored))
    )

    return GaitSignature(
        speed=speed,
        froude=speed**2,
        stride_period=period,
        stride_length=speed * period,
        duty_factor=tuple(float(v) for v in contact.mean(axis=(0, 1))),
        dcm_at_touchdown=at_landing,
        asymmetry=asymmetry,
        touchdowns=int(landing.sum()),
        contact_threshold=threshold,
    )


__all__ = [
    "GaitSignature",
    "episode_start",
    "gait_signature",
    "stride_periods",
    "touchdowns",
]
