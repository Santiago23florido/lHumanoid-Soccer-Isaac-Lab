"""Dynamic similarity between legged systems of different size.

The transfer question forces a choice that looks like a detail and is not: when
a teacher walks at some speed, what is the corresponding speed for a student
half its height? Copying the number is wrong, and the reason is old physics
rather than anything about robots.

Kept here, with the rest of the embodiment-agnostic theory, because it is not a
property of either robot. It takes two lengths and returns a ratio.

References
----------
Alexander (1976, 1984) on the Froude number in terrestrial gaits, which is where
the criterion and the observation that animals of different sizes change gait at
the same Froude number both come from.
"""

from __future__ import annotations

import math

GRAVITY = 9.81
"""Standard gravity, in m/s^2."""


def froude_number(speed: float, leg_length: float, gravity: float = GRAVITY) -> float:
    """Dimensionless ``Fr = v^2 / (g * l)``.

    It compares the centripetal acceleration required to pivot over a stance leg
    against gravity. At ``Fr = 1`` the two are equal, which is roughly where an
    inverted-pendulum walking gait stops being possible and animals break into a
    run -- across a very wide range of body sizes.

    Args:
        speed: Forward speed, in m/s.
        leg_length: Characteristic leg length, in metres.
        gravity: Gravitational acceleration, in m/s^2.
    """
    if leg_length <= 0.0:
        raise ValueError(f"leg_length must be positive, got {leg_length}")
    return speed * speed / (gravity * leg_length)


def froude_matched_speed(
    speed: float, from_leg_length: float, to_leg_length: float
) -> float:
    """Speed on a second system that is dynamically similar to ``speed`` on the first.

    Equal Froude numbers give

    .. math::

        v_2 = v_1 \\sqrt{l_2 / l_1}

    so the scaling goes with the *square root* of the length ratio, not with the
    ratio itself. Halving the leg length divides the similar speed by 1.41, not
    by 2 -- which is why small robots look hurried at speeds a large one strolls
    through, and why a linear rescaling of a gait is wrong in a way no amount of
    joint retargeting repairs.

    Args:
        speed: Speed on the source system, in m/s.
        from_leg_length: Characteristic leg length of the source, in metres.
        to_leg_length: Characteristic leg length of the target, in metres.

    Returns:
        The dynamically similar speed on the target system, in m/s.
    """
    if from_leg_length <= 0.0 or to_leg_length <= 0.0:
        raise ValueError(
            f"leg lengths must be positive, got {from_leg_length} and {to_leg_length}"
        )
    return speed * math.sqrt(to_leg_length / from_leg_length)


def froude_matched_time(
    duration: float, from_leg_length: float, to_leg_length: float
) -> float:
    """Duration on a second system dynamically similar to ``duration`` on the first.

    Times scale as :math:`\\sqrt{l/g}`, the pendulum period. A gait cycle that
    takes one second on a large robot takes less on a small one, and a contact
    schedule copied without this correction asks the student to hold each phase
    too long.

    Needed by the contact-schedule teacher signal, which is otherwise the one
    candidate that carries no length information at all.
    """
    if from_leg_length <= 0.0 or to_leg_length <= 0.0:
        raise ValueError(
            f"leg lengths must be positive, got {from_leg_length} and {to_leg_length}"
        )
    return duration * math.sqrt(to_leg_length / from_leg_length)


__all__ = [
    "GRAVITY",
    "froude_matched_speed",
    "froude_matched_time",
    "froude_number",
]
