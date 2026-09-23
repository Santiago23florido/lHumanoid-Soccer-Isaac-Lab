"""Embodiment-agnostic theory.

Everything here is expressed in terms of masses, a centre of mass and a support
polygon, so it holds for any legged robot rather than for one of them. The
per-robot numbers live in each embodiment package and are fed through these
functions.

Pure NumPy: importable without Isaac Lab, which is what lets the tests check the
theory directly.
"""

from .capturability import (
    capturable_velocity,
    capturable_velocity_in_direction,
    dcm,
    lipm_omega,
    ray_box_distance,
)

__all__ = [
    "capturable_velocity",
    "capturable_velocity_in_direction",
    "dcm",
    "lipm_omega",
    "ray_box_distance",
]
