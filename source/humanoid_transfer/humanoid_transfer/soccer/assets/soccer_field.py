"""Soccer field and ball asset placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoccerFieldAssetPlan:
    field_length_m: float = 12.0
    field_width_m: float = 8.0
    goal_width_m: float = 2.0
    ball_radius_m: float = 0.11
    ball_mass_kg: float = 0.43
