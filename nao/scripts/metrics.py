"""Simulation-independent recording and validation for the PD standing baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path


def validate_run(duration: float, settle_time: float, push_velocity: float, push_at: float) -> None:
    """Reject runs that cannot measure a full baseline or deliver their requested push."""
    if not all(math.isfinite(v) for v in (duration, settle_time, push_velocity, push_at)):
        raise ValueError("Duration, settling time and push parameters must be finite.")
    if duration <= 0.0 or not 0.0 <= settle_time < duration:
        raise ValueError("Require 0 <= --settle-time < --duration.")
    if push_velocity != 0.0 and not settle_time <= push_at < duration:
        raise ValueError("Require --settle-time <= --push-at < --duration for a push.")


def minimum_margin(values: list[float], lower: float, upper: float) -> float:
    """Worst signed interval margin across ALL samples, including asymmetric bounds."""
    if not values:
        raise ValueError("A margin requires at least one sample.")
    return min(min(value - lower, upper - value) for value in values)


class Recorder:
    """Keep time-aligned diagnostics, with missing quasi-static CoP samples explicit."""

    def __init__(self) -> None:
        self.time: list[float] = []
        self.com_offset: list[float] = []
        self.dcm_offset: list[float] = []
        self.cop_offset: list[float | None] = []
        self.base_height: list[float] = []
        self.tilt: list[float] = []
        self.ankle_torque: list[float] = []
        self.fell_at: float | None = None
        self.recovered_at: float | None = None
        self.pushed_at: float | None = None
        self.completed = False

    @staticmethod
    def _summarise(values: list[float]) -> str:
        if not values:
            return "    (no samples)"
        peak = max(values, key=abs)
        mean = sum(values) / len(values)
        return f"mean {mean:+.4f}   peak {peak:+.4f}"

    def export(self, path: Path, parameters: dict, support_x: tuple[float, float]) -> None:
        """Write SI units and provenance before Kit shuts down the Python process."""
        payload = {
            "schema_version": 1,
            "parameters": parameters,
            "notes": {
                "cop": "Quasi-static ankle-wrench estimate; unreliable during impacts/flight.",
                "torque": "Clipped implicit-PD estimate, not measured motor torque.",
                "margin": "Sagittal nominal double-support interval, not live contact geometry.",
                "fall": "Height/tilt checked from the first physics step, including settling.",
            },
            "completed": self.completed,
            "upright": self.completed and self.fell_at is None and bool(self.time),
            "fell_at_s": self.fell_at,
            "recovered_at_s": self.recovered_at,
            "pushed_at_s": self.pushed_at,
            "minimum_dcm_margin_m": (
                minimum_margin(self.dcm_offset, *support_x) if self.dcm_offset else None
            ),
            "samples": {
                "time_s": self.time,
                "com_x_m": self.com_offset,
                "dcm_x_m": self.dcm_offset,
                "cop_x_quasistatic_m": self.cop_offset,
                "base_height_m": self.base_height,
                "gravity_b_z": self.tilt,
                "ankle_torque_estimate_fraction": self.ankle_torque,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
