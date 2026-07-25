"""Humanoid asset placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanoidAssetPlan:
    name: str = "TBD_HUMANOID"
    usd_path: str = "assets/robots/humanoid/TBD_HUMANOID.usd"
    prim_path: str = "/World/envs/env_.*/Robot"
    notes: str = "Replace with an Isaac Lab ArticulationCfg after the robot is selected."
