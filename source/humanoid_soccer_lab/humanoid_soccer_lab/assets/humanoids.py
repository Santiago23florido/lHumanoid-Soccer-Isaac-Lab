"""Humanoid asset helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanoidAssetPlan:
    name: str = "IsaacLab_MuJoCo_Humanoid"
    usd_path: str = "${ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Humanoid/humanoid_instanceable.usd"
    prim_path: str = "/World/Humanoids/Origin.*/Robot"
    notes: str = "Built-in Isaac Lab humanoid used for first visual simulation."


def get_builtin_humanoid_cfg(prim_path: str = "/World/Humanoids/Origin.*/Robot"):
    """Return Isaac Lab's built-in humanoid ArticulationCfg.

    Import lazily so this package can still be inspected on machines that do
    not have Isaac Lab installed.
    """
    try:
        from isaaclab_assets.robots.humanoid import HUMANOID_CFG
    except ModuleNotFoundError:
        from isaaclab_assets import HUMANOID_CFG

    cfg = HUMANOID_CFG.copy()
    cfg.prim_path = prim_path
    return cfg
