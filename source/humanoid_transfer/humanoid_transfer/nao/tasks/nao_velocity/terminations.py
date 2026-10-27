"""The oracle's fall, measured from the ground under the robot.

Kept apart from the environment configuration, which needs the Kit application, so the
tests can run it on a stand-in scene.
"""

from __future__ import annotations

import torch


def base_below(
    env, minimum_height: float, scanner: str | None = None, robot: str = "robot"
) -> torch.Tensor:
    """The trackers' fall: the base below ``minimum_height`` above the ground under it.

    Isaac Lab's ``root_height_below_minimum`` measures from the world's zero, which is the
    ground only on a plane: on the rough terrains a robot spawned in a sunken sub-terrain
    fell on its first step. With a height ``scanner`` the ground is its rays' mean, as
    Isaac Lab's ``base_height_l2`` reads it; on a plane the two agree.
    """
    height = env.scene[robot].data.root_pos_w[:, 2]
    if scanner is not None:
        height = height - env.scene[scanner].data.ray_hits_w[..., 2].mean(dim=1)
    return height < minimum_height
