"""Isaac Lab's rough-terrain G1 task on flat ground: the zero-shot study's blind task.

``transfer/docs/skill_space.md`` §8 tests zero-shot transfer on a task whose
gait nobody looked at before the trackers were frozen. Isaac Lab publishes a
checkpoint of ``Isaac-Velocity-Rough-G1-v0``; its policy reads, besides the
123 numbers of the flat task, a height scan of the ground under the torso.
This is that task unchanged -- robot, actions, observations, scanner,
rewards -- with the terrain generator replaced by a plane, so the published
policy runs as trained and its walk can be captured on the ground every
other capture of the study uses. On a plane the scan reads a constant.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
)


@configclass
class G1RoughOnFlatEnvCfg(G1RoughEnvCfg):
    """The rough task, scanner and all, on a plane."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class G1RoughOnFlatEnvCfg_PLAY(G1RoughOnFlatEnvCfg):
    """Replay: few environments, no noise, no pushes (as Isaac Lab's own play variant)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
