"""Environment configuration placeholders for humanoid soccer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HumanoidSoccerEnvCfg:
    task_id: str = "HumanoidSoccer-Direct-v0"
    num_envs: int = 4096
    episode_length_s: float = 20.0
    action_scale: float = 0.5
    observation_terms: list[str] = field(
        default_factory=lambda: [
            "base_velocity",
            "projected_gravity",
            "joint_positions",
            "joint_velocities",
            "feet_contacts",
            "ball_relative_pose",
            "goal_relative_pose",
        ]
    )
    reward_terms: list[str] = field(
        default_factory=lambda: [
            "alive",
            "upright",
            "velocity_tracking",
            "ball_approach",
            "ball_control",
            "goal_progress",
        ]
    )
