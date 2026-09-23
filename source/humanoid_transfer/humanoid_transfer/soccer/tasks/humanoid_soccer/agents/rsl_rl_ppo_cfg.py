"""RSL-RL PPO runner placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HumanoidSoccerPPORunnerCfg:
    experiment_name: str = "humanoid_soccer_direct"
    max_iterations: int = 10000
    num_steps_per_env: int = 24
    save_interval: int = 200
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
