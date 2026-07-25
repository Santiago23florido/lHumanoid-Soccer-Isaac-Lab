"""Direct RL environment placeholder for humanoid soccer."""

from __future__ import annotations


class HumanoidSoccerEnv:
    """Placeholder for the future Isaac Lab DirectRLEnv implementation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "HumanoidSoccerEnv is structure-only. Implement it with Isaac Lab DirectRLEnv "
            "after robot, field, ball, observations, rewards, and terminations are defined."
        )
