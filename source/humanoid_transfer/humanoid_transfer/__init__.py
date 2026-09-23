"""Humanoid soccer Isaac Lab extension scaffold."""

try:
    from .tasks.direct.humanoid_soccer import TASK_ID
except ModuleNotFoundError:
    TASK_ID = "HumanoidSoccer-Direct-v0"

__all__ = ["TASK_ID"]
