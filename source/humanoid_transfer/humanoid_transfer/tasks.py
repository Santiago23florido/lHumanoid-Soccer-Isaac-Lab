"""Task registration for every embodiment.

Importing this module registers all gym ids the project defines, which is what
``scripts/train.py`` and ``scripts/play.py`` rely on: Isaac Lab discovers tasks
by importing ``isaaclab_tasks``, which knows nothing about this external
extension, so the ids have to exist before ``gym.make`` is reached.

It is a module rather than a package on purpose. Tasks live with the robot they
run on; this only pulls them in.
"""

from __future__ import annotations

from .g1 import tasks as g1_tasks  # noqa: F401
from .nao import tasks as nao_tasks  # noqa: F401
from .soccer import tasks as soccer_tasks  # noqa: F401

__all__ = ["g1_tasks", "nao_tasks", "soccer_tasks"]
