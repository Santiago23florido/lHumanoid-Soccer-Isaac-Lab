"""Direct workflow tasks.

Importing this package registers every task with gymnasium, so that
``import humanoid_soccer_lab`` is enough for the Isaac Lab training scripts
to resolve a task id. Registration itself pulls in no Isaac Lab code; the
environment modules are named by string and imported only when a task is
actually instantiated.
"""

from . import humanoid_soccer, nao_stand  # noqa: F401

__all__ = ["humanoid_soccer", "nao_stand"]
