"""Replay a trained policy for this project's tasks with RSL-RL.

A thin wrapper over Isaac Lab's own ``rsl_rl/play.py``. It exists for one
reason: Isaac Lab discovers tasks by importing ``isaaclab_tasks``, which knows
nothing about this external extension, so the task ids have to be registered
before ``gym.make`` is reached. Importing ``humanoid_transfer.tasks`` does
that.

Everything else is delegated, so the flags are Isaac Lab's and stay correct
when Isaac Lab changes them.

Usage::

    python scripts/train.py --task NaoStand-Direct-v0 --headless \
        --num_envs 4096 --max_iterations 1500 --seed 1

Logs land in ``logs/rsl_rl/<experiment_name>/<timestamp>/`` relative to the
working directory, so run it from the repository root unless you mean
otherwise.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

ISAACLAB_ROOT = Path(r"C:\IsaacLab")
"""Where Isaac Lab is checked out.

Overridable with the ISAACLAB_PATH environment variable, since this is the one
machine-specific path in the project.
"""


def _resolve_player() -> Path:
    import os

    root = Path(os.environ.get("ISAACLAB_PATH", ISAACLAB_ROOT))
    trainer = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    if not trainer.is_file():
        raise SystemExit(
            f"Isaac Lab trainer not found at {trainer}.\n"
            "Set ISAACLAB_PATH to your Isaac Lab checkout."
        )
    return trainer


def main() -> None:
    player = _resolve_player()

    # Registration must happen before Isaac Lab resolves the task id, and it
    # pulls in no Isaac Lab code of its own: the environment classes are named
    # by string and imported only when a task is actually instantiated.
    import humanoid_transfer.tasks  # noqa: F401

    # Isaac Lab's scripts import cli_args as a sibling module, so their
    # own directory has to be importable before runpy executes them.
    if str(player.parent) not in sys.path:
        sys.path.insert(0, str(player.parent))

    sys.argv[0] = str(player)
    runpy.run_path(str(player), run_name="__main__")


if __name__ == "__main__":
    main()
