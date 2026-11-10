"""A decoder's base tasks, read from ``transfer/scripts/base_task_set.py``'s file.

``ReferenceCommandCfg.tasks`` takes one entry per further task. The file names each task's
expert capture once. A body with states of its own in it (the G1, whose experts these are)
starts from them; another body (the NAO) seeds its history from the same capture's ``z*``
windows. The held-out task is refused here as well, whatever the file says.
"""

from __future__ import annotations

import json
from pathlib import Path

HELD_OUT = "rough"


def task_specs(path: str | Path, root: str | Path, own_states: bool) -> list[dict]:
    """``ReferenceCommandCfg.tasks`` from a task-set file; paths resolved against ``root``."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("held_out") != HELD_OUT:
        raise ValueError(f"{path}: the task set does not declare {HELD_OUT!r} held out")
    specs = []
    for task in report["tasks"]:
        named = (task["name"], task["contract"], task["dart"])
        if any(HELD_OUT in str(value) for value in named):
            raise ValueError(f"{path}: {task['name']} names the held-out task")
        dart = str(Path(root) / task["dart"])
        spec = {
            "name": task["name"],
            "contract": str(Path(root) / task["contract"]),
            "ranges": {k: tuple(v) for k, v in task["ranges"].items()},
        }
        spec["starts" if own_states else "windows"] = dart
        specs.append(spec)
    return specs


def task_weights(text: str) -> list[float] | None:
    """``HUMANOID_TASK_WEIGHTS``: comma-separated, task 0 first; empty for uniform."""
    return [float(v) for v in text.split(",")] if text.strip() else None


__all__ = ["HELD_OUT", "task_specs", "task_weights"]
