"""List the gym task ids this project registers.

Queries the gymnasium registry rather than printing a hand-written list. The
previous version was a hardcoded dictionary, which meant it happily reported
tasks that no longer existed and silently omitted ones that did -- it listed the
NAO tasks and not the G1 teacher, which is the opposite of what a discovery tool
is for.

Importing ``humanoid_transfer.tasks`` is what performs the registration, so the
output here is exactly what ``scripts/train.py`` will be able to launch.

Usage::

    python scripts/list_tasks.py            # ids and their entry points
    python scripts/list_tasks.py --verbose  # plus the config entry points
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTENSION_ROOT = _REPO_ROOT / "source" / "humanoid_transfer"
if str(_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_ROOT))

# Which embodiment each id belongs to, for grouping. Matched as a prefix.
TRACKS = {
    "NaoStand": "nao — target embodiment",
    "G1Walk": "g1 — source embodiment",
    "HumanoidSoccer": "soccer — long-horizon target",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose", action="store_true", help="Also print the config entry points."
    )
    args = parser.parse_args()

    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        print("gymnasium is not installed; no task can be registered.", file=sys.stderr)
        return 1

    before = set(gym.envs.registry)

    import humanoid_transfer.tasks  # noqa: F401  (registers the ids)

    registered = sorted(set(gym.envs.registry) - before)
    if not registered:
        print(
            "No task registered. Either the extension is not installed "
            "(`pip install -e source/humanoid_transfer`) or registration was "
            "skipped because Isaac Lab is unavailable.",
            file=sys.stderr,
        )
        return 1

    grouped: dict[str, list[str]] = {}
    for task_id in registered:
        track = next(
            (label for prefix, label in TRACKS.items() if task_id.startswith(prefix)),
            "other",
        )
        grouped.setdefault(track, []).append(task_id)

    print()
    for track in sorted(grouped):
        print(f"  {track}")
        for task_id in grouped[track]:
            spec = gym.envs.registry[task_id]
            print(f"    {task_id}")
            if args.verbose:
                for key, value in sorted(spec.kwargs.items()):
                    if key.endswith("_entry_point"):
                        print(f"        {key:<24s} {value}")
        print()

    print(f"  {len(registered)} task(s) registered.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
