"""The G1 tracker's mirror augmentation, bound to the G1's measured mirror.

rsl-rl resolves the augmentation function by name, so the body's mirror
(``g1/body/mirror.json``, from ``transfer/scripts/measure_mirror.py``) is bound
here rather than passed in.
"""

from __future__ import annotations

from pathlib import Path

from ....transfer.tasks.tracking.symmetry import mirror_augmentation

REPO_ROOT = Path(__file__).resolve().parents[6]
MIRROR = REPO_ROOT / "g1" / "body" / "mirror.json"


def augment(env, obs=None, actions=None):
    """Each sample and its mirror image, for rsl-rl's PPO."""
    return mirror_augmentation(env, obs, actions, mirror=MIRROR)


__all__ = ["MIRROR", "augment"]
