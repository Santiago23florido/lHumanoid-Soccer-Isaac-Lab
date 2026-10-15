"""The NAO tracker's mirror augmentation, bound to the NAO's measured mirror.

rsl-rl resolves the augmentation function by name, so the body's mirror
(``nao/body/mirror.json``, from ``transfer/scripts/measure_mirror.py``, with the
right hip yaw-pitch recorded as the left's mimic) is bound here.
"""

from __future__ import annotations

from pathlib import Path

from ....transfer.tasks.tracking.symmetry import mirror_augmentation

REPO_ROOT = Path(__file__).resolve().parents[6]
MIRROR = REPO_ROOT / "nao" / "body" / "mirror.json"


def augment(env, obs=None, actions=None):
    """Each sample and its mirror image, for rsl-rl's PPO."""
    return mirror_augmentation(env, obs, actions, mirror=MIRROR)


__all__ = ["MIRROR", "augment"]
