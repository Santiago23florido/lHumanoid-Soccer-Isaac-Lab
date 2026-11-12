"""How far a tracker's inputs move when another contract's statistics normalise them.

Every ``z*`` the command term hands a tracker (history, estimate, plan) is normalised with
its contract's ``skill_norm``. L4 makes the task's generator the contract, so a tracker
trained under one contract reads the task in that generator's units. A field ``x`` is seen
as ``(x - m') / s'`` instead of ``(x - m) / s``: an offset ``(m - m') / s'`` and a scale
``s / s'`` in the tracker's own units. The G1's contracts share one ``skill_norm``; the
NAO's walk contract (``walk_dagger2e``) does not (found by W13's smoke run).

    python transfer/scripts/units_shift.py --own logs/contract/walk_dagger2e/generator.pt \
        --other logs/contract/flat_dagger2/generator.pt logs/contract/rough_b4/generator.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.contract import load_contract  # noqa: E402


def shift(own: Path, other: Path) -> dict:
    """Per-component offset (in the other contract's std) and relative scale change."""
    mine = load_contract(own).generator.skill_norm
    theirs = load_contract(other).generator.skill_norm
    offset = ((mine.mean - theirs.mean) / theirs.std).abs()
    scale = (mine.std / theirs.std - 1).abs()
    names = list(SKILL_LAYOUT.component_names())
    i, j = int(offset.argmax()), int(scale.argmax())
    return {
        "largest_offset": float(offset.max()),
        "largest_offset_at": names[i],
        "mean_offset": float(offset.mean()),
        "largest_scale_change": float(scale.max()),
        "largest_scale_change_at": names[j],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--other", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    for other in args.other:
        r = shift(args.own, other)
        print(
            f"{other.parent.name:16s} offset up to {r['largest_offset']:.3f} std"
            f" ({r['largest_offset_at']}), mean {r['mean_offset']:.3f};"
            f" scale up to {100 * r['largest_scale_change']:.1f} %"
            f" ({r['largest_scale_change_at']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
