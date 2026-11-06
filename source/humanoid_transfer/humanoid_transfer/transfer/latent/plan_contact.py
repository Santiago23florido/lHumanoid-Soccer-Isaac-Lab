"""A plan's contacts made consistent with its feet: a loaded foot rests on the ground.

``z*`` carries each foot's load and its position as separate fields, and the generator
models both. In the teachers' own gaits the two agree
(``transfer/scripts/plan_contact.py``, ``transfer/results/plan_contact.json``).
- *The teachers.* A loaded foot rests on the ground: a median 0.000 ``L``, 0.006 (flat) and
  0.013 (rough) at the 90th percentile.
- *The plans the G1 was given.* A loaded foot sits at a median 0.0025 (flat) and 0.0050
  (rough) ``L``, and at 0.018 and 0.029 at the 90th percentile. In light contact it sits at
  0.013--0.014, against the teachers' 0.004--0.007.

Where they disagree, a tracker that follows the foot's position leaves the ground. On
rough's plans the G1 lifts off earlier the higher the plan holds the still-loaded foot
before its planned lift-off (Spearman −0.35), and lands later the higher it places the
first loaded samples (+0.49). On flat's plans it does the same (−0.27, +0.53), but flat's
plans hold their loaded feet lower.

:func:`ground_loaded_feet` puts every foot a plan loads (above the contact threshold) on
the ground, by its height alone. Its horizontal position, velocity, load and every other
field are kept. It enforces the contact a load already implies; it does not say how the
body should move.
"""

from __future__ import annotations

import torch

from ...common.skill_state import DEFAULT_CONTACT_THRESHOLD, SKILL_LAYOUT

PLAN_CONTACTS = ("as_drawn", "ground")
_HEIGHT = SKILL_LAYOUT.slice_of("height").start
_FEET = SKILL_LAYOUT.slice_of("foot_position").start
_LOAD = SKILL_LAYOUT.slice_of("foot_load")


def ground_loaded_feet(
    rows: torch.Tensor, norm, threshold: float = DEFAULT_CONTACT_THRESHOLD
) -> torch.Tensor:
    """``rows`` (normalised ``z*``, ``(..., 27)``) with each loaded foot's sole at height 0.

    ``norm`` is the generator's :class:`~.model.Normaliser`. A foot is loaded where its
    load share exceeds ``threshold``; there its vertical position from the CoM becomes
    ``-height``, so the sole (``height + foot_z``) rests on the ground. Every other entry is
    returned exactly as given.
    """
    physical = norm.inverse(rows)
    load = physical[..., _LOAD]
    grounded = physical.clone()
    out = rows.clone()
    for side in range(2):
        column = _FEET + 3 * side + 2
        grounded[..., column] = -physical[..., _HEIGHT]
        normalised = norm(grounded)[..., column]
        out[..., column] = torch.where(load[..., side] > threshold, normalised, rows[..., column])
    return out


__all__ = ["PLAN_CONTACTS", "ground_loaded_feet"]
