"""The contact schedule a rollout followed, read off its skill state.

The skill state says which contacts are loaded *now*. What it does not say is
the schedule: when each contact will next change, and where the next landing
will be. That is the third signal the transfer plan names -- after joint
targets and centroidal state -- and the one no instantaneous state can carry.

Nothing here assumes a gait. A contact is an event, not a point on a cycle:
a part is loaded or not, and the labels are *time to its next change* and
*where it lands next*. A kick, a get-up or a turn has a schedule as much as a
walk does; only the parts involved differ. Today the skill state carries the
feet, so the parts are the feet.

Labels come from the recorded future, and only from the skill state, so a
model trained on them learns nothing about the body that ``z*`` does not
already show. Per stream step ``s`` and part ``p``:

``contact``
    Load share above the threshold of :func:`~...common.skill_state.contact_state`.
``to_change``
    Control steps until the part's contact next differs from the step before,
    within the episode; ``inf`` if it never does before the episode ends.
``landing``
    Where the part's sole is, relative to the CoM, in the heading frame and in
    leg lengths, at its next touchdown -- the ``xy`` of ``foot_position``.
``landing_known``
    Whether a touchdown follows in the episode.

A change between an episode's first step and its second is not an event: the
first step's state can be read before the simulator applies the reset pose
(:mod:`.sequences`), so a "change" there is the reset, not the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...common.skill_state import DEFAULT_CONTACT_THRESHOLD, SKILL_LAYOUT, ZLayout
from .sequences import Streams

_NEVER = np.iinfo(np.int64).max


@dataclass
class ContactEvents:
    contact: torch.Tensor
    """``(S, P)`` bool."""

    to_change: torch.Tensor
    """``(S, P)`` float control steps, ``inf`` if no change follows in the episode."""

    landing: torch.Tensor
    """``(S, P, 2)``, leg lengths: next touchdown's sole relative to the CoM, heading frame."""

    landing_known: torch.Tensor
    """``(S, P)`` bool."""

    remaining: torch.Tensor
    """``(S,)`` control steps left in the episode after ``s``: how far the future was observed."""

    def to(self, device: torch.device | str) -> ContactEvents:
        return ContactEvents(**{k: getattr(self, k).to(device) for k in self.__dataclass_fields__})

    @property
    def parts(self) -> int:
        return self.contact.shape[1]


def _next_index(flags: np.ndarray) -> np.ndarray:
    """Per row ``s``: the smallest ``s' > s`` where ``flags`` is set, ``_NEVER`` if none."""
    index = np.where(flags, np.arange(flags.shape[0])[:, None], _NEVER)
    at_or_after = np.minimum.accumulate(index[::-1], axis=0)[::-1]
    after = np.full_like(at_or_after, _NEVER)
    after[:-1] = at_or_after[1:]
    return after


def contact_events(
    streams: Streams,
    threshold: float = DEFAULT_CONTACT_THRESHOLD,
    layout: ZLayout = SKILL_LAYOUT,
) -> ContactEvents:
    """Label every stream step with its contacts' next change and next landing."""
    fields = layout.unpack(streams.skill.cpu().numpy())
    contact = fields["foot_load"] > threshold
    feet = fields["foot_position"][..., :2]
    start = streams.start.cpu().numpy()
    end = streams.end.cpu().numpy()
    steps = np.arange(contact.shape[0])

    # A change at s compares s with s - 1; both must be trustworthy steps of one episode.
    change = np.zeros_like(contact)
    change[1:] = contact[1:] != contact[:-1]
    change[steps <= start] = False
    landed = change & contact

    upcoming = _next_index(change)
    within = upcoming <= end[:, None]
    to_change = np.where(within, upcoming - steps[:, None], np.inf)

    landing_at = _next_index(landed)
    known = landing_at <= end[:, None]
    landing = np.zeros_like(feet)
    rows, parts = np.nonzero(known)
    landing[rows, parts] = feet[landing_at[rows, parts], parts]

    return ContactEvents(
        contact=torch.as_tensor(contact),
        to_change=torch.as_tensor(to_change, dtype=torch.float32),
        landing=torch.as_tensor(landing, dtype=torch.float32),
        landing_known=torch.as_tensor(known),
        remaining=torch.as_tensor(end - steps, dtype=torch.float32),
    ).to(streams.skill.device)


__all__ = ["ContactEvents", "contact_events"]
