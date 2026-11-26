"""A tracker's reward for keeping each foot's contact when its plan does.

The contract's likelihood (:func:`~.mdp.contract_likelihood`) scores the feet's loads as
one field among nine, in tolerances of the plan's spread. Its plans taper each stance's
load to zero, so a foot left lightly loaded past its planned lift-off (or set down early)
costs almost nothing there: measured on the G1's tracker over the probe map, the load
field's error in its failing mid-duty probes is 0.07 tolerances, the smallest of all
fields, while its duty runs 0.06 of a cycle long at every contact threshold from 1 % to
10 % of the weight. A stance's start and end are what the plan's gait is, so this term
scores them directly: each foot earns 1 when its contact state (load above ``threshold``,
in the body's weight) is the plan's, 0 otherwise, averaged over the feet; 0 without a plan.

It is the plan's own contact schedule, not a strategy: the plan already says when each
foot is down.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from ....common.skill_state import SKILL_LAYOUT

_LOAD = SKILL_LAYOUT.slice_of("foot_load")


def contact_schedule(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.05
) -> torch.Tensor:
    """``(N,)`` the share of feet whose contact state is the plan's (0 without a plan)."""
    term = env.command_manager.get_term(command_name)
    norm = term.generator.skill_norm
    planned = norm.inverse(term.due())[:, _LOAD] > threshold
    reached = norm.inverse(term.true_skill())[:, _LOAD] > threshold
    agree = (planned == reached).float().mean(dim=-1)
    return agree * term.has_plan.float()


__all__ = ["contact_schedule"]
