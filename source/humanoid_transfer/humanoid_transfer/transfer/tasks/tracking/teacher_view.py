"""The plain tracker's view of a physics-task env: its own observation, with its own last action.

The physics task's observation groups begin with the plain tracker's terms, so a plain
tracker can be run on them -- except one. Its ``actions`` term is the physics task's last
action: a residual on the prior, in the residual's units. The plain tracker was trained,
and is deployed, on its own previous output there; shown the residual instead it acts
on an input it never saw, and so do labels drawn from it (the first two distilled
physics trackers, ``transfer/docs/skill_space.md`` §9.1).
"""

from __future__ import annotations

import torch


def term_columns(manager, group: str, name: str) -> slice:
    """Where the observation term ``name`` sits in the concatenated group ``group``."""
    start = 0
    names = manager.active_terms[group]
    for term, shape in zip(names, manager.group_obs_term_dim[group], strict=True):
        width = int(torch.tensor(shape).prod())
        if term == name:
            return slice(start, start + width)
        start += width
    raise KeyError(f"observation group {group!r} has no term {name!r}")


class TeacherView:
    """A plain tracker's observations in a physics-task env, its last action its own output."""

    def __init__(self, manager, widths: dict[str, int], num_envs: int, action_dim: int, device):
        self.widths = dict(widths)
        self.slots = {group: term_columns(manager, group, "actions") for group in self.widths}
        for group, slot in self.slots.items():
            if slot.stop > self.widths[group] or slot.stop - slot.start != action_dim:
                raise ValueError(f"{group}: the plain tracker's last action is not at {slot}")
        self.last = torch.zeros(num_envs, action_dim, device=device)

    def observation(self, obs, group: str) -> torch.Tensor:
        view = obs[group][:, : self.widths[group]].clone()
        view[:, self.slots[group]] = self.last
        return view

    def record(self, action: torch.Tensor, dones: torch.Tensor) -> None:
        """After a step: the teacher's output becomes its last action; a reset env's is zero,
        as an action manager's is."""
        self.last = torch.where(dones.bool()[:, None], 0.0, action)

    def reset(self) -> None:
        self.last.zero_()


__all__ = ["TeacherView", "term_columns"]
