"""The command a future achieved: hindsight relabelling of a generator's command (W17).

A generator learns ``p(future | history, command)``. Trained on the command an expert was
*given*, it learns the expert's tracking error as part of the task: the published flat expert
walks 0.73 of a 0.25 m/s command, and the walk expert 0.88 with a duty factor of 0.88
(``transfer/results/w16_task_region.json``). A multi-task generator asked for a held-out task
at those commands then plans the base experts' speed, not the command (W16).

Relabelled, the command a training future is paired with is the one it achieved: the mean,
over the next ``span`` samples inside its episode, of the CoM velocity in the heading frame
(``vx / V``, ``vy / V``) and of the base's yaw rate (``wz T``) -- the same dimensionless
units as :attr:`.sequences.Streams.command`. A command then means what the plan does, which
is how the task is defined ("track the command"); it says nothing about how.

The span is one stride of every G1 expert, so the relabelled command does not follow the
velocity's oscillation within a stride: 50 samples of the G1's control step (1.0 s; strides of
0.93-1.29 s, ``held_out_chains.py``).
"""

from __future__ import annotations

import torch

from ...common.skill_state import SKILL_LAYOUT
from .sequences import Streams

SPAN = 50
_VELOCITY = SKILL_LAYOUT.slice_of("com_velocity").start
_YAW = SKILL_LAYOUT.slice_of("base_angular_velocity").start + 2


def achieved(skill: torch.Tensor) -> torch.Tensor:
    """``(..., 3)``: forward and lateral CoM velocity and yaw rate of ``z*`` ``(..., 27)``."""
    return torch.stack([skill[..., _VELOCITY], skill[..., _VELOCITY + 1], skill[..., _YAW]], dim=-1)


def hindsight_commands(streams: Streams, span: int = SPAN) -> torch.Tensor:
    """``(steps, 3)``: per step, the mean of :func:`achieved` over its next ``span`` samples.

    The successors stop at the episode's end; a step with none keeps its own state's value.
    """
    if span < 1:
        raise ValueError("span must be at least one sample")
    values = achieved(streams.skill).double()
    total = torch.cat([values.new_zeros(1, 3), values.cumsum(dim=0)])
    steps = torch.arange(len(streams), device=values.device)
    last = torch.minimum(steps + span, streams.end.to(values.device))
    count = (last - steps).clamp_min(0)
    summed = total[last + 1] - total[steps + 1]
    mean = summed / count.clamp_min(1)[:, None]
    mean = torch.where(count[:, None] > 0, mean, values)
    return mean.to(streams.command.dtype)


def relabel(streams: Streams, span: int = SPAN) -> Streams:
    """``streams`` with every command replaced by the one its future achieved."""
    fields = {k: getattr(streams, k) for k in streams.__dataclass_fields__}
    fields["command"] = hindsight_commands(streams, span)
    return Streams(**fields)


__all__ = ["SPAN", "achieved", "hindsight_commands", "relabel"]
