"""The student objective.

Three ways of using a teacher, in increasing order of how much they respect the
embodiment gap. They are written here as one configurable objective rather than
three implementations because the experiment is the comparison between them, and
a comparison is only meaningful if everything except the term under test is
identical.

.. math::

    L = \\lambda_{\\text{task}} L_{\\text{task}}
      + \\lambda_{\\text{im}} \\, w \\, L_{\\text{imitate}}

``L_task`` is the student's own reward, the same shape as the one the standing
task already uses. ``L_imitate`` is the teacher term. ``w`` is the per-sample
feasibility weight from :mod:`feasibility`, which is what stops the teacher from
being followed into states the student cannot occupy.

Status
------
Configuration and the feasibility weighting are implemented. The loss terms
themselves are not: they need the student environment to exist first, and
writing them against an environment that has not been built is how the
zero-step task acquired a reward term that computed a number nobody had
checked. See ``docs/transfer.md`` for the order of work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TeacherSignal(str, Enum):
    """What the student is asked to match.

    The three options differ in how much of the teacher's embodiment they drag
    along, and the hypothesis of the study is that the ranking of these on the
    student is the reverse of how much information they carry.
    """

    JOINT_TARGETS = "joint_targets"
    """Match the teacher's joint angles through the correspondence map.

    The most information and the least transferable. Segment lengths differ, so
    equal joint angles do not give equal postures, and equal postures would not
    give equal dynamics anyway. Included as the baseline that a reader would
    otherwise ask for.
    """

    CENTROIDAL = "centroidal"
    """Match the teacher's centre-of-mass trajectory and centroidal momentum,
    scaled by leg length and mass.

    Discards the joint configuration entirely and keeps what the linear
    inverted pendulum says actually matters. Invariant to the joint-count
    mismatch, which is the reason to expect it to survive the gap better than
    joint matching.
    """

    CONTACT_SCHEDULE = "contact_schedule"
    """Match only the timing and placement of contacts, scaled by leg length.

    The least information: when each foot lands and where, and nothing about
    what the body does in between. If this transfers as well as the centroidal
    term, the interesting claim is that most of what a walking teacher provides
    is a gait clock, and that is a claim about what a skill *is* rather than
    about how to move one.
    """


@dataclass
class DistillationCfg:
    """Weights and choices for the student objective."""

    signal: TeacherSignal = TeacherSignal.CENTROIDAL
    """Which teacher signal to match."""

    task_weight: float = 1.0
    """Weight on the student's own reward."""

    imitation_weight: float = 0.5
    """Weight on the teacher term.

    Deliberately below the task weight. The student's own reward encodes what it
    means to stay upright on *this* robot; the teacher encodes what walking
    looked like on another one. When they disagree, the robot that has to stay
    standing is the student.
    """

    feasibility_margin: float = 0.9
    """Where the feasibility weight starts decaying. See ``feasibility``."""

    anneal_steps: int = 0
    """Environment steps over which to decay ``imitation_weight`` to zero.

    Zero disables annealing and keeps the teacher present for the whole run.

    Annealing exists because a teacher is scaffolding: useful while the student
    has no competence of its own, and a ceiling afterwards. Whether that ceiling
    is real here is measurable -- run with and without, and compare final task
    reward rather than assuming.
    """

    def imitation_weight_at(self, step: int) -> float:
        """Imitation weight after ``step`` environment steps."""
        if self.anneal_steps <= 0:
            return self.imitation_weight
        remaining = max(0.0, 1.0 - step / self.anneal_steps)
        return self.imitation_weight * remaining


__all__ = ["DistillationCfg", "TeacherSignal"]
