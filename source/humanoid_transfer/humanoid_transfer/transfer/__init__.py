"""Cross-embodiment skill transfer --- the research question.

The two embodiment packages exist so that this one has something to work with.
A skill is learned on the source robot, where it is learnable, and the problem
addressed here is what has to happen for it to end up on the target robot, which
cannot simply execute it.

That gap is not a matter of rescaling. The source and target differ in joint
count, in segment lengths, in mass distribution, in sign conventions, and
--- the part that matters --- in what the target's body can do at all. The NAO's
elbow cannot straighten, so a faithful copy of the teacher is not merely a hard
target, it is an impossible one.

A distinction runs through this package and is worth stating before the module
list, because getting it wrong once already cost a rewrite: **what the body
cannot do is a fair reason to ignore the teacher; what the body does badly is
the thing being learned.** Weighting the teacher down by the second teaches the
student to avoid the skill. Joint limits belong to the first. Capturability,
stepping, contact and velocity belong to the second, and are reported rather
than imposed.

Contents:

``correspondence``
    Which joint on one robot stands for which on the other, with the sign that
    makes the angle mean the same thing, and what the map cannot express.
``feasibility``
    Whether the target's joints can reach a configuration at all, so the
    teacher signal can be ignored where it asks for the impossible. Also
    reports the zero-step margin a gait keeps, as a measurement.
``embodiment``
    Each body as its dimensionless groups, and the similarity defect between
    two of them: which aspects of the student's body the skill will strain.
``distillation``
    The student objective.

See ``docs/transfer.md`` for the research plan and the open questions.
"""
