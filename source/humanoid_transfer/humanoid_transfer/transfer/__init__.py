"""Cross-embodiment skill transfer --- the research question.

The two embodiment packages exist so that this one has something to work with.
A skill is learned on the source robot, where it is learnable, and the problem
addressed here is what has to happen for it to end up on the target robot, which
cannot simply execute it.

That gap is not a matter of rescaling. The source and target differ in joint
count, in segment lengths, in mass distribution, and --- the part that matters
--- in what is dynamically feasible at all. A gait the G1 performs comfortably
may exceed the NAO's capturability envelope in some directions and sit well
inside it in others, so a faithful copy of the teacher is the wrong target.

Contents:

``correspondence``
    Which joint on one robot stands for which on the other, and what the map
    cannot express.
``feasibility``
    What the target can physically do, per direction, so the teacher signal can
    be masked where it asks for the impossible.
``distillation``
    The student objective.

See ``docs/transfer.md`` for the research plan and the open questions.
"""
