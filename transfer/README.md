# `transfer/` — teaching one robot through the other

The research question:

> A skill is learned on a robot capable enough to learn it. What has to happen
> for that skill to end up on a robot that is not?

The two embodiment tracks exist so that this one has something to work with.

## Why it is a question and not an engineering task

The obvious answer — retarget the joint trajectories and fine-tune — assumes the
gap is a matter of scale. Three things break independently.

**Kinematics.** The teacher has joints the student does not. Worse than missing:
the G1 has two independent hip-yaw joints where the NAO drives both hips from a
single motor through a mimic constraint. A differential command there is a
configuration the student *cannot reach*, not one it is bad at reaching.

**Scale.** Froude matching gives `v_student = v_teacher · √(l_student/l_teacher)`.
Copying the teacher's speed asks the student to move at roughly 1.7 times its
dynamically similar pace; copying its joint angles gives a different posture,
because equal angles on unequal segments do not give equal geometry.

**Feasibility.** A gait the G1 executes comfortably can imply centre-of-mass
states outside the NAO's capturable envelope in some directions and well inside
it in others — a factor of **1.87** between its weakest and strongest direction.
There the teacher is not hard to follow; it is asking for something no
controller on that body can do.

So a faithful copy of the teacher is the wrong target, and the research question
is what the right one is.

## Hypothesis

> What transfers across an embodiment gap is *inversely* related to how much of
> the teacher's body it describes.

Three candidate signals, in decreasing order of information:

| Signal | Carries | Why it might work |
| --- | --- | --- |
| Joint targets | The full configuration through the correspondence map | It is what "imitation" usually means; the baseline a reader will ask for |
| **Centroidal** | CoM trajectory and centroidal momentum, scaled | Invariant to joint count; what the LIPM says actually governs balance |
| Contact schedule | When each foot lands and where, scaled | If this works as well, most of what a walking teacher provides is a gait clock — a claim about what a skill *is* |

The publishable content is the comparison, not any one of them working. A
negative result would be equally informative and is a real possibility.

## The feasibility mask

```
L = λ_task · L_task  +  λ_im · w(s) · L_imitate
```

`w(s)` is full where the teacher asks for something comfortably inside the
student's capturable envelope and decays to zero at the boundary.

This comes directly from the balance work, which found that a *penalty* prices a
behaviour while only a *constraint* removes it: soft foot penalties left 51 of
51 policies stepping, and only a hard termination stopped it. Whether that
carries from a reward term to a loss weight is an open question, and the
hard-cut variant is the control condition.

## What is implemented

| Module | State |
| --- | --- |
| `correspondence.py` | Role map done and tested. Value map deliberately absent — it is the experiment. |
| `feasibility.py` | Done, tested. Masks the teacher where it exceeds the student's envelope. |
| `distillation.py` | Configuration only. The loss terms need the student environment first. |
| Student walking task | **Not started.** The critical path. |

The losses are unwritten on purpose. Writing a reward term against an
environment that does not exist is how the balance task acquired a
centre-of-mass height term that computed 0.00015 instead of 0.149 and ran
without error for weeks.

Code lives in `source/humanoid_transfer/humanoid_transfer/transfer/`.

## Order of work

1. **Teacher.** Train `G1Walk-Teacher-v0` to a stable gait. Unblocked.
2. **Rollout capture.** Record joint states, CoM, centroidal momentum, contacts.
3. **Student task.** A NAO *walking* environment. Does not exist; the balance
   task is not one, and its zero-step termination is exactly wrong for a gait.
4. **Baseline.** Train the student with `λ_im = 0`. Without this number none of
   the comparisons mean anything, and it is the one most likely to be skipped.
5. **The comparison.** Three signals × masked/unmasked × 3 seeds.
6. **Ablate the mask** on the best signal.

Step 3 gates everything after it.

Full plan and threats to the result: [`docs/plan.md`](docs/plan.md).
