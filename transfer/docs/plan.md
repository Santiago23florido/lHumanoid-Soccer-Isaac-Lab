# Cross-embodiment skill transfer — research plan

The question this project now asks:

> A skill is learned on a robot capable enough to learn it. What has to happen
> for that skill to end up on a robot that is not?

## Why this is a question and not an engineering task

The obvious answer — retarget the joint trajectories and fine-tune — assumes the
gap between two humanoids is a matter of scale. It is not. Three things break
independently:

**Kinematics.** The teacher has joints the student does not. On a 29-DOF G1
against the NAO's 19 commanded joints, the waist and wrists have no counterpart
at all, and `hip_yaw` is worse than missing: the G1 has two independent hip-yaw
joints, the NAO drives both hips from a single motor through a mimic constraint.
A teacher that yaws its hips differentially is commanding a configuration the
student cannot reach, not one it is merely bad at reaching.

**Scale.** Dynamic similarity between legged systems requires matched Froude
numbers, `Fr = v²/(g·l)`. With `l` of 0.74 m against 0.2689 m, a teacher walking
at 0.25 m/s corresponds to a student walking at 0.15 m/s. Copy the teacher's
speed and the student is being asked to move at roughly 1.7 times its
dynamically similar pace. Copy its joint angles and the resulting posture is
wrong, because equal angles on unequal segments do not give equal geometry.

**Feasibility.** This is the one that makes the problem interesting. The NAO's
zero-step capturable envelope runs from 0.443 m/s backward to 0.827 m/s toward
the forward corners — a factor of **1.87** between its weakest and strongest
direction. A gait the G1 executes comfortably can imply centre-of-mass states
that are outside the NAO's envelope in some directions and well inside it in
others. The teacher is not merely hard to follow there; it is asking for
something that no controller on that body can do.

So a faithful copy of the teacher is the wrong target, and the research question
is what the right one is.

## Hypothesis

> What transfers across an embodiment gap is inversely related to how much of
> the teacher's body it describes.

Three candidate teacher signals, in decreasing order of information and, if the
hypothesis holds, increasing order of transfer quality:

| Signal | What it carries | Why it might fail | Why it might work |
| --- | --- | --- | --- |
| **Joint targets** | The full configuration through the correspondence map | Segment lengths differ, joint counts differ, unmapped joints are simply lost | It is what "imitation" usually means, and is the baseline a reader will ask for |
| **Centroidal** | CoM trajectory and centroidal momentum, scaled by leg length and mass | Discards how the motion is produced, which may be most of the skill | Invariant to joint count; it is what the linear inverted pendulum says actually governs balance |
| **Contact schedule** | When each foot lands and where, scaled by leg length | Almost no information; may amount to giving the student a metronome | If it works as well as centroidal, then most of what a walking teacher provides *is* a gait clock — which is a claim about what a skill is |

The publishable content is the comparison, not any one of them working. A
negative result — that joint-level imitation is no worse — would be equally
informative and is a real possibility.

## The feasibility mask

Whatever the signal, it is weighted per sample by whether the implied student
state is recoverable:

```
L = λ_task · L_task  +  λ_im · w(s) · L_imitate
```

with `w(s)` from `transfer.feasibility.imitation_mask`: full weight where the
teacher asks for something comfortably inside the student's capturable envelope,
decaying to zero at the boundary.

This is the part of the design that comes directly from the standing work. That
study found that a *penalty* prices a behaviour while only a *constraint*
removes it — soft foot penalties left 51 of 51 policies stepping, and only a
hard termination stopped it. Whether that carries from a reward term to a loss
weight is an open question. The hard-cut variant is the obvious control
condition and should be run.

The mask is a necessary condition, not a sufficient one. A state inside the
capturable envelope may still be unreachable because the NAO's actuators cannot
produce the torque. Torque feasibility is not implemented and is listed below.

## Order of work

1. **Teacher.** Train `G1Walk-Teacher-v0` to a stable gait at the Froude-matched
   speed. Nothing novel; Isaac Lab's configuration already does this.
2. **Rollout capture.** Record teacher trajectories: joint states, CoM,
   centroidal momentum, contact events. This defines what the student can be
   shown.
3. **Student task.** A NAO walking environment. This does not exist yet and is
   the largest single piece of work — the standing task is not a walking task,
   and its zero-step termination is exactly wrong for one.
4. **Baseline.** Train the student with `λ_im = 0`. Without this number none of
   the comparisons mean anything, and it is the one most likely to be skipped.
5. **The comparison.** Three signals × masked/unmasked × 3 seeds.
6. **Ablate the mask** on the best signal.

Steps 1–2 are unblocked. Step 3 gates everything after it.

## What is implemented

| Module | State |
| --- | --- |
| `common/capturability.py` | Done, tested. Embodiment-agnostic LIPM/DCM/capturability. |
| `g1/assets/g1.py` | Done. Source-robot selection, swappable to H1 in one line. |
| `g1/tasks/g1_walk/` | Done. Teacher task, Froude-capped commands. |
| `transfer/correspondence.py` | Name map done and tested. Value map deliberately absent — it is the experiment. |
| `transfer/feasibility.py` | Done, tested. |
| `transfer/distillation.py` | Configuration only. The loss terms need the student environment first. |
| Student walking task | **Not started.** The critical path. |

The distillation losses are unwritten on purpose. Writing a reward term against
an environment that does not exist is how the standing task acquired a
centre-of-mass height term that computed 0.00015 instead of 0.149 and that
nobody checked for weeks, because it ran without error the whole time.

## Threats to the result

- **The student task may be unlearnable.** If the NAO cannot walk under any
  supervision, the comparison has no content. Checking this early — can a
  student learn to walk at all with `λ_im = 0`? — is cheap and should precede
  everything else.
- **Single teacher.** One robot, one gait, one speed. Conclusions about "what
  transfers" from a sample of one embodiment pair are weak. The H1 switch exists
  to give a second pair cheaply.
- **No hardware.** As with all of this project.
- **Prior work.** Cross-embodiment transfer and policy distillation are active
  fields. Before writing anything up, search for `cross-embodiment` +
  `humanoid` + `retargeting`, and for capturability-constrained imitation
  specifically. The contribution may reduce to the feasibility mask, and it may
  already exist.

## Related documents

- [`task_nao_stand.md`](../../nao/docs/task_balance.md) — the student's balance task, and the
  source of the capturability numbers used here.
- [`architecture.md`](../../ARCHITECTURE.md) — where each package lives.
