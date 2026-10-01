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

**Feasibility.** The NAO's elbow cannot straighten: `LElbowRoll` spans
[−1.545, −0.035] and `RElbowRoll` spans [+0.035, +1.545], so zero is in neither
range. There the teacher is not hard to follow; it is asking for a configuration
that body cannot occupy, and no amount of training buys it.

The question has to be *can the student's body occupy this state*, not *would it
handle this state well*. The first is a property of the robot. The second is the
thing being learned, and discounting the teacher by it teaches the student to
avoid the skill.

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

## The reachability mask

```
L = λ_task · L_task  +  λ_im · w(s) · L_imitate
```

`w(s)` is full while every mapped joint sits inside the NAO's range and decays to
zero as the worst joint goes 0.05 rad past its bound. It constrains the **body**
and carries no term about stepping, contact or velocity — those are things the
student can physically do, so they belong to `L_task` and to the dynamics.

Run it against a rollout before training on it:

```bash
python transfer/scripts/check_reachability.py
```

### This replaced a zero-step capturability mask

That mask weighted the teacher by whether the implied student state could be
brought to rest without stepping. It was inherited from the balance task, where
it is correct — "recover without stepping" *is* what that task poses.

Here it is wrong, structurally. Walking *is* the DCM diverging from the current
support polygon, `ξ̇ = ω₀(ξ − p)`, with the next footstep moving `p`. A state
needing a step is not infeasible, it is a *walking* state. The mask measured how
close to standing still the teacher was and discounted it exactly where it was
demonstrating the skill.

It was also inert. On 12 800 captured samples at the Froude-matched speed the
capturable fraction peaked at 0.564, so a mask decaying from 0.9 down-weighted
**0 of 12 800**. It would not bite until 0.493 m/s, 3.3× the experiment's speed
— and it used the double-stance polygon while a walking robot is in single
support most of the time, making it 2.58× too generous laterally.

The balance finding that a *penalty* prices a behaviour while only a
*constraint* removes it — 51 of 51 policies stepping under soft penalties —
still stands. It is a finding about that task. Importing its criterion into a
task whose objective is the opposite was the error.

Capturability survives as instrumentation: `feasibility.standing_margin` reports
the margin a learned gait *chose* to keep. Requiring it would make the same
number an assumption.

## What is implemented

| Module | State |
| --- | --- |
| `correspondence.py` | Role map and sign done and tested. Scale map deliberately absent — it is the experiment. |
| `feasibility.py` | Reachability done and tested. Torque feasibility not implemented. |
| `scripts/check_reachability.py` | Done. Caught a sign error that affected 100% of samples. |
| `distillation.py` | Configuration only. The loss terms need the student environment first. |
| `embodiment.py` | Each body as dimensionless groups and the defect between them. |
| `latent/` | The shared latent space, G1 side: anchored to the skill state, distilled, walks closed-loop. Not pinned: stride timing misses its criterion. |
| `scripts/encode_skill.py`, `train_latent.py`, `evaluate_latent.py` | Certify the skill state, train the latent model, compare policies closed-loop. |
| Student walking task | **Not started.** The critical path. |

The losses are unwritten on purpose. Writing a reward term against an
environment that does not exist is how the balance task acquired a
centre-of-mass height term that computed 0.00015 instead of 0.149 and ran
without error for weeks.

Code lives in `source/humanoid_transfer/humanoid_transfer/transfer/`.

## What this starts from

| | |
| --- | --- |
| Teacher | **Fixed.** `g1/policy/teacher_policy.pt`, hashed and pinned. Every comparison runs against this one gait, or it measures the teacher instead of the transfer. |
| Student | **Its body only.** `nao/assets` and `common/` are shared; `nao/tasks/nao_stand` and `nao/controllers` are frozen out and enforced by a test. |

The NAO's balance task is finished, kept and registered. It is not the starting
point for a gait: its zero-step termination defines a problem whose objective is
the opposite. The same criterion has leaked into the walking work twice — once as
the imitation weight here, once as the justification for the teacher's command
speed — and both times nothing raised. See
[`../ARCHITECTURE.md`](../ARCHITECTURE.md#boundaries).

## The shared skill space

The centroidal signal, made exact: a 27-number state in each body's own units
that every humanoid shares by construction, and a learned latent space
anchored to it, with a per-body encoder and decoder around a skill every body
shares. Full account and results: [`docs/skill_space.md`](docs/skill_space.md).

## Order of work

1. **Teacher.** Trained, verified walking, pinned. Done.
2. **Rollout capture.** Record joint states, CoM, centroidal momentum, contacts.
3. **Student task.** A NAO *walking* environment. Does not exist. It must be a
   new task rather than a variant of the balance one, whose zero-step
   termination defines it and is exactly wrong for a gait. The objective is
   open — track the command, do not fall — with no condition on whether, when
   or how the student steps. Whether a gait emerges is the question, so it
   cannot be an input.
4. **Baseline.** Train the student with `λ_im = 0`. Without this number none of
   the comparisons mean anything, and it is the one most likely to be skipped.
5. **The comparison.** Three signals × masked/unmasked × 3 seeds.
6. **Ablate the mask** on the best signal.

Step 3 gates everything after it.

Full plan and threats to the result: [`docs/plan.md`](docs/plan.md).
