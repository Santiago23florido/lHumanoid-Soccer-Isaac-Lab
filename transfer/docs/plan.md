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

**Feasibility.** This is the one that makes the problem interesting, and the one
it is easiest to get wrong. The question has to be *can the student's body
occupy this state*, not *would the student's body handle this state well*. The
first is a property of the robot and is a fair reason to ignore the teacher; the
second is the thing being learned, and discounting the teacher by it teaches the
student to avoid the skill.

The honest version for this pair is joint range. The NAO's elbow cannot
straighten — `LElbowRoll` spans [−1.545, −0.035], `RElbowRoll` spans
[+0.035, +1.545], and zero is in neither — so a straight-arm teacher pose is
unreachable no matter what the policy does. No training buys it and no
retargeting argues with it.

A **sign convention** is part of this and is not part of the experiment. In the
NAO's URDF the left and right axis vectors are identical and the mirroring lives
in the parent link's rotated origin, so roll and yaw joints have negated ranges
across sides while pitch joints do not. The NAO puts elbow flexion on a roll
axis and the G1 puts it on a pitch axis, so copying that angle across unsigned
sends +0.87 rad to a joint bounded at −0.035: outside for every sample, by
0.905 rad, with nothing raised. The command would simply clamp, holding the left
arm straight and the right at 50° of flexion. `correspondence.JOINT_SIGN` fixes
it and `scripts/check_reachability.py` is what would catch the next one.

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

Whatever the signal, it is weighted per sample by whether the student's **body**
can occupy the configuration at all:

```
L = λ_task · L_task  +  λ_im · w(s) · L_imitate
```

with `w(s)` from `transfer.feasibility.reachability_mask`: full weight while
every mapped joint sits inside the NAO's range, decaying to zero as the worst
joint goes 0.05 rad past its bound.

`w(s)` constrains the body and says nothing about behaviour. No term about
stepping, contact, velocity or momentum — those are things the student can
physically do, so they belong to `L_task` and to the dynamics. For this pair the
criterion binds hard and for a reason no retargeting can argue with: the NAO's
elbow cannot straighten. `LElbowRoll` is bounded to [−1.545, −0.035] and
`RElbowRoll` to [+0.035, +1.545], so zero is in neither range.

### What this replaced, and why

The earlier version weighted by **zero-step capturability**: full weight where
the implied student state could be brought to rest without stepping. That was
inherited from the standing task, where it is correct — "recover without
stepping" *is* what `nao_stand` poses, and the envelope is its definition. The
standing task keeps it.

It is wrong here, structurally rather than by degree. Walking *is* the
divergence of the DCM from the current support polygon: `ξ̇ = ω₀(ξ − p)`, and a
gait continues because the next footstep moves `p`. A state that needs a step is
not infeasible, it is a *walking* state. So the criterion did not measure what
the student could reach; it measured how close to standing still the teacher
was, and discounted the teacher exactly where the teacher was demonstrating the
skill.

It was also inert. Measured on the 12 800 captured samples at the Froude-matched
speed:

| quantity | value |
| --- | --- |
| capturable fraction, median | 0.213 |
| capturable fraction, maximum | 0.564 |
| samples down-weighted at all | **0 of 12 800** |

The margin would not start biting until 0.493 m/s, 3.3× the speed the experiment
runs at. And it was computed on the **double-stance** polygon while a walking
robot is in single support for most of a cycle, which makes it 2.58× too
generous laterally — the one-foot lateral bound is 0.240 m/s, not 0.620 m/s.

Wrong in principle, doing nothing in practice, on the wrong geometry. It had
been listed here as the part of the design that carried the contribution.

The standing study's finding — that a *penalty* prices a behaviour while only a
*constraint* removes it, 51 of 51 policies stepping under soft penalties — still
stands. It is a finding about that task. Importing its *criterion* into a task
whose objective is the opposite was the error; the finding did not travel with
it.

Capturability is not deleted. `feasibility.standing_margin` reports the margin a
learned gait *chose* to keep, which is a result: it says the policy stayed
recoverable without being told to. Requiring 0.21 would turn the same number
into an assumption and the finding circular.

### Still not addressed

Torque feasibility. A configuration inside the joint limits may still be
unreachable because the actuators cannot hold it — `LAnklePitch` has 3.023 N·m
against a 12.98 N·m/rad destabilising gravitational stiffness. That is a limit of
the body and belongs in `feasibility`; it is not implemented.

## Order of work

1. **Teacher.** Train `G1Walk-Teacher-v0` to a stable gait at the Froude-matched
   speed. Nothing novel; Isaac Lab's configuration already does this.
2. **Rollout capture.** Record teacher trajectories: joint states, CoM,
   centroidal momentum, contact events. This defines what the student can be
   shown.
3. **Student task.** A NAO walking environment. This does not exist yet and is
   the largest single piece of work. It must be built as a new task, not as a
   variant of `nao_stand`: that task terminates on a step, which defines it and
   is exactly wrong here. The student's objective is open — track the commanded
   velocity, do not fall — and carries no condition on whether, when or how it
   steps. Whether a gait emerges is the question, so it cannot be an input.
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
| `transfer/correspondence.py` | Name map and **sign** done and tested. Scale map deliberately absent — it is the experiment. |
| `transfer/feasibility.py` | Reachability criterion done and tested. Torque feasibility not implemented. |
| `transfer/scripts/check_reachability.py` | Done. Run it on a rollout before training on it. |
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
