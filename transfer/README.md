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

The centroidal signal won, as a **contract**: the short future of `z*`. The
sections above record how the question was first framed. What runs now:

| Part | Where | State |
| --- | --- | --- |
| The skill state `z*` (27 dimensionless numbers) | `common/skill_state.py`, `transfer/latent/skill_torch.py` | Certified against Isaac's own quantities; mirror, gauge and similarity tested. |
| Generators, one per skill (plans of `z*`) | `transfer/latent/contract.py`, `scripts/train_contract.py` | `flat_dagger2` and `rough_b4` (noise-conditional) are the incumbents. They are trained on the G1's expert only. |
| Estimator, per body | `proprioceptive_skill_state` | `z*` from the body's kinematics, with the height measured from the sole's lowest corner; the loads come from a learned encoder. |
| Trackers, per body | `tasks/tracking/`, `g1/tasks/g1_track`, `nao/tasks/nao_track` | A plain MLP trained on a procedural family of physical plans, with a mirror loss, deployed symmetrized. The incumbents are G1 `r3_6194` and NAO `r1_3498`. |
| The ladder and the competence map | `scripts/zero_shot_ladder.sh`, `evaluate_contract.py`, `evaluate_competence.py`, `compare_competence.py` | L1 (walk) to L4 (the published skill, autonomous reference); grid and corner probes with bootstrap comparisons. |
| Decoder v2 (not adopted) | `tasks/tracking/physics_prior.py`, `physics_cfg.py`, `latent/physics_model.py`, `latent/core_net.py` (on `feat/decoder-core`) | An IK and feed-forward prior with an RL residual, physics heads, a physics critic, and a deep-learning core. |
| Warm starts at equal competence | `scripts/distill_tracker.py`, `teacher_in_physics.py`, `warm_study.sh`, `ppo_start_ablation.sh` | DAgger from the incumbent, gated on the student's terminations, then PPO from the incumbent's adapted learning rate. |
| A learned free latent `ζ` | `latent/free_latent.py`, `scripts/train_free_latent.py` | Shared across bodies and blind to the body; informative on the G1, barely on the NAO. Not in the contract yet. |

Code lives in `source/humanoid_transfer/humanoid_transfer/transfer/`; the scripts
in `transfer/scripts/`; every result as JSON in `transfer/results/`.

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

## Where it stands

1. **Done.**
   - The teacher, and the walking tasks on both bodies.
   - The contract: generators trained on the G1's data only, and trackers trained on
     a procedural family of physical plans only.
   - Both of Isaac Lab's published G1 locomotion policies, transferred to the NAO
     zero-shot: 0 falls on either body, on the deployed path.
2. **Tried and not adopted.**
   - Decoder v2: the physics prior and the deep-learning core.
   - At equal competence both are robust on rough (0 falls, against 62 of 225 for a
     control distilled the same way). They shorten the stride, though, and the walk
     falls below S0's speed (0.81 and 0.85 of the teacher's). The full log is in
     [`docs/skill_space.md`](docs/skill_space.md) §9.1.
3. **Next.**
   - The new decoders' stride shortfall against the plan.
   - Seeds for every comparison.
   - `ζ` in the contract.
   - Torque-based loads.
   - The theory section and baselines from the literature.

The plan and the threats to the result are in [`docs/plan.md`](docs/plan.md).
