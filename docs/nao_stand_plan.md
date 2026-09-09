# Phase 1 Of Robot Learning — Plan For `NaoStand-Direct-v0`

Written for: whoever continues this repository, including future me.

The environment now instantiates, steps, and produces a measured baseline. This
plan takes it from "runs" to "a trained policy that demonstrably beats the PD",
and says up front what that is allowed to mean.

## The target, stated as a number

Everything below is judged against one protocol, the same one
`scripts/check_nao_stand_env.py --pushes` already runs: 128 environments,
600 control steps, pushes uniform in direction at up to 0.548 m/s.

| Controller | Fall-free | Mean return |
| --- | --- | --- |
| Zero action (pure PD) | **85.2 %** | +35.7 |
| Random action | 32.8 % | +26.9 |
| Trained policy | *target ≥ 95 %* | *target > +38* |

Without pushes the PD never falls in 400 steps. So the entire question of
Phase 1 is whether learning can close some of the **14.8 % headroom** the PD
leaves under perturbation.

### The honest ceiling

That headroom is not all winnable, and the plan should not pretend otherwise.

Pushes are sampled uniformly in direction. The zero-step capturable bound is
not: 0.548 m/s forward, 0.443 m/s backward, 0.620 m/s lateral. Roughly a
quarter of the direction circle is therefore near or past the bound at the top
of the curriculum.

### The capturable bound is not a hard ceiling — correction

An earlier version of this plan said a full-magnitude backward push "exceeds
what *any* controller can absorb without taking a step". That is wrong, and
measurement showed it: the capture-point controller holds at 0.55 m/s, above
the 0.548 m/s forward bound.

The reason is in the derivation. `ẋ_max = ω₀ d` comes from the linear inverted
pendulum, which assumes the **centroidal angular momentum does not change**.
The hip and arm strategy is precisely the violation of that assumption: swinging
the trunk and arms generates a horizontal ground reaction that the model does
not account for.

So the bound is the ceiling for an *ankle-only* strategy, not for the robot.
The honest statement is:

- `ω₀ d` bounds what the ankle alone can do.
- Angular momentum buys some margin past it, and the size of that margin is an
  empirical question this project can now answer.
- Stepping would buy much more, and is outside this task.

A realistic target is 95 %, not 100 %. A policy reporting 100 % on this protocol
should still be treated as suspicious — most likely it learned to crouch, which
lowers the centre of mass and evades the height reward's intent, or the
termination threshold stopped firing.

## Why a policy can beat the PD at all

Four concrete reasons, each of which maps to something measurable rather than to
a general claim that learning is better.

**1. The PD is linear around one operating point; the policy is not.** The gains
are sized from the inertia at the nominal crouch. Under a large push the
configuration moves far from that point, the effective inertia about each joint
changes, and the gains are no longer the ones the derivation chose. A policy
conditioned on the full state can apply a different correction in each regime.
*Measurable as:* fall-free rate at the top of the curriculum, where the linear
law is furthest from its design point.

**2. The PD cannot use the arms; the policy has eight arm joints.** When the
centre of pressure saturates at the edge of the foot, the only remaining way to
generate a restoring moment is to change centroidal angular momentum — in
practice, to swing the arms. The PD holds the arms at nominal by construction:
that is what a position loop does. The policy commands
`LShoulderPitch/Roll`, `LElbowYaw/Roll` and their right-side pair, so it can
express the strategy at all.
*Measurable as:* arm joint excursion correlated with push magnitude, and maximum
recoverable push with the arm joints frozen versus free. This is the single most
interesting experiment in Phase 1.

**3. The PD is tuned for one robot; the policy can be trained over a
distribution.** The URDF's inertials are approximate, the collision geometry is
convex hulls, and no actuator was measured on hardware. Domain randomization
turns that uncertainty into part of the training distribution.
*Measurable as:* fall-free rate under held-out friction and mass values.
**Blocked** until issue R1 below is fixed — randomization currently never
reaches the first step.

**4. The PD has no notion of capturability; the policy is rewarded on it.** The
PD reacts to joint error. It has no representation of
`ξ = x + ẋ/ω₀`, so it cannot trade posture error now against recoverability
later. The reward makes that trade explicit.
*Measurable as:* peak DCM excursion per push magnitude, compared against the
baseline's.

### What learning will *not* buy

- It cannot move the centre of pressure past the toe. The foot is 103.3 mm long
  and that is a geometric fact; in double support the foot binds before the
  ankle torque does.
- It cannot recover a push that needs a step, because stepping is outside the
  task. Note this is a *weaker* limit than it first appears: angular momentum
  extends the recoverable range past the ankle-only bound, as the capture-point
  baseline demonstrates, so "needs a step" is further out than `ω₀ d` suggests.
- It will not be smoother than the PD for free. The action-rate and torque
  penalties exist precisely because a 100 Hz policy can chatter at frequencies
  no real actuator would follow.

## Blocking work, in dependency order

These come from the static review in [nao_stand.md](nao_stand.md) plus runtime
testing. Each one is small; the order matters more than the size.

### R1 — Reset overwrites randomization *(blocks benefit 3 entirely)*

`_reset_idx` calls `super()._reset_idx(env_ids)`, which applies the `reset` mode
events, and then writes `default_joint_pos` and `default_root_state` over the
top. The randomized state never survives to the first step. Confirmed by
measurement: toggling randomization produced identical fall-free rates.

*Fix:* write the nominal state **before** `super()._reset_idx()`, so the events
perturb it, or drop the explicit writes and let the events own the reset state
entirely. Prefer the second — one owner per piece of state.

*Verify:* the two `check_nao_stand_env.py` modes must now differ.

### R2 — Stale derived values in the post-reset observation

`_compute_intermediate_values()` runs in `_get_dones()`, which the base class
calls **before** `_reset_idx()`. The observation is assembled after. So a
freshly reset environment reports the centre of mass, DCM and contacts of the
episode that just ended — a large, systematically wrong first observation on
every episode.

*Fix:* recompute at the top of `_get_observations()` as well, or recompute for
the reset subset at the end of `_reset_idx()`.

*Verify:* assert the first observation after a forced reset matches the nominal
posture within tolerance.

### R3 — Held joints never receive a target

Only the 19 actuated joints get `set_joint_position_target`. The head, wrists and
hands keep whatever the target buffer holds. Their nominal is zero so this is
probably harmless today, but it is unowned state and it silently stops being
harmless the moment a nominal changes.

*Fix:* write the held joints' nominal targets once at reset.

*Verify:* the zero-action trial must reproduce `scripts/stand_nao.py` numbers to
within noise. That equivalence is the whole justification for calling the zero
action "the baseline", and it is currently assumed rather than tested.

### R4 — Actions are unbounded

The Gaussian policy is not bounded by `action_scale`. A tail sample can command a
joint far past its limit, where the only thing stopping it is the simulator's own
clamp — at which point the gradient disappears.

*Fix:* set `clip_actions` in the runner config, and clamp the processed targets
into `soft_joint_pos_limits`.

### R5 — Mislabelled foot force

`_foot_normal_force` is a historical maximum of force *magnitude*, not the
current vertical component. It feeds the critic and reads as a normal load.

*Fix:* use the current vertical component, and rename whichever quantity it ends
up being.

### R6 — Observation buffer mutation

`_get_observations()` writes `_previous_actions`. It happens to be called once
per step, so the action-rate term is currently correct, but a second read would
silently zero it.

*Fix:* update the buffer in `_pre_physics_step` instead, where the action
actually changes.

### R7 — Mass weights after randomization

The centre-of-mass weights are cached from `default_mass` in `__init__`, while
the startup event changes the torso mass in PhysX. Every CoM and DCM value is
then slightly wrong, in a way that varies per environment and is invisible.

*Fix:* build the weights after startup events, or read masses live.

### R8 — Curriculum arithmetic

`push_curriculum_steps = 24_000_000` with 4096 environments and 24 steps per
iteration is ~244 iterations, not the 2000 the comment claims. As written the
curriculum finishes in the first 8 % of a 3000-iteration run, which is close to
having no curriculum at all.

*Fix:* pick the ramp in iterations and convert, and state the conversion.

### R9 — Direction-dependent capturability

The forward bound is used for every push direction, and the support polygon is a
fixed rectangle that does not track which feet are actually loaded.

*Fix:* scale the sampled push by the directional bound. Lower priority: it makes
the curriculum honest rather than fixing a defect.

### R10 — Training entry points

`scripts/train.py` and `scripts/play.py` are still placeholders, and the
extension is not pip-installed, so Isaac Lab's own scripts cannot resolve the
task. The play config still randomizes.

*Fix:* `pip install -e source/humanoid_soccer_lab`, then either thin wrappers or
documented direct use of Isaac Lab's `rsl_rl/train.py`.

## Then: the training campaign

1. **Sanity run.** 512 environments, 50 iterations. Confirm the return rises and
   nothing diverges. Check `Episode_Reward/*` in TensorBoard — if one term
   dominates the sum, the scales are wrong, not the policy.
2. **Save/load.** Checkpoint, reload, confirm the reloaded policy scores the same.
   rsl-rl 5.x changed the checkpoint layout; this is worth proving once.
3. **Full run.** 4096 environments, 3000 iterations, seed fixed.
4. **Three seeds.** A single seed proves nothing about a 10-point difference.

## Evaluation protocol

Run the trained policy through exactly the harness the baseline used, so the
comparison is like for like:

- **Headline:** fall-free rate on the 0.548 m/s protocol. PD = 85.2 %.
- **Push sweep:** maximum recoverable push per direction, in 0.05 m/s steps,
  against the capturable bounds. This is where the arm strategy should show up
  as exceeding the ankle-only prediction.
- **Ablation:** freeze the arm joints at nominal and re-run. The drop is the
  measured value of benefit 2, and it is the result worth writing up.
- **Robustness:** held-out friction and mass, once R1 lands.
- **Deployability:** action rate and torque against the URDF limits. A policy
  that wins by chattering has not won.

## Acceptance

Phase 1 is done when, on the same protocol:

1. The policy is fall-free on ≥ 95 % of episodes against the PD's 85.2 %.
2. The advantage holds across three seeds.
3. The arm ablation shows a measurable drop, establishing *why* it wins rather
   than just *that* it wins.
4. Peak torque and action rate stay inside the PD's envelope.
5. `scripts/check_nao_stand_env.py` passes with randomization both on and off.

Point 3 is the one that matters. A policy that beats the baseline for reasons
nobody can name is a result that does not transfer to the next task, and the
next task here is walking.
