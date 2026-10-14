# The shared skill space — separating a skill from the body that learned it

> Stage 2 of the transfer: describe what the G1 teacher *does* in terms that do
> not belong to the G1, and build the G1's side of a latent space that the NAO
> can later be attached to. The NAO side is the next stage and is not done here.

Every number on this page is read from a file a script in this repository
writes: `transfer/results/skill_space_teacher.json` (`encode_skill.py`),
`transfer/results/latent_g1*.json` (`train_latent.py`) and
`transfer/results/latent_g1_closed_loop*.json` (`evaluate_latent.py`).

## The problem

A gait recorded as joint angles describes the robot that walked it. The G1's 37
joint positions have no meaning on a NAO with 19 commanded joints and segments a
third as long. What *can* be shared is a description of what the body as a
whole is doing: where its centre of mass is and how it moves, how it is tilted
and turning, where its feet are, how its weight is shared between them, how
much it spins about its own centre.

The usual answer in cross-embodiment learning is a **learned** latent space: an
encoder and a decoder per robot around modules every robot shares (Devin et
al. 2017; Wang et al. 2024; more recently CrossBFM, PHASOR and H-Zero for
humanoids). The latent is aligned across robots by adversarial or
cycle-consistency losses on data from *every* robot, and nothing certifies that
it is free of the body it was learned on. Here that would mean data from a NAO
that walks — which does not exist, because teaching it to walk is the point.

## The idea

**Make the shared space physical where physics gives a body-independent
answer, and learned elsewhere.**

Take each body's own leg length `L`, mass `m` and gravity `g` as units
(Buckingham's π theorem). With `T = √(L/g)`, `V = √(gL)`, forces in `mg` and
angular momentum in `mLV`, the centroidal equations of motion become

```
dv*/dt* = Σ f_i* − ẑ          dk*/dt* = Σ (r_i* − c*) × f_i* + τ_i*
```

with no mass, no length and no joint count in them: **every humanoid obeys the
same dimensionless centroidal dynamics.** Two bodies differ only in which
states and contact wrenches they can reach — foot size, joint torque and speed,
mass distribution — and each of those, in body units, is a pure number. So:

| | is | lives in |
| --- | --- | --- |
| the **skill** | a trajectory of the 27-number skill state `z*` | `common/skill_state.py` |
| the **body** | a vector of dimensionless groups `Π` | `transfer/embodiment.py` |
| the **gap** | `δ = ln Π_student − ln Π_teacher`, one named number per group | `transfer/embodiment.py` |

The learned latent space is then **anchored** to `z*`: part of it, `ẑ*`, is
trained to equal the analytic skill state. Because `z*` means the same thing on
any humanoid, a second body's encoder can be pinned to the same coordinates
from *its own* simulation — no paired data, no adversary, no teacher data.

### What is new, as far as the search went

Each ingredient is classical: dimensional analysis and the Froude number in
locomotion (Alexander and Jayes 1983), centroidal dynamics (Orin, Goswami and
Lee 2013), the divergent component of motion (Englsberger et al. 2015),
morphological symmetry (Ordoñez-Apraez et al. 2023, 2025). Buckingham-π
transfer of *policies* exists (Girard 2024; Pascoa, Lalonde and Girard 2025;
Charvet et al. 2025) but for one morphology rescaled — pendulum, cart-pole,
HalfCheetah — and assuming similarity holds. Learned cross-humanoid latents
(Yan and Lee 2026; CrossBFM 2026; PHASOR 2026; H-Zero 2025) carry no physical
normalisation. The search did not find the combination used here: a learned
latent for humanoids of different joint topology, anchored to a dimensionless
centroidal state whose invariances are tested rather than encouraged, with the
dissimilarity between the bodies measured group by group instead of assumed
away. The searches are listed in the plan's prior-work threat; the claim is
"not found", not "does not exist".

## 1. The skill state `z*`

| field | size | frame | unit |
| --- | --- | --- | --- |
| `height` | 1 | — | `L` |
| `com_velocity` | 3 | heading | `V` (its square is the Froude number) |
| `gravity` | 3 | base | — |
| `base_angular_velocity` | 3 | base | `1/T` |
| `foot_position` (L, R) | 2×3 | heading, from the CoM | `L` |
| `foot_velocity` (L, R) | 2×3 | heading | `V` |
| `foot_load` (L, R) | 2 | — | `mg` |
| `angular_momentum` | 3 | heading | `mLV` |

The heading frame is the world turned by the yaw of the body carrying both hips
(G1 pelvis, NAO `base_link`); positions are taken from the CoM. Contact is the
**load share** `f_z/(mg)`, not a 0/1 flag: a threshold in newtons is a different
criterion on each body (Isaac Lab's 1 N is 0.32 % of the G1's weight and 1.9 %
of the NAO's).

`L` is the hip-pitch-to-sole length with the leg straight — a property of the
body. How much a task crouches stays in the state as `c_z/L`.

**Tested as group actions** (`tests/test_skill_state.py`):

- dynamically similar motions of bodies of any size encode identically, and
  copying speeds unscaled — the "copy the teacher's speed" error — does not;
- world translation and yaw drop out;
- the mirror image of a motion encodes to `M z*`, with `M` generated from each
  field's declared kind: polar vectors flip `y`, axial vectors (angular
  velocity, angular momentum) flip `x` and `z`, left and right swap. Breaking
  either axial rule fails the test;
- a NAO built from its URDF by forward kinematics encodes to its known CoM
  height and a 50/50 load.

## 2. The teacher's gait, encoded and certified

`g1/scripts/capture_rollouts.py` records the pinned teacher (sha256 checked and
stored) and computes `z*` online in the G1's units. Checked against quantities
the encoder never used (`skill_space_teacher.json`, 32 environments × 600 steps,
play task):

| check | result |
| --- | --- |
| feet carry the body's weight on average | 1.0084 `mg` |
| Newton, vertical, per sample: `load_L + load_R − 1 = c̈_z/g` | residual 1.4·10⁻⁷ g RMS |
| share of `\|k_G\|` from the spin term | median 4.4 %, 90th percentile 6.8 % |
| raw per-body state re-encoded offline against the online state | 6·10⁻⁸ |

The Newton check compares two independent simulator outputs — the contact
sensor and the link accelerations — so it pins the mass, the sensor indexing and
the CoM aggregation together. It is also what the previous capture would have
failed: it indexed the contact sensor with the articulation's body order and
recorded zero foot force in all 25 600 samples, and it omitted the spin term its
own docstring listed.

**The gait** (`gait.signature`): Froude number 0.0073, stride 0.156 `L` every
1.82 `T`, duty factor 0.92 / 0.85 (between 1 % and 10 % of body weight as the
contact threshold it moves by less than 2 points), left foot carrying 0.534 of
the weight against 0.474. Alexander's animal relation would give 0.525 `L` at
this Froude number — outside the range it was fitted on, so a reference rather
than a target. The teacher shuffles: short strides with both feet down most of
the time.

**Froude matching, redone with one length.** The teacher's command was sized
with 0.74 m (the G1's pelvis spawn height) against 0.2689 m (the NAO's crouched
CoM height) — two different lengths, pinned by tests and left alone. With the
hip-to-sole length on both bodies the teacher's command corresponds to
0.1532 m/s on the NAO, against the 0.15 m/s assumed. The two errors nearly
cancel; the number is now derived rather than lucky.

## 3. The two bodies as dimensionless groups

From `skill_space_teacher.json`, in each body's own units; the last column is
`Π_NAO / Π_G1` (`= e^δ`):

| group | G1 | NAO | NAO / G1 |
| --- | ---: | ---: | ---: |
| CoM height / `L` | 1.014 | 1.114 | 1.10 |
| inertia, roll / pitch / yaw (`mL²`) | 0.233 / 0.212 / 0.038 | 0.447 / 0.391 / 0.074 | 1.92 / 1.84 / 1.95 |
| foot length / `L` | 0.310 | 0.661 | 2.13 |
| foot width / `L` | 0.100 | 0.372 | 3.72 |
| sole separation / `L` | 0.360 | 0.403 | 1.12 |
| torque, hip pitch / knee (`mgL`) | 1.449 | 0.234 | 0.16 |
| torque, ankle pitch (`mgL`) | 0.097 | 0.234 | 2.42 |
| speed, hip pitch (`1/T`) | 8.27 | 1.02 | 0.12 |
| speed, ankle pitch (`1/T`) | 13.69 | 1.02 | 0.07 |

So "the weak robot" is weaker in some directions and *stronger* in others: in
body units the NAO has a sixth of the G1's hip and knee torque but 2.4× its
ankle torque — the simulated G1's ankle group is capped at 20 N·m against
300 N·m elsewhere — feet twice as long and almost four times as wide, and joint
speeds an order of magnitude lower. A single strength ratio would hide all of
that. A scaled copy of a body has zero defect (tested); matching every group is
necessary for similarity and not sufficient.

## 4. The learned shared space

```
o_r ──E_r──► (ẑ*, ζ) ──┐
                       ├─► π(ẑ*, ζ, c*) ──► u ──D_r(u, κ_r)──► a_r
c* (dimensionless cmd) ┘                     │
                             T(ẑ*, ζ, u) ──► (ẑ*', ζ')
```

- `E_r`, `D_r` per body; `π` (the skill) and `T` (latent dynamics) shared.
- `ẑ*` anchored to `z*`; `ζ` a small free part behind a KL bottleneck.
- `c*` is the command in body units, so Froude matching is done by the units.
- The decoder's context `κ_r` is the body's joint positions and velocities —
  nothing about the floating base, so balance and intent can only arrive
  through `u`; and not the last action, because consecutive actions are nearly
  equal and a decoder that sees the previous one imitates by copying it (the
  copycat shortcut, Wen et al. 2020).
- `T` predicts the next physical state from `u`, which gives `u` a meaning a
  new body's decoder can be held to: *the `u` that produces this change of the
  centroidal state*.

Distilled offline from the pinned teacher on DART-noised rollouts across its
command range (`latent_g1.json`; 111 552 training transitions, held-out
environments for validation). Three variants, same seed and capacity:

| held-out, walking | anchored | anchored, open `ζ` | free (no anchor) |
| --- | ---: | ---: | ---: |
| imitation R² | 0.969 | 0.971 | 0.951 |
| imitation R² with `u` frozen (bypass) | 0.277 | 0.289 | 0.413 |
| `z*` decoded linearly from `π`'s input, R² | 0.947 | 0.944 | 0.422 |
| G1 joint configuration decoded from `π`'s input, R² | 0.512 | 0.888 | 0.445 |
| … from the true `z*` alone, R² | 0.434 | 0.434 | 0.434 |

Three findings:

1. **The skill flows through the shared space.** Freezing `u` drops imitation
   from 0.97 to 0.28.
2. **The anchor costs nothing and gives the space a meaning.** The anchored
   model imitates at least as well as the free one, and its skill policy's
   input is the physical state (0.95) rather than an arbitrary code (0.42).
3. **The free part is where a body leaks in.** With a weak bottleneck
   (β = 10⁻³) `ζ` carried about 7 nats that, combined with `ẑ*`, decoded the
   G1's joint configuration at R² 0.89 — against 0.43 for the physical state
   alone — for no gain in imitation. At β = 10⁻² it closes, and the skill
   policy runs on the physical state alone. Such a channel is exactly what a
   NAO encoder could not reproduce.

The anchor is accurate while walking (median R² 0.95 held out, 0.98 on the
clean gait) and poor in the second before a fall, of which the training capture
has two; the report keeps the two apart.

## 5. Closed loop

Offline imitation says nothing about what happens once a policy chooses its
own states. Each policy was run in the G1's play task (the teacher's command,
0.2488 m/s; 32 environments × 900 steps) through
`capture_rollouts.py --driver-policy`, the teacher labelling every step.
Criteria were fixed before the runs: speed ≥ 90 % of the teacher's, fall rate
no more than 2 points above it, stride length, stride period and duty factor
within 10 %. DAgger (Ross et al. 2011) ran three rounds, each a 128 × 1000
capture with the latest anchored student driving across the teacher's command
range, aggregated with everything before it.

From `latent_g1_closed_loop.json` (the DART-only stage is also kept in
`latent_g1_closed_loop_dart.json`):

| run | falls | speed (m/s) | stride (`L`) | period (`T`) | duty L / R | rhythm (`T`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 0 / 32 | 0.213 | 0.159 | 1.87 | 0.92 / 0.86 | 0.97 |
| anchored, DART only | 0 / 32 | 0.217 | 0.131 | 1.50 | 0.90 / 0.82 | 1.47 |
| anchored, DAgger 1 | 0 / 32 | 0.210 | 0.139 | 1.65 | 0.91 / 0.82 | 1.35 |
| anchored, DAgger 2 | 0 / 32 | 0.202 | 0.182 | 2.24 | 0.92 / 0.87 | 1.02 |
| anchored, DAgger 3 | 0 / 32 | 0.213 | 0.186 | 2.16 | 0.93 / 0.85 | 0.85 |
| free (no anchor), DART only | 0 / 32 | 0.191 | 0.115 | 1.50 | 0.91 / 0.83 | 0.75 |
| anchored, open `ζ`, DART only | 62 / 94 | 0.323 | 0.237 | 1.75 | 0.90 / 0.75 | 0.80 |
| anchored, `u` frozen (bypass) | 288 / 320 | — | 0.929 | 6.04 | 0.90 / 0.98 | 7.74 |

What it says:

1. **The skill is in the shared space, closed-loop too.** With `u` frozen the
   G1 falls nine times per environment.
2. **A leaky free part costs stability on the very body it leaks from**: 62
   falls with the open bottleneck, none with it closed.
3. **The anchored policy walks.** No falls in any round, at 0.95–1.02 of the
   teacher's speed, duty factor within four points. The unanchored one also
   walks, at 0.90 of the speed.
4. **Stride timing is not reproduced to 10 %.** Trained on DART alone the
   student takes strides about 18 % shorter and quicker; DAgger turns that
   into strides 14–16 % longer. **No run meets the pre-registered stride
   criterion, so none passes, and the skill has not been pinned** — the plan
   makes pinning conditional on every criterion.

The instrument is part of the story. The teacher's own touchdown-based period
moves from 1.97 to 1.69 `T` as the contact threshold goes from 1 % to 20 % of
body weight: with a duty factor of 0.92 its swings last about two control
steps, and a missed lift-off merges two strides. The threshold-free rhythm of
the left-right weight shift (`common.gait.dominant_rhythm`, added after seeing
this and therefore reported as supplementary, not used for the verdict) puts
the teacher's peak at 0.97 `T` holding only 9 % of the power: a broadband
shuffle rather than a clock. On that measure DAgger 2 is within 5 % and
DAgger 3 within 12 %.

Whether to pin DAgger 3 as the frozen skill for the NAO stage — accepting the
timing gap as measured — or to work on timing first is an open decision, not
something this stage settles.

## 6. A memory without a back door

The timing gap was attacked first. `z*` is instantaneous, and with a duty
factor of 0.92 the present says little about where in the stride the body is,
so the skill was given a memory. A memory is also the easiest way for a body
to leak back into the shared space: a history of observations carries the
G1's own joint trajectories, which no NAO encoder could reproduce. So the
leak is ruled out by construction, not discouraged by a penalty
(`transfer/latent/memory.py`):

1. **The memory `Φ` is shared and reads only the history of `ẑ*`**, on a grid
   in body time (`Δ* = 0.077 T`, 26 samples ≈ 2 `T`), never an observation.
   `φ` is then a function of the physical trajectory, so it cannot carry
   anything about the body that the trajectory does not. Each body resamples
   its own history onto the grid: at 50 Hz the NAO steps 0.126 `T`, and a
   window counted in control steps would mean 1.6 times more physical time.
2. **The encoder learns only the anchor.** No imitation gradient reaches it,
   so it has no reason to hide facts about the G1 in its errors — the route
   the open `ζ` of §4 took.
3. **The shared modules and the decoder train on the true `z*` plus noise**
   of the encoder's measured held-out error, so `π` never learns to read the
   G1 encoder's residuals. `T` is rolled out 8 steps.

The run-time policy is a scripted module with a per-environment buffer that
`capture_rollouts.py` resets at episode ends. The per-limb decoder of
`transfer/latent/limbs.py` was trained as a second variant: `u` split into a
part every head reads and one share per limb (trunk, legs, arms; still 16
wide), each head seeing only its own joints.

**The gate.** Before training, probes on held-out DART environments checked
whether the restriction costs anything (`phase_probe.json`, rule fixed in
advance: physical history ≥ observation history − 0.05). It passes:

| MLP probe R² | `z*` now | `z*` history | observation history |
| --- | ---: | ---: | ---: |
| `z*` 8 steps ahead | 0.521 | 0.518 | 0.465 |
| steps to each foot's next landing | 0.590 | 0.541 | 0.312 |

It also says something the rule did not ask: **the history predicts no better
than the present.** The teacher's next landing is about as foreseeable from
now as from the last two strides — consistent with the broadband shuffle of
§5.

**Offline** (`latent_g1_memory.json`; DART plus the three DAgger captures,
held-out walking):

| | memory | memory, per limb |
| --- | ---: | ---: |
| imitation R², deployed path | 0.921 | 0.869 |
| … with `u` frozen | 0.626 | 0.270 |
| … with `φ` frozen | 0.833 | 0.703 |
| joints from `π`'s input, MLP R² | 0.588 | 0.603 |
| joints from the true `z*` history, MLP R² | 0.665 | 0.665 |

The memory is used, and **nothing leaked past the physical history**: the
skill policy's input knows less about the G1's joints than the true physical
trajectory does. The single decoder still imitates at 0.63 with `u` frozen —
it leans on the joints it sees — where the per-limb decoder drops to 0.27.

**Closed loop** (`latent_g1_closed_loop_memory.json`; same harness and
criteria as §5, DAgger rounds driven by each variant's own policy):

| run | falls | speed (m/s) | stride (`L`) | period (`T`) | duty L / R | rhythm (`T`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 0 / 32 | 0.213 | 0.159 | 1.87 | 0.92 / 0.86 | 0.97 |
| anchored, DAgger 3 (§5) | 0 / 32 | 0.213 | 0.186 | 2.16 | 0.93 / 0.85 | 0.85 |
| memory | 46 / 78 | 0.327 | 0.194 | 1.44 | 0.87 / 0.77 | 1.41 |
| memory, DAgger 1 | 0 / 32 | 0.223 | 0.113 | 1.26 | 0.90 / 0.81 | 1.05 |
| memory, DAgger 2 | 4 / 36 | 0.206 | 0.116 | 1.39 | 0.90 / 0.83 | 1.38 |
| per limb | 33 / 65 | 0.538 | 0.270 | 1.35 | 0.86 / 0.76 | 0.92 |
| per limb, DAgger 1 | 1 / 33 | 0.241 | 0.117 | 1.21 | 0.90 / 0.79 | 1.07 |
| per limb, DAgger 2 | 16 / 48 | 0.271 | 0.148 | 1.34 | 0.90 / 0.76 | 1.44 |
| memory, DAgger 1, `φ` frozen | 130 / 162 | — | — | — | — | — |
| per limb, DAgger 1, `φ` frozen | 125 / 157 | — | — | — | — | — |
| memory, DAgger 1, `u` frozen | 256 / 288 | — | — | — | — | — |
| per limb, DAgger 1, `u` frozen | 288 / 320 | — | — | — | — | — |

What it says:

1. **Training on noise is not the same as running on an encoder.** Before
   DAgger both variants run too fast and fall: off the teacher's
   distribution the encoder under-reads forward speed, the policy
   accelerates, and the encoder falls further behind — a loop the
   independent noise of training never showed it. One DAgger round removes
   it; a second does not help and lets falls back in.
2. **The memory carries balance, not only timing.** With `φ` frozen the
   round-1 models fall in 130 and 125 episodes; with `u` frozen they collapse.
3. **The timing gap changed sign, it did not close.** The memoryless model
   took strides 16 % too long; the memory model takes them about 29 % too
   short, at every contact threshold. The threshold-free rhythm of DAgger 1 is
   within 10 % of the teacher's (1.05 and 1.07 against 0.97), but it is the
   supplementary measure, and on the pre-registered criteria **no variant
   passes**.

Together with the gate, this points at the teacher rather than the student:
a gait whose next landing the past does not predict gives a memory little to
learn, and the stride length the student settles on moves with whatever
else changes. Whether to pin a policy with the gap as measured, or to replace
the stride criterion with the threshold-free rhythm (a change to criteria
fixed in advance, and to be justified as such), remains the open decision of
§5.

## 7. A contract instead of an action code

Two further attempts on the timing gap followed §6 and both failed the gate
fixed for them, in a way that pointed past timing:

- **A contact schedule** (`latent/events.py`, `plan.py`; `latent_g1_plan*.json`):
  a shared planner for when each contact next changes (a censored
  discrete-time hazard) and where the foot lands, with the policy committed
  to a sampled plan. The planner was calibrated (0.020 / 0.005) and
  informative (+0.45 / +0.74 nats over the population), and the policy did
  not use it: imitation R² moved by +0.004 and +0.011 when the plan was
  frozen, against the 0.02 required.
- **Holding the policy to its plan** through the latent dynamics did not
  change that (+0.000, +0.002).

The reason is general. A deterministic teacher acts `a = π_T(o)`: once `o` is
known, nothing computed from the future adds anything about `a`. Imitating
actions can therefore never make a student use a plan; the plan can only
explain what the student's input misses of `o` (at R² 0.915, 8.5 %). And a
latent action `u` means whatever the decoders learned from walking, so a body
decoder trained on it learns to walk, not to carry out `u` — a second task
would need new decoders on every body.

So the interface between a skill and a body was fixed by physics instead of
learned (`transfer/latent/contract.py`): **the next values of `z*` on the
body-time grid.**

```
obs ─E_b─► ẑ* ─► G_task(history of ẑ*, c*) ─► plan of z* ─► D_b(obs, ẑ*, plan) ─► joints
                 └ task side: no body ┘        (contract)     └ body side: no task ┘
```

- **The task side** is a generator per task: the last 2 `T` of `ẑ*` and the
  dimensionless command in, a *sampled* 1 `T` of `z*` out. It is trained
  offline on the teacher's states — never its actions — with noise of the
  encoder's measured error on the history.
- **The body side** is an encoder `E_b` (anchor only, as in §6) and a tracker
  `D_b` trained by RL to reach the plans (`transfer/tasks/tracking`,
  `G1Track-v0`, `NaoTrack-v0`). It never sees the command: the plan is the
  only thing that says what to do.
- **A plan is held** for 6 samples (0.46 `T`) and redrawn from the history
  that actually happened. With a calibrated generator and exact tracking,
  that is ancestral sampling of the teacher's trajectory distribution, so
  every temporal statistic of it carries over without being named.
- **A new task** is a new generator on the same body's encoder and units
  (`train_contract.py --encoder-from`); a new body is a new tracker.

### 7.1 The generator, iterated

| generator | calibration (held-out) | timing gain (nats) | note |
| --- | ---: | ---: | --- |
| conditional VAE | 0.093 / 0.085 | 0.48 / 0.63 | averaged futures: changes under-forecast |
| flow matching, 16-wide history | 0.060 / 0.009 | 0.38 / 0.64 | left stance lift-off early |
| flow, 64-wide history, 1024 flow | 0.012 / 0.027 | 0.43 / 0.66 | gate passes |
| … sampled by midpoint, 8 steps | 0.005 / 0.017 | 0.42 / 0.65 | the converged sampler, half the cost |
| … loads modelled as `log(load + 0.01)` | 0.010 / 0.006 | 0.48 / 0.73 | gate passes; current |

(`contract_walk.json`, `contract_walk_big.json`, `contract_walk_warp.json`.)
Two findings shaped it beyond the gate:

1. **A mean future is not a future.** The VAE decoded the mean given its
   intent, and an averaged lift-off is a ramp that never crosses the
   threshold. Flow matching samples whole trajectories.
2. **Forces are not negative.** In plain units the flow put mass on both
   sides of zero load: plans asked for −0.48 body weights and their swings
   collapsed to a single step (median 1 against 2 in the data). Modelling the
   load share in log units fixes the boundary for any contact of any task.

**Chained on its own plans** — what a perfect tracker would walk
(`transfer/scripts/chain_contract.py`, `contract_chain_walk_*.json`) — the
generator still steps fast: stance 15 against 23 steps, stride period −22 %,
rhythm −16 %, stable in every environment. Its single plans are calibrated
by stance age too (lift-off within 13 steps at ages 10–15: 0.41 forecast,
0.50 data), so the shortening comes from reading its own plans back as
history. This was first reported as a pessimistic bound, on the grounds
that a closed loop's history is the physical state a body reached. That
was wrong: a body that tracks perfectly reaches exactly its plans, so its
physical history *is* the chain. The chain is the perfect-tracking limit
of the deployed loop, and its bias has to be removed in the generator
(§7.2, round 10).

**What did not remove it, measured:**
- *History noise and commitment:* the chained period is the same with
  history noise 0, 1 or 2 σ (1.46 / 1.47 / 1.45 `T`) and with a 6- or
  13-step commitment (1.47 / 1.40).
- *Three DAgger rounds:* chain period 1.471, 1.469, 1.482 against the
  teacher's 1.852. Round 3 drew on a loop that already walked at the
  teacher's speed, with no teacher falls.
- *Single plans are calibrated:* not only in their first contact change
  but in all of them. Contact changes per 13-step plan per foot:

  | | left | right | first 6 steps (left / right) |
  | --- | ---: | ---: | --- |
  | generator | 1.16 | 1.27 | 0.55 / 0.58 |
  | teacher | 1.13 | 1.30 | 0.52 / 0.60 |

The acceleration lives in the recursion. The generator answers histories
made of its own output differently from the teacher's histories, and no
body's history reproduces its output well enough for DAgger to reach it.
A sequence-level objective (training through chained plans, or a
discriminator on chained sequences) is the general next step, and is
left open.

### 7.2 The body side, iterated

**Round 2** (`contract_g1_closed_loop_r2.json`): fields scored apart, scores
averaged, kernels at 1, 10 and 100 tolerances. Trained without falls, and
then, closed-loop against the teacher:

| path | falls | speed (m/s) | own / other plan error | verdict |
| --- | ---: | ---: | ---: | --- |
| contract on the true state | 0 / 64 | 0.005 | 0.35 / 2.17 (use passes) | fail |
| deployed, through the encoder | 291 / 323 | −0.34 | 0.59 ratio | fail |

On the true state it follows its plans and stands still. From its own
standing history the generator does ask it to start (0.056 → 0.115 m/s
within 1 `T`), but each plan restarts from where the body is, so not
accelerating never accumulates, and an averaged field score lets it give up
the one field it misses: standing earned 0.87 of the reward. Two general
corrections followed:

- **One likelihood for the whole state** — every field at once.
- **The plan's integrals** since it was drawn, CoM displacement and turn,
  against their spread across plans: the root-trajectory term of every
  motion tracker, which a state (as opposed to a trajectory) leaves out.

On the deployed path the encoder, fitted to the teacher's actions, read the
tracker's out of distribution — the same loop §6 found before DAgger.

**Round 3** (both corrections, `walk_warp` contract) stood too: checkpoint
300 on the true state walked at 0.02 m/s with both feet down 99 % of the
time. Probing the generator explained the rest:

- **The tolerance was the wrong quantity.** It was the generator's spread of
  plans — how uncertain the future was *before* a plan was drawn. Where the
  future is uncertain, as at a start, not following the drawn plan cost
  almost nothing. A drawn plan is a commitment, so the reached state is now
  scored at the precision the state is known with: the encoder's held-out
  error per field, and that error on the velocities carried through time
  (growing as `√t`) for the integrals. The pinned teacher, which follows none
  of the plans it is scored against, now sits 5–13 tolerances away; a body
  following its plan should sit near one.
- **The data had no starts.** The teacher walks off on the step after every
  reset, so from a standing history the generator jumped to 0.058 m/s on the
  next step — a request no body meets. `teacher_starts.npz` captures the
  teacher with 30 % of environments standing at each resample and commands
  every 2–6 s (36 clean starts, 0 → 0.06 m/s within 13 steps); the
  generator trained on it with DART passes its gate at 0.004 / 0.004
  (`contract_walk_starts.json`), and the tracker trains on starts and stops
  too.

**Rounds 4 to 6** all ended standing, each for a reason a measurement
pointed at, and each correction general:

| round | change | closed loop (true state) | own / other plan error |
| --- | --- | --- | --- |
| 4 | precision tolerance, start-aware contract | 0.000 m/s, both feet down 99.5 % | 1.19 / 15.8 |
| 5 | + reference state initialisation | 0.000 m/s, even started mid-walk | 1.08 / 25.6 |
| 6 | + DAgger on the contract (round 5 resumed) | 0.000 m/s | 1.39 / 7.0 |

(`contract_g1_closed_loop_r4_500.json`, `_r5_500.json`, `_r6_700.json`.)
Every tracker follows its plans — the use gate passes by a wide margin —
and the plans themselves stop asking for motion:

- From standing the teacher **leans before it steps**: its plans ramp the
  CoM to 0.08 m/s over 1 `T` with a foot lifting in 1 % of them. Over the
  6 samples a plan was held, that is 0.04 m/s and a few millimetres — a
  cost of about 0.2 in the joint error for not moving at all.
- **Each plan restarts from where the body is.** A body that does not move
  is handed the start of the ramp again, so the veto never accumulates.
  Started mid-walk, the tracker does less than asked for one plan, and from
  then on it is a standing body being asked to start.
- **DAgger** (`teacher_from_states.py`: the teacher run from 1024 states
  the round-5 loop visited) removed the generator's artefacts — it had
  planned an impossible 0.058 m/s jump on the first step from standing; it
  now plans the teacher's real start — and the loop still stands, because
  the real start is the slow one.

Re-anchoring on the physical state is what keeps the generator in its own
distribution (no exposure bias, whatever the chain test says); the veto is
the price, and its size is set by how long a plan is held. Round 7 holds
each plan for 1 `T` (13 samples, horizon 20) instead of 0.46 `T`: not
following an acceleration then costs quadratically more by the plan's end,
and nothing is lost in fidelity, since every plan still starts from a
physical history.

**Round 7** stood as well (`_r7_900.json`): −0.003 m/s, both feet down
100 % of the time, own/other plan error 1.97 / 5.82. A longer commitment
only changes the price of the veto, not who holds it. As long as a plan
restarts from the body's state, the body chooses what it is asked to do,
and a precision reward makes the easiest regime the best one.

**The structural correction: a reference the body cannot steer.** Motion
trackers avoid this by construction. In DeepMimic (Peng et al., 2018) the
reference clip plays on whatever the character does, and only a fall ends
it. The contract can do the same.
- **Autonomous history** (`ReferenceCommand.autonomous`): during training
  the generator reads the plan it committed to (plus its noise), not the
  body. The task's trajectory is the generator's own chain, conditioned on
  the command, and the body has to follow it.
- **Round 8** kept a way back: a resync to the body's history after 6 steps
  more than 25 tolerances away. It stood again (`_r8_500.json`: 0.023 m/s,
  error 2.08 / 16.97), because the resync handed back the veto: drift far
  enough and the task restarts from you.
- **Round 9** removes the resync (`resync_error = inf`), resumed from round
  8 at 500. It is the first tracker that walks (`_r9_700.json`, from
  standing at the play command): 0.13 m/s against the teacher's 0.21, use
  ratio 0.19. The stride is still −27 % and the period +16 %, with both feet
  down 97 % of the time, a shuffle. At 700 iterations it was still
  improving (0.03 → 0.13 m/s in 200).

**Training and deployment read different histories, on purpose.** The
autonomous reference serves training: it removes the incentive to steer.
A deployed body draws its plans from its own history, the ancestral
sampling the theory covers, which also keeps the task's feedback (a pushed
body gets a plan from where it is). The tracker it uses is the same, since
it learned to reach plans and not to choose them. `close_loop_contract.sh`
captures both. The capture also records each plan's due state, so the
evaluator can split a miss into the generator's share (the motion asked
for against the teacher's) and the tracker's (the body against what was
asked).

**Round 9 at 1000 iterations, split this way** (`_r9b_1000.json`, play
command, teacher 0.213 m/s, stride 0.159, period 1.87, duty 0.92 / 0.86):

| path | falls | body speed | asked speed | asked period | median error by field |
| --- | ---: | ---: | ---: | ---: | --- |
| deployed (encoder, body's history) | 50 | 0.239 | 0.179 | 1.30 | foot position 14, progress 17 |
| true state, body's history | 0 | 0.114 | 0.150 | 1.73 | 0.4–2.6 in every field |
| true state, autonomous | 0 | 0.161 | 0.219 | 1.45 | foot position 8, progress 6 |

Four findings:
- **The tracker follows plans to within the precision they are scored at.**
  On the true state every field sits 0.4–2.6 tolerances from its plan, and
  the use ratio is 0.067.
- **It delivers about three quarters of the speed it is asked for, on
  either reference** (0.114 of 0.150, 0.161 of 0.219). On the body's own
  history the shortfall compounds: each plan starts from a slower state,
  and the loop settles at 0.114 m/s.
- **The body's history keeps the generator near the teacher's timing**
  (asked period 1.73 against the teacher's 1.87). Its own chain does not
  (1.45, the −21 % of §7.1).
- **The encoder misreads the tracker's states**, so the plans it seeds are
  wrong (period 1.30) and the body falls. The encoder belongs to the body
  and has to be refitted on the body's own states.

A shortfall that is the same in every plan is a *bias*, and the reward
could not see it. The plan's integrals were scored only over the steps
each plan was held (at most 6). Over k steps an unbiased body drifts from
the reference like a random walk, about σ√k, and a biased one drifts like
k. At k ≤ 6 the two are barely apart, which is why the shortfall cost
about 1 in the joint error.

**The correction** carries the reference's displacement across plans,
which is how a reference clip's root trajectory works in motion tracking
(Peng et al., 2018):
- a plan drawn from the task's own trajectory starts where that
  trajectory has got to, not where the body is;
- the integrals' tolerance grows as σ√t, counted from when the reference
  last started from the body.

Over an episode a bias then costs in proportion to its duration, while an
unbiased drift costs nothing extra. With the body's history, where every
plan starts from the body, the term reduces exactly to the per-plan one.

**Round 10** (that term, resumed from round 9 at 1000, on `walk_dagger2e`)
made things worse. Over 200 iterations the body's lag behind the
reference grew from 0.96 to 1.59 leg lengths, and the rewards fell. The
reward now depended on an accumulated lag that neither actor nor critic
could see: `z*` says how a body should move, not where it should be. The
problem the tracker learned on was no longer Markov in what it observed.

**Round 11 lets it see the lag.** Every motion tracker observes the
target root position relative to its own (Peng et al., 2018), and the
contract's view now does too.
- **What it observes:** for each step of the view, where the reference's
  CoM will be in the body's heading frame, and its heading relative to
  the body's.
- **Deployed:** the reference starts at the body, so this is the plan's
  own path ahead, which a real body integrates from its velocity estimate
  over one plan.
- **Warm start:** the new inputs go last in each group. `widen_inputs.py`
  widens the trained network with zero weights on them, so it starts out
  computing exactly what it did and learns to use them.
- **Effect:** over 300 iterations the lag fell from 1.8 to 0.24 leg
  lengths.

**Round 11 at 1599** (`_r11_1599.json`), against the teacher at its play
command:

| path | falls | speed (ratio) | stride | period | duty | use |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| deployed (encoder) | 0 / 64 | 0.262 (1.23) | +30 % | +6 % | 0.98 / 0.90 | 0.111 |
| true state, own history | 0 / 64 | 0.233 (1.10) | +21 % | +10 % | 0.98 / 0.89 | 0.088 |
| true state, autonomous | 0 / 64 | 0.222 (1.04) | +20 % | +15 % | 0.98 / 0.90 | 0.075 |

- **What is fixed:** the deployed path no longer falls (50 before the
  encoder was refitted), and the speed bias is gone. The use gate passes
  on every path, and period and duty are within 6–15 %.
- **What fails gate C:** the stride.
- **Split by source:** the plans ask for shorter, quicker steps than the
  teacher (period 1.37–1.45, the chain's bias of §7.1). The body takes
  fewer, longer steps at the right speed: it reaches the CoM's motion
  (0.55–0.78 tolerances) better than the feet's (2.2–4.2).
- **The feet, per side:** the body drags its left foot. When the plan
  asks for a left swing, the left foot leaves the ground 3 % of the time;
  for the right foot it is 54–63 %. Share of time each foot is in swing:

  | | left | right |
  | --- | ---: | ---: |
  | teacher | 0.08 | 0.15 |
  | plans | 0.10 | 0.15 |
  | body | 0.02 | 0.11 |

  This is the familiar local optimum of legged RL, here without the
  air-time term that usually keeps a robot out of it. The contract asks
  for the lift, with a tolerance on foot height below a millimetre, and
  the tracker settled for the slide.

**Round 12 passes the gates on the deployed path**
(`_r12_2198.json`). It is round 11's tracker after 600 more iterations
with early termination (§7.4), on `walk_dagger3e`: DAgger round 3, and
the encoder refitted on round 11's states.

| path | falls | speed ratio | stride | period | duty | use | gate C |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| deployed (encoder) | 1 / 64 | 1.00 | +6.4 % | +5.1 % | +5.4 / +4.5 % | 0.149 | **pass** |
| true state, own history | 0 / 64 | 0.98 | +22 % | +23 % | +6.5 / +5.2 % | 0.114 | fail |
| true state, autonomous | 1 / 64 | 1.05 | +24 % | +18 % | +6.7 / +4.2 % | 0.053 | fail |

The plan pre-registered two gates on the deployed path:
- **Gate B (use)** passes on every path.
- **Gate C (behaviour)** passes on the deployed path. The fall rate is
  1.6 points above the teacher's, against an allowance of 2.

The pass is fragile. Fed the true state, the same tracker takes strides
a fifth longer and slower. The encoder's error moves the plans (asked
period 1.38 `T`, against 1.48 on the true state), and with them the gait,
by about 15 %. Here that moves it towards the teacher's. A robust pass
needs the two paths to agree. The remaining gap on the true state is the
one §7.1 leaves open: the generator's chain steps fast, while the body
takes longer strides.

### 7.3 A new task on the same body, zero-shot

The published `Isaac-Velocity-Flat-G1-v0` policy is the new task. Its
contract is `flat_h20e`: its generator, with the G1's refitted encoder
moved over by `body_encoder.py`. It drives round 11's tracker, unchanged
(`contract_new_task_r11_*.json`):

| command | path | falls (episodes) | speed ratio | use |
| --- | --- | ---: | ---: | ---: |
| forward 0.25 m/s | true state | 120 / 154 | 0.83 | 0.31 |
| fast 0.6 m/s | true state | 141 / 178 | 0.37 | 0.33 |
| turning | true state | 144 / 179 | 0.93 | 0.37 |
| all three | encoder | 191–238 / 224–270 | 0.22–0.38 | 0.49–0.64 |

**It does not transfer, not even on the body that learned the walk.**
- **Two different gaits:** the published policy walks with duty 0.47 /
  0.53 and a period of 0.98 `T`. The walk teacher shuffles at duty 0.92
  and 1.87 `T`.
- **The tracker:** it learned to reach the walk's plans and cannot make
  quick alternating steps. It falls, its history leaves anything either
  generator saw, and the plans drawn from that history are meaningless
  (period 0.5).
- **The autonomous reference fails too** (322 falls, forward). The flat
  generator has had no DAgger round, and its own chain diverges from a
  standing start.

This is the limit of the scheme as specified, not a defect of one piece.
A tracker trained by RL is an inverse model over the plans it practised,
and nothing makes it competent outside them. The contract makes a skill
*expressible* on any body. It makes it *executable* only on a body whose
tracker has practised plans like it.

The general correction is to train each body's tracker once on every
skill available, and to test with a skill held out of that set. It is
still N + M trainings, but on the union of skills rather than on one.
That changes the requirement ("trained once, with walking"), so it is
left to the user.

### 7.4 The NAO

`NaoTrack-v0` is the G1's tracking task on the NAO: the walk's contract,
plans read at the NAO's own control instants (0.126 `T` against the G1's
0.077), and the same reward. Training it from scratch exposed two
problems the G1 never met, because the G1 reached them already walking.

1. **A reference that has walked away teaches nothing.** Within 100
   iterations the NAO learned to stay upright (tilt terminations 60 % →
   2 %), but not to walk. It was 3.7 leg lengths behind a reference it
   cannot steer, so every kernel of the likelihood but the widest was
   dead, and the widest gave no gradient to start.

   Motion trackers end the episode once the body is too far from its
   reference to catch up (DeepMimic; PHC at half a metre), which keeps
   practice where the reward can teach. `contract_left` does the same:
   past half a leg length behind (several steps, more than the next step
   can recover), the episode ends as a fall does. This applies to both
   bodies in training; play captures count falls only. Within 90
   iterations the NAO kept up (lag 0.13 L, whole episodes).

2. **The bodies are not geometrically similar.** `z*` is in each body's
   leg lengths. In its nominal crouch the NAO's CoM sits 1.084 `L` above
   its soles, where the G1 walks at 0.95 `L`. That gap was nine tenths of
   the NAO's remaining error (1600 tolerances), which drowns every finer
   term:

   | field | NAO nominal | G1 walking | gap (tolerances) |
   | --- | ---: | ---: | ---: |
   | height | 1.084 | 0.950 | 99 |
   | foot height below CoM | −1.084 | −0.94 | 73–88 |
   | foot fore-aft | −0.051 | ≈ 0 | 5–8 |
   | foot lateral | 0.202 | 0.193 | 2–3 |

   Dynamic similarity in leg-length units asks the NAO to stand as low
   as the task's body did. The tracking task's nominal posture is now
   the crouch whose CoM height, from the URDF, is the contract's mean
   (0.942 `L`: knee 1.56 rad, hip and ankle at minus half of it). The
   actions are offsets around that crouch, which belongs to the tracker
   (`HUMANOID_POSTURE_CONTRACT`). In a smoke run the joint error started
   at 210 instead of 1600, and the 10-kernel term earned a hundred times
   more.

   Whether another length unit would make such bodies similar without a
   deep crouch is a design question. The obvious candidate is the CoM
   height, which is the pendulum's own length.

**At 500 iterations the NAO walks the G1's walk**
(`contract_nao_walk_mid_500.json`). It starts from standing at the G1
teacher's play command in its own units, on the true state, drawing
plans from its own history. Measured against the G1 teacher in body
units:
- **Stability and speed:** 0 / 32 falls, 0.0815 `V` (0.127 m/s, a speed
  ratio of 0.97).
- **Use gate:** passes at 0.153; every field but the feet's position is
  within 2 tolerances.
- **Gait:** stride 0.111 (−30 %), period 1.35 `T` (−28 %), duty 0.83 /
  0.79.
- **Against the plans:** it steps quicker and shorter than asked (asked
  period 1.66 `T`). The G1's tracker misses the other way, with fewer,
  longer steps.

**At 1500 iterations, on every path** (`contract_nao_walk_nao2.json`):

| path | falls | speed ratio (`V`) | stride | period | duty | use |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| deployed (the NAO's encoder) | 0 / 64 | 1.05 | −18 % | −21 % | 0.84 / 0.82 | 0.083 |
| true state, own history | 0 / 64 | 1.01 | −18 % | −19 % | 0.85 / 0.82 | 0.093 |
| true state, autonomous | 0 / 64 | 1.00 | −21 % | −21 % | 0.84 / 0.82 | 0.095 |

Every field is within 1.3 tolerances of its plan, except the feet's
position (about 7). The NAO's encoder (`train_body_encoder.py`, anchor
only) needed the starts:
- **Steady walking only:** fitted on that, its held-out error was 0.034,
  and the deployed loop fell 500 times in 540 episodes. Every deployed
  episode starts from the spawn pose, which the captures had discarded,
  so the encoder misread the start, the plan was meaningless, the NAO
  fell, and the same start came round again.
- **With starts and stops from the spawn pose:** its error is 0.031 and
  the deployed path walks.

**The published task on the NAO, zero-shot** (`contract_nao_new_task_nao2_*.json`).
The setup:
- the flat contract, with the NAO's encoder moved over;
- the NAO's tracker unchanged, still in the crouch it trained in;
- the G1's three test commands, converted to the NAO's units.

Results:

| command | falls (encoder / true state, of 64) | speed ratio | use |
| --- | ---: | ---: | ---: |
| forward | 1 / 3 | 0.20 / 0.14 | 0.67 / 0.57 |
| fast | 0 / 0 | 0.32 / 0.34 | 0.64 / 0.59 |
| turning | 0 / 1 | 0.41 / 0.37 | 0.71 / 0.64 |

The NAO stays up but does not follow: it shuffles (duty about 0.95)
where the published gait steps quickly with alternating feet (duty
0.5). The coverage limit of §7.3 holds on the second body. A tracker
trained on the walk alone executes the walk on any body, and no other
gait.

## 8. Zero-shot by coverage: every tracker practises a physical family of plans

§7.3 and §7.4 ended on a limit, not a defect: a tracker trained by RL is an
inverse model over the plans it practised. Trained on the walk's contract
alone, the G1 fell 120–240 times on the published policy's gait and the NAO
shuffled. This section keeps the contract and everything about it — the
generator per task, the encoder and tracker per body, ancestral sampling when
deployed, the autonomous reference in training, the precision likelihood, the
anchored integral and the path in view — and changes one thing: the
distribution of plans each body's tracker is trained on.

### 8.1 Why, and what is claimed

- **Zero-shot, defined.** No gradient step of a body's encoder or tracker uses
  data from a new task's generator or teacher. A new task trains only its own
  generator, on G1 data (its teacher, and DAgger on the G1).
- **Why it fails today.** The tracker's loss on a task is bounded by its
  training loss times the concentrability `C = sup d_task / d_train` (Kakade
  and Langford, 2002; Munos, 2003). Outside the support of `d_train`, `C` is
  infinite and nothing is guaranteed. The walk's support does not hold the
  published gait (duty 0.5 and 0.98 `T` against 0.9 and 1.9 `T`).
- **What to train on, without knowing the task.** Over tasks whose plans have
  density at most `ρ` on the set `F` of physically feasible plans, the worst
  `C` is `ρ / inf d_train`, minimised by the uniform distribution on `F`: the
  minimax argument of domain randomisation, applied to plans.
- **Why `F` can be sampled.** A plan of `z*` is not a free 13 × 27 array.
  Centroidal dynamics fix it once the contact schedule, the footholds and the
  CoM's start are given, so `F` is parameterised by about a dozen
  dimensionless numbers per segment. They are drawn uniformly, and
  log-uniformly for the period (a scale: Jeffreys' prior).
- **Why it is shared.** The parameters are dimensionless (Froude, `T`, `L`,
  duty, `m g`), so one family trains every body; each body executes the part
  of it its tracker can (its competence), measured with no task in sight.
- **When a task transfers.** Its plans lie in the family (i), within the
  body's competence (ii) — which the tracker has to have learned (iii) — and
  the tracker does not treat real plans differently from procedural ones of
  the same gait (iv: no shortcut, Geirhos et al., 2020). The closed loop must
  also keep the task's generator in its own support (v), and the encoder must
  cover the states (vi). Each level of the evaluation isolates one of these.

### 8.2 The family

`transfer/latent/procedural.py`: a point mass on massless legs that push
along themselves.

- **Contacts.** Event-based contacts on a phase clock; duties below one half
  give flight phases.
- **Loads.** A half-sine of load per stance, its impulse set by what follows
  the stance — one model across the walk–run transition.
- **Stepping and balance.**
  - Foot placement is deadbeat for the model's own centre of pressure,
    double support and turns included, with an integral on the speed.
  - The CoP tracks the periodic DCM within the soles.
  - Standing holds the DCM over the feet; starts lean before they step.
- **Recovery.** A recovery steps in place when a compound transition drifts.
- **Calibrated fields.** The three fields a point mass cannot derive — tilt,
  the base's rate beyond the turn, and angular momentum — are calibrated on
  both bodies' walks (`procedural_calibration.json`). Their model error is 4–6
  times the encoder's, and becomes their tolerance on procedural plans.

The ranges, set by physics before any task was trained:

| parameter | range |
| --- | --- |
| speed `v_x`, `v_y` | −0.10 – 0.45 `V`, ±0.15 `V` |
| turn | ±0.3 / `T` |
| period | 0.6 – 3.0 `T`, log-uniform |
| duty | 0.30 – 0.95, asymmetry ±0.08 |
| step width | 0.20 – 0.55 `L` |
| CoM height | 0.85 – 1.00 `L` |
| clearance | 0.01 – 0.12 `L` |
| lean | 0 – 0.20 rad |

Constraints:
- the shortest swing is 1.5 control steps of the body, the only body-specific
  term;
- stride ≤ 1 `L`;
- step ≤ 0.55 `L`;
- mean swing speed ≤ 2 `V`;
- the stance's CoM bounce is ≤ 0.08 `L`.

Measured (`tests/test_procedural.py`, random schedules over 77 `T`):
- the vertical acceleration is the load minus the weight;
- the CoP implied by the horizontal acceleration stays under the feet;
- fixed gaits come out with their period, duty and speed (3 %, 0.03, 10 %);
- 2.8 % of the time is spent recovering;
- 0.7 % of episodes fall, after which the plan holds still and the body's
  episode ends on its lag.

**Coverage, H1** (`procedural_coverage.json`):

| capture | inside the family for the G1 | for the NAO |
| --- | --- | --- |
| the walk teacher | yes | no: its 0.15 `T` swing is under 1.5 NAO steps |
| published policy, forward / fast / turning | yes | yes |

The published policy walks with its CoM at 0.86 `L` (the walk at 0.95), at
the low edge of the range.

**Style, a diagnostic.** A logistic probe separates walk windows from
procedural ones with AUC 0.96 on every field and 0.89 on the derived fields.
On the derived fields the published policy's windows are nearer to the
family than to the walk (60–90 %); on every field they are nearer to the walk.
The calibrated fields give the source away. This is the shortcut risk that
level L3 below exists to catch.

**Data hygiene.**
- The ranges were set before any tracker was trained on them.
- The published policy's gait was known (§7.3), so it is a semi-blind test.
- Isaac Lab's `Isaac-Velocity-Rough-G1-v0` is the blind one. Its gait is not
  looked at until both trackers are frozen, and it is evaluated once.

### 8.3 The loop, pre-registered

Each tracker is trained on 70 % procedural and 30 % walk plans, warm-started
from its walk tracker (G1 round 12, NAO crouch 2). It is then evaluated level
by level. Levels L1–L3 read the true state (plus the encoder's noise). L4
reads both the true state and the encoder; the deployed path decides.

| level | condition | gate |
| --- | --- | --- |
| L1 learning | (iii) | random schedules as in training: use < 0.5, falls ≤ 10 %; competence map before/after, reported |
| L1b no regression | the walk | each path: falls ≤ round 12's + 2 points, speed ratio ≥ 0.9, stride/period/duty ≤ round 12's + 5 points, use passes |
| L2 twins | (ii) + (iii) on the published gait | each procedural twin: falls ≤ 5 %, speed within 10 % of asked (floor 0.05 `V`), period and duty within 10 %, use < 0.5 |
| L3 the published generator, autonomous | (iv) | started inside the published gait: use < 0.5, falls ≤ 5 %, body within 10 % of asked |
| L4 the published generator, deployed | (v) + (vi): the zero-shot result | gates B and C against the published policy, per command |

One correction per round, chosen by the first level that fails. A correction
is never aimed at the published gait.

| fails | correction |
| --- | --- |
| L1 | learning-progress curriculum over cells of the family (ALP-GMM, Portelas et al., 2019; PLR, Jiang et al., 2021) → action std reset to 0.6 → widen the network, function-preserving |
| L1b | walk share 0.3 → 0.5 |
| L2 (L1 passes) | the curriculum of L1 |
| L3 (L2 passes) | model error on more fields → randomised shapes (load ramps, swing profile) → procedural plans only |
| L4 true state (L3 passes) | DAgger of the published generator on the G1 (task side) |
| L4 encoder (true state passes) | refit the body's encoder on more procedural states and starts |

Limits and stopping rules:
- **Rounds.** At most three RL rounds on the G1 and two on the NAO.
- **The NAO.** It is trained only once the G1 passes L4, and the published
  generator stays frozen for it. A twin outside the NAO's competence is
  reported as a physical limit of that body, not a transfer failure.
- **Escalations, decided by the user.** A model-based decoder, training on
  the union of skills, or a different length unit.

**Threats to validity, declared.**
- The published policy is a semi-blind test.
- Each RL round is a single seed.
- The 10 % / 2-point thresholds are §5's, unchanged.
- The deployed path decides, per command.

### 8.4 Log

- *Pre-registration* (this section, before any tracker was trained on the family).
- **Baselines**, the trackers trained on the walk alone (`competence_g1_r12.json`,
  `competence_nao_nao2.json`):
  - *Grid:* 3 of 61 probes on the G1, 6 of 57 on the NAO.
  - *Twins:* the walk's twin passes on both — the walk tracker follows the biped's walk as
    it follows the real one (use 0.02). The published policy's twins fail on duty exactly
    as the zero-shot did (0.85 against 0.5): the map predicts the failure without the task.
- **G1 round 1** — round 12 resumed with 70 % procedural plans, for 2 × 1000 iterations
  (`competence_g1_r1_*.json`, `zero_shot_g1_r1_4196_walk.json`):
  - *L1 passes:* random schedules, 0 falls, plans used.
  - *Map:* the grid goes 3 → 8 → 13 of 61. On every grid probe falls, speed and use now
    pass, and the period on 85 %; the duty on 20 %.
  - *L2 fails, on the duty alone:* the published policy's twins come out with period and
    velocity exact, no falls, plans used, and duty 0.67–0.78 against 0.47–0.53. Asked to
    swing, the foot carries more than 5 % of the weight 40 % of the time, whatever the
    contact threshold, and lifts half as high.
  - *L1b, measured against round 12, fails, and not because of the tracker:* it now steps at
    the rate its plans ask. Body over asked: period 1.12–1.18 and stride 1.04–1.10, against
    round 12's 1.43–1.56 and 1.48–1.61. Round 12 passed gate C on the deployed path by not
    following its plans; the walk's generator asks for quick steps (§7.1, its chain bias,
    asked period 1.23–1.47 against the teacher's 1.87), and a tracker that follows them takes
    them. More walk in the mix, the pre-registered correction, would teach it to ignore its
    plans again; it is not applied. The walk's fidelity now waits on the task side.
  - *What the duty failure is, measured:* on the twins the joint error sits at about 15
    tolerances, dominated by fields the encoder knows to a millimetre (CoM height 43,
    feet 11–43). The reward's kernels were 1, 10 and 100: at 15 the first is dead and the
    second flat, and a correct swing changes the reward by 0.2 % per tolerance — no
    signal, which neither a curriculum (the pre-registered correction for L2, about
    coverage, which every other criterion shows is there) nor more exploration supplies.
  - *Two hypotheses ruled out:* the encoder's error on the family's states (refitted on
    them it is 0.2–0.6 of the walk-fitted one: the tolerance is not too tight for lack of
    data), and the point-mass model's error on every field
    (`procedural_model_error.json`: 3–14 encoder errors, mostly the walk teacher's own
    stride-to-stride irregularity; widening by it would erase the contact signal).
- **Round 2's correction** (a deviation from the pre-registered order, for the reason above):
  the same likelihood with the error's scale marginalised by an octave quadrature,
  kernels 1, 2, … 128 with the old power law of weights (`OCTAVES`). The three old kernels
  are among them. At 15 tolerances the gradient is 13 times the old, at 2 it is 1.5
  times: no field loses its pull.
- **G1 round 2** — the octave likelihood, round 1 resumed for 1000 iterations
  (`competence_g1_r2_5195*.json`, `zero_shot_g1_r2_5195_*.json`):
  - *Map:* the grid goes 13 → 26 of 61, the manoeuvres 0 → 5 of 6. Over the grid the duty
    passes on 43 % (20 %) and the period on 98 %.
  - *L2 passes:* all four twins. The published policy's three come out with duty
    0.51–0.58 against 0.47–0.53, period exact, no falls, use 0.00–0.03.
  - *L1b:* no falls on the true state or through the encoder (3 of 64 autonomous), speed
    ratio 1.04–1.06, use 0.06 (round 12: 0.15). Stride −15 % and period −20 %: the walk's
    generator's chain bias, followed (§7.1).
  - *L3:* started inside the published gait, the tracker walks it on the published
    generator's own chain with no falls in 128 episodes, at period 0.97 and duty 0.43/0.53
    against the published policy's 0.98 and 0.47/0.53. Against the plans' own stitched
    gait (period 0.77, its seams every 13 samples counted as steps) the gate's check fails;
    against the policy it walks like, it does not.
  - *L4 fails on falls:* 119–169 per 32 environments over 900 steps, with the plans used
    (0.016–0.04). The falls come 60–105 steps into each episode, as the generator starts
    the body from rest — the published generator's own chain diverges from a standing
    start (§7.3), now on the body's history. A task-side failure (v): the pre-registered
    correction is DAgger of the published generator on the G1.
- **Task-side DAgger, round 1** (`task_dagger.sh`, `contract_flat_dagger1.json`,
  `zero_shot_g1_r2_5195_flat_dagger1_*.json`). The G1's frozen tracker ran with the
  published generator under five commands of the task's range, plus starts and stops. The
  published policy acted from 1024 of the loop's states, and the generator was retrained on
  its data and this round's (gate passes, chain period 0.93 against 0.98).
  - *Falls:* 0 deployed (136–169 before), 3–9 on the true state.
  - *Deployed:* period 0.96–0.98 against 0.98, plans used (0.31–0.36), every field
    within 1.5 tolerances of its plan.
  - *What fails now is the speed.* Asked 0.30, 0.60 and 0.32 m/s where the published
    policy walks 0.18, 0.49 and 0.24 at those commands; the body takes what it is asked.
    The generator's chain was already 11 % fast. Now that the tracker delivers its plans,
    that bias compounds upward through the body's history, where round 12 had delivered 75 %
    and compounded down. Stride, which is speed times period, fails gate C. The true-state
    path also suffers: its history's noise makes this generator's plans jitter, and the
    body steps at 0.6 `T`.
  - Both are the task side's, and DAgger is the correction: a second round, from this
    loop's states, at the speeds the published policy does not walk.
- **Task-side DAgger, round 2** (`contract_flat_dagger2.json`,
  `zero_shot_g1_r2_5195_flat_dagger2_*.json`). Round 1's loop was captured on both paths,
  the true state and the encoder. The published policy acted from 1024 of its states, and
  the generator was retrained on the teacher's data and both rounds' (gate passes, chain
  period 0.94).
  - *Deployed, no falls.* The plans are used (0.31–0.49) and period 0.96–0.98.
  - *The speed runs further away.* Asked 0.48, 0.76 and 0.40 m/s at the three commands,
    against round 1's 0.30, 0.60 and 0.32 and the policy's 0.18, 0.49 and 0.24. Stride is
    +49 to +155 %.
  - *True state:* 14–36 falls, period 0.66–0.69. *L3:* 30 falls in 133 episodes, mostly in
    the first 1–3 s after starting from the policy's noisy states, at commands up to
    0.96 m/s forward and 0.42 m/s sideways.
- **Why DAgger diverges here, measured** (offline, nothing trained):
  - *The generator on its own chain is the policy's.* Chained on its own plans from the
    G1's spawn state — what a perfect tracker would walk — rounds 1 and 2 start 32 of 32
    chains, reach 0.20, 0.51 and 0.27 m/s within two seconds, and keep period 0.93–0.99 and
    duty 0.49/0.54. The published policy, in its training environment, walks 0.20 m/s at the
    first command. The pre-DAgger generator starts only 6–7 of 32. Started from the loop's
    own 0.49 m/s states, every round's chain returns to 0.20 within 6 s, also when every
    plan is overshot by 3 %.
  - *On the body's history it barely reverts.* Fed the deployed loop's own windows (encoder,
    true state, or true state plus noise), round 2 plans 0.47 m/s where the body was
    walking 0.49. The tracker overshoots its plans by 1.7 %. That balance is the loop's
    fixed point.
  - *The policy's labels say so.* From the loop's states, the published policy speeds up
    for the first 0.26 s — a commitment's length — and only then slows down (command
    0.25: 0.35 → 0.43 → 0.37 → 0.23 m/s over one second). Round 2's fast half goes
    0.50 → 0.53. From its own states the jump is small (0.183 → 0.196, or 0.186 with the
    joint targets in force written as its last action, `--last-action source`). From the
    loop's states the same correction changes the jump by 0.002 m/s, so the transplant is
    not to blame.
  - *So the labels are right and the loop is not.* Every 0.26 s the generator is asked,
    again, what the policy does in the first 0.26 s after finding itself in the G1
    tracker's state, and the policy's answer — a pendulum ahead of its foot, caught by a
    quick step — is to speed up. Each round adds data at the higher speeds the previous
    round reached; the fixed point moves up with it.
- **What this says about reading the body's history.** Let a plan revert a deviation from
  the commanded gait by a fraction `κ` per commitment. Let a body's tracker overshoot by
  `ε`, and let `δ` be how the task's expert answers that body's way of moving. The loop
  then settles where `(1 − κ)(1 + ε)` and `δ` balance, which is far from the expert's gait
  whenever `κ ≲ ε`, as here. Both `ε` and `δ` belong to the pair of task and body:
  - a generator kept stable on a body's history must be trained on that body's
    histories — N × M trainings, the thing the contract exists to avoid;
  - DAgger on the G1 writes the published policy's reaction to the *G1 tracker's* style
    into the task's generator, which is wrong for any other body.

  Reading its own plans instead (the autonomous reference, already the tracker's training
  mode and §7's L3), the generator's chain is a property of the task alone — it is what
  the task's gate checks. The tracker follows the plan's anchored displacement, so its
  `ε` cannot accumulate, and the body enters the history only where it must: at a start
  and when it has left the plan (a resync). That splits the loop into the two halves N + M
  can certify separately. Task-side DAgger keeps one job, the starts, which is what it
  fixed (6 → 32 of 32).
- **Amendment to L4, made before its result was read:** the deployed path is the encoder
  with the autonomous reference. The physical reference stays as a diagnostic, and its
  failure above stands.
- **L4, amended** (`zero_shot_g1_r2_5195_flat_dagger2_auto_l4_*.json`). Round 2's tracker
  and round 2's generator, both frozen, at the three commands:
  - *Deployed:*
    - no falls in 64 episodes at any command;
    - plans used (0.22–0.27);
    - speed 1.10, 1.06 and 1.15 times the policy's;
    - stride +3 to +7 %, period −2.5 to −3.5 %;
    - the plans ask for duties within 4 % of the policy's (0.48–0.50 / 0.54).
  - *Gate C* passes at the fast command. At forward and turning it fails on the left
    foot's duty alone: 10.2 % and 10.5 % short of the policy's, against a limit of 10 %.
    The body's left stance is 13–14 % shorter than its plan asks; the right is within 3 %.
  - *True state:* 11–17 falls, period 0.60–0.65, use 0.78. This path writes the task
    contract's `σ` into the tracker's view, and the published contract's `σ` is the walk
    encoder's error on the published policy's states when the contract was trained. That is
    8.8 times (median; up to 56) the noise the tracker trained with, and 4–10 times the
    deployed encoder's error on these same runs. So this path measures that earlier
    encoder, not the one deployed, and is not read as a diagnostic of this one.
  - *The left foot, measured without the task.* Over the competence map
    (`competence_g1_r2_5195.json`), where every plan's two duties are equal, the body's
    left duty is 0.041 below its right; on moving gaits near duty 0.55 it is 0.054 below.
    The family is mirror-symmetric (tested) and so is the robot; the tracker is not. That
    asymmetry belongs to the body side, the map shows it without the task, and it decides
    both of the remaining failures.

## What this stage hands to the next

- `common/skill_state.py` and `transfer/embodiment.py`: the shared coordinates
  and the NAO's place in them, both NumPy and tested; `latent/skill_torch.py`,
  the same on the GPU.
- **The contract (§7), the interface that replaced the learned action code.**
  - *Task side:* a generator per task (`latent/contract.py`, `train_contract.py`,
    DAgger rounds with `dagger_contract.sh`). The walk's is `walk_dagger2e`; the
    published task's is `flat_h20e`.
  - *Body side:* the shared tracking task (`transfer/tasks/tracking`): autonomous
    references in training, the body's own history when deployed, and the
    reference's path in view.
  - *Per body:* an encoder refitted on the body's own states (`train_encoder.py`,
    `train_body_encoder.py`), moved between a body's contracts with
    `body_encoder.py`.
- **G1:** round 12 passes both pre-registered gates on the deployed path
  (speed ratio 1.00; stride, period and duty within 6.4 %; 1 fall in 64).
  Fed the true state it still misses stride and period by about 22 %. The
  encoder's error moves the gait by about 15 %, and the generator's chain
  steps fast (§7.1, open).
- **The published task, zero-shot on the G1, fails** (§7.3). A tracker
  practised on one gait does not follow another. The open decision is
  whether each body's tracker trains once on every available skill, tested
  on a held-out one such as Isaac Lab's `Isaac-Velocity-Rough-G1-v0`.
- **NAO** (§7.4): trained once on the walk's contract, it walks the G1's walk
  on every path, deployed included.
  - Results: no falls, the Froude-matched speed (ratio 1.00–1.05 in `V`), its
    plans used; stride and period −18 to −21 %.
  - What it needed: its own encoder fitted on starts, and the crouch dynamic
    similarity asks for.
  - The published task does not transfer to it either.
- The command in body units: the NAO's speed follows from `c*` and its own `V`.

## References

- Alexander, R. McN. and Jayes, A. S. (1983). A dynamic similarity hypothesis
  for the gaits of quadrupedal mammals. *J. Zool.*
- Buckingham, E. (1914). On physically similar systems. *Phys. Rev.*
- Charvet, V., Stein, S. and Murray-Smith, R. (2025). Improving controller
  generalization with dimensionless Markov decision processes. arXiv:2504.10006.
- Devin, C. et al. (2017). Learning modular neural network policies for
  multi-task and multi-robot transfer. *ICRA*.
- Englsberger, J., Ott, C. and Albu-Schäffer, A. (2015). Three-dimensional
  bipedal walking control based on divergent component of motion. *T-RO*.
- Girard, A. (2024). Dimensionless policies based on the Buckingham π theorem.
  *Mathematics* 12(5), arXiv:2307.15852.
- Laskey, M. et al. (2017). DART: noise injection for robust imitation
  learning. *CoRL*.
- Oke, N. et al. (2026). Allometric scaling laws for bipedal robots.
  arXiv:2603.22560.
- Ordoñez-Apraez, D. et al. (2025). Morphological symmetries in robotics.
  *IJRR*.
- Orin, D., Goswami, A. and Lee, S.-H. (2013). Centroidal dynamics of a
  humanoid robot. *Autonomous Robots*.
- Pascoa, F., Lalonde, I. and Girard, A. (2025). Zero-shot policy transfer in
  reinforcement learning using Buckingham's π theorem. arXiv:2510.08768.
- Lipman, Y. et al. (2023). Flow matching for generative modeling. *ICLR*.
- Peng, X. B., Abbeel, P., Levine, S. and van de Panne, M. (2018).
  DeepMimic: example-guided deep reinforcement learning of physics-based
  character skills. *ACM TOG*.
- Ross, S., Gordon, G. and Bagnell, D. (2011). A reduction of imitation
  learning and structured prediction to no-regret online learning (DAgger).
  *AISTATS*.
- Wang, T., Bhatt, D., Wang, X. and Atanasov, N. (2024). Cross-embodiment robot
  manipulation skill transfer using latent space alignment. arXiv:2406.01968.
- Wen, C. et al. (2020). Fighting copycat agents in behavioral cloning from
  observation histories. *NeurIPS*.
- Yan, Y. and Lee, D. (2026). Learning a unified latent space for
  cross-embodiment robot control. arXiv:2601.15419.
- CrossBFM (2026), arXiv:2609.38087; PHASOR (2026), arXiv:2606.01851;
  H-Zero (2025), arXiv:2512.00971.
- Caron, S., Escande, A., Lanari, L. and Mallein, B. (2019). Capturability-based
  pattern generation for walking with variable height. *IEEE T-RO*.
- Geirhos, R. et al. (2020). Shortcut learning in deep neural networks. *Nature
  Machine Intelligence*.
- Jiang, M., Grefenstette, E. and Rocktäschel, T. (2021). Prioritized level replay. *ICML*.
- Kakade, S. and Langford, J. (2002). Approximately optimal approximate reinforcement
  learning. *ICML*.
- Munos, R. (2003). Error bounds for approximate policy iteration. *ICML*.
- Portelas, R., Colas, C., Hofmann, K. and Oudeyer, P.-Y. (2019). Teacher algorithms for
  curriculum learning of deep RL in continuously parameterized environments. *CoRL*.
- Raibert, M. H. (1986). *Legged robots that balance*. MIT Press.
