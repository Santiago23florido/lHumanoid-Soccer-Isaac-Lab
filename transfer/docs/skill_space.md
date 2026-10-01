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

## What this stage hands to the next

- `common/skill_state.py` and `transfer/embodiment.py`: the shared coordinates
  and the NAO's place in them, both NumPy and tested.
- The G1 side of the latent space and the shared `π` and `T` — walking
  closed-loop, not yet pinned (stride timing, §5).
- For the NAO: an encoder trained against its own analytic `z*`, a decoder
  trained by RL with `π` and `T` frozen and the anchored dynamics term holding
  `u` to its physical meaning, and the defect table saying where the frozen
  skill will ask for more than the body has (hip and knee torque, joint speed)
  and where it will leave capacity unused (ankle torque, foot size).
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
