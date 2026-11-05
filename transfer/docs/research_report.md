# Research report: contact transitions across the transfer chain

The record of the study started on 2026-11-04 on branch `feat/contact-fidelity`. Its
pre-registrations and short log live in [`skill_space.md`](skill_space.md) §10; this file
holds the audit, the reasoning behind each choice, the experiment log and the
conclusions. Every number names the script and the result file it comes from. Each
claim is marked **validated** (a pre-registered test ran and its result is read from its
file), **measured** (computed from existing captures, no new run), **predicted** (offline
model, not yet run in the simulator) or **hypothesis**.

## 1. Architecture audit: what is deployed, and what was tried

### 1.1 The deployed system (`transfer/results/incumbents.json`)

```
teacher (Isaac Lab's published G1 policy)
  └─ captures of z*  ──►  generator G_task  (offline, G1 data only)
body:  proprioception ─► estimator E_b ─► ẑ*        tracker D_b (RL, procedural family)
deployed loop:  G_task reads its own committed plans (autonomous reference)
                plan of z*, resampled to the body's instants ─► D_b(plan, ẑ*, path) ─► joint targets
```

- **The skill state `z*`** (`common/skill_state.py`, `transfer/latent/skill_torch.py`):
  - *What it holds.* 27 dimensionless numbers in each body's own units
    (`L` = hip-pitch-to-sole length, `m`, `g`): CoM height (1), CoM velocity (3, heading
    frame), gravity in the base (3), base angular velocity (3), feet positions from the
    CoM (6) and velocities (6), foot load shares `f_z/(mg)` (2), centroidal angular
    momentum (3).
  - *Certified, not assumed.* Newton's vertical residual is 1.4e-7 g; mirror, gauge
    and similarity are tested as group actions.
  - *What is embodiment-invariant.* The equations of centroidal dynamics in these units
    are. Which states and contact wrenches a body can reach is not: foot size, torque
    and joint speed, as dimensionless groups in `transfer/embodiment.py`.
  - *Discarded by construction.* Joint configuration, the distribution of mass beyond
    centroidal inertia, and contact geometry beyond one load per foot. The centre of
    pressure and shear are not in `z*`.
- **The generator** (`transfer/latent/contract.py`; `flat_dagger2`, `rough_b4`):
  - *Model.* Conditional flow matching on the residual of a 20-sample future
    (1.55 `T`) over the current state.
  - *Conditioning.* The current `ẑ*`, a temporal-convolution summary of 26 samples of
    history (2 `T`, `SkillMemory`), the dimensionless command, and for `rough_b4` the
    history's noise level.
  - *Network and sampling.* The velocity field is a 3-layer MLP of width 1024 with
    sinusoidal flow-time features. Loads are modelled as `log(load + 0.01)`; sampling
    is midpoint, 8 steps.
  - *Training.* On the teacher's states only, never its actions, with history noise of
    the body estimator's precision: `rough_b4` log-uniform across 0.125–32×, read as an
    input. `flat_dagger2` was trained under the inflated noise found in §8.4 (σ median
    3.1) plus two task-side DAgger rounds for its starts.
- **How plans are consumed** (`transfer/tasks/tracking/command.py`, `ReferenceCommand`):
  - *Commitment.* A plan is held for its commitment (13 grid samples, 1 `T`; 8 NAO
    steps) and redrawn.
  - *Autonomous reference* (training and deployment). The generator reads the plans it
    committed to, plus the estimator's noise, so the body cannot steer what it is
    asked. The body enters at resets and resyncs.
  - *Resampling.* The history is read onto the generator's grid and each plan back onto
    the body's control instants by linear interpolation in time: the NAO's step is
    0.1258 `T` against the grid's 0.0774. For the G1 both are selections.
- **The estimator** (`proprioceptive_skill_state`). Every field except the loads comes
  from the body's kinematics, IMU and model; height is measured from the sole's lowest
  corner. The loads come from a learned MLP encoder.
- **The tracker** (`g1/tasks/g1_track`, `nao/tasks/nao_track`; rsl-rl PPO, plain MLP):
  - *Inputs.* The held plan over a view (8 G1 samples, 4 NAO steps) relative to `ẑ*`,
    `ẑ*` itself, the reference's path (CoM and heading relative to the body), and
    proprioception.
  - *Reward.* A likelihood of the whole state, marginalised over the error's scale by
    an octave quadrature (kernels 1–128 tolerances; `OCTAVES`). Each field of `z*` is
    scored at the estimator's precision, plus the plan's integrals at σ√t. The G1
    scores loads in log space; the NAO in linear units.
  - *Early termination* when the body lags its reference by more than 0.5 `L`.
  - *Training plans.* 70 % come from a procedural point-mass biped
    (`latent/procedural.py`, gaits drawn from physical ranges) and 30 % from the walk
    contract.
  - *Symmetry.* Trained with a mirror loss, deployed symmetrized.
  - *Checkpoints.* G1 `warm_r3_logload`/7193; NAO `r1_3498`.
- **Evaluation:**
  - *The ladder:* L1 the family; L1b the walk (S0); L2 twins; L3 chain; L4 the published
    skill at three commands, 32 envs × 900 steps, deployed path.
  - *The competence map:* 61 grid probes and 36 corners.
  - *Gate C:* speed ≥ 0.9 of the teacher's, fall excess ≤ 0.02, stride, period and both
    duties within 10 %.
  - *The adoption rule:* better than 2 bootstrap SE on the target, no regression on S0;
    methods compared by seed means.

**What is learned from what.**
- The generator learns the skill from the teacher.
- The tracker learns to execute physical plans from the procedural family and the walk,
  never from the task.
- The estimator's loads are learned on the body's own practice.
- What is shared between bodies is `z*` and the generator.
- The family's parameters are dimensionless, so one family trains both bodies.

### 1.2 What was tried before, and what it taught

| Variant | Hypothesis | Outcome | Lesson carried |
| --- | --- | --- | --- |
| Learned action code `u` with per-body encoder/decoder (§4–6) | A latent action carries the skill | Imitation R² 0.97, but stride timing missed by 12–29 %; a free `ζ` leaked the body (R² 0.89 of joints) unless bottlenecked | A deterministic teacher's actions cannot make a student use a plan (§7); a free channel is where a body leaks in |
| Contact-schedule planner (§7) | An explicit timing plan fixes timing | Calibrated, informative, unused (+0.004 R²) | Per-step imitation cannot force plan use |
| CVAE generator → flow matching (§7.1) | Distributional futures keep contact changes | CVAE averaged lift-offs into ramps; flow passes calibration | A mean future is not a future |
| Loads in log space in the generator (§7.1) | Forces cannot be negative | Fixed swings collapsing to one step | Model a variable in a space whose boundary is physical |
| Physical reference (plans restart from the body) (§7.2, §8.4) | Ancestral sampling of the teacher | Trackers learned to stand (veto); DAgger ratcheted speed to 2.6× | The body must not steer its reference: autonomous reference |
| Precision tolerance, anchored integrals, path in view (§7.2) | A drawn plan is a commitment; a bias must cost | Rounds 9–12 walk; the path made the reward Markov | Score what the body is asked to commit to, and let it see what it is scored on |
| Procedural family + octave likelihood (§8) | Zero-shot by coverage of physical plans | G1 grid 3 → 39/61; flat transfers to both bodies | Coverage of the plan distribution is the zero-shot mechanism |
| Mirror augmentation → mirror loss → symmetrized deployment (§8.4) | The optimal tracker is equivariant | Augmentation failed on a warm start (importance ratio ≈ 0); the loss and symmetrization fixed the left-right duty bias | Equivariance by loss and projection, not augmentation, on warm starts |
| Chain-training objectives: DaD, phase retrieval, rhythm (§8.4) | A sequence objective fixes chain drift | All four made the chain worse | The cause was the training noise (B3): check σ before objectives |
| B3/B4 noise (§8.4, §9.1) | The phase is unreadable under 6σ history noise | rough chain 0.162 → 0.033; B4 robust across levels | A generator must see the noise the deployed loop reads |
| Learned encoder → kinematic estimator (§9.1) | Kinematics holds where a learned map does not | G1 rough falls 45/83 → 15/64 | Never learn what physics computes |
| Decoder v2: physics prior (IK + feed-forward + RL residual), DL core (§9.1) | Physics in the decoder extrapolates | Robust on rough (0 falls vs a distilled control's 62/225), but a slower walk and worse family duty; single seeds | Compare at equal competence, against a control distilled the same way |
| Free latent `ζ` with entity attention (F5) | A shared, body-blind latent carries balance `z*` lacks | Informative on the G1 (−25 % error), 2–3 % on the NAO; not in the contract | A body-blind latent mostly encodes the body it is richest on |
| Log-space loads in the tracker's reward (§9.1) | Duty is a 5 % threshold the reward must resolve | G1 grid +5.3 ± 2.1 over 3 seeds, adopted; no change on the NAO | Score a thresholded quantity in the space where the threshold sits |

**Open before this study.**
- The duty factor, on both bodies.
- The NAO's duty gap, attributed to its control rate.
- B1 (training on the task's plans) helps the G1 and hurts the NAO.
- The true-state path of zero-shot trackers fails on flat where the deployed one passes.

## 2. Failure analysis

### 2.1 A measurement bias: the NAO's "pass at its rate" (measured, corrected)

`verdict_at_body_rate` resampled the teacher to the NAO's step by blending loads
linearly, then thresholding at 5 %. That lengthens every stance by 0.03–0.06, although a
duty factor does not depend on the sampling rate (nearest sampling and log units: within
0.01). Unbiased, the NAO passes no command of either skill, and the record's "the NAO's
duty gap is its control rate" is withdrawn. Everything the wrong reading affected is
listed in `skill_space.md` §10.1 (`transfer/results/body_rate_verdicts.json`).

### 2.2 Where contact transitions are lost (measured)

From `transfer/results/contact_fidelity_table.json` (129 L4 results; deployed path):

1. **The generator blurs contact transitions.** Its plans spend 0.10–0.11 of the time
   in light contact (5–25 % of the weight) where the flat teacher spends 0.03–0.04; on
   rough 0.13–0.18 against 0.08–0.15.
   - *Not the sampler (W2, validated offline).* Midpoint 8 up to Euler 64 give the same
     light share within 0.003 (`sampler_sharpness.json`).
   - *Not the training noise.* `flat_b3`, trained at the body's own noise, blurs more
     (0.136).
   - *Not the data.* The training captures hold 0.04–0.05 on flat. Every flow variant
     produces 2–3× the light contact it was trained on: a learned property.
2. **The interface lengthens the NAO's asked stances.** The same linear blend as 2.1, in
   `plan_interpolation` and `TimeGrid.interpolation`: the NAO is asked 0.022–0.030 more
   duty than the generator plans (predicted by `chain_at_rate.py`, whose linear chain
   reproduces every recorded capture's asked duty within 0.007).
   - *Not transmitted to the body (W1, validated; `skill_space.md` §10.3).* Read in the
     generator's space, the asked duty falls by 0.029 (flat) and 0.024 (rough), as
     predicted. The body's duty does not fall: +0.006 and +0.003. The body's lift-offs and
     touchdowns stay where they were while the plan's move by an eighth of a control step
     (`w1_contact.json`). The plan's excess is real, but it is not what makes the NAO's
     stances long.
3. **The trackers make contact binary.**
   - *G1, zero-shot.* Where light contact is asked, the foot is in the air 0.47–0.53 of
     the time. The left foot lifts 0.067 `T` early and lands 0.058 `T` late, so the
     stance is 0.06–0.09 short of asked on rough.
   - *G1, B1* (task-trained): airborne 0.30–0.35, and the duty is right.
   - *The family.* It asks for light contact 5.5 % of the time.
   - *Real and procedural plans are treated differently.* On procedural plans the
     zero-shot stance runs 0.02 *longer* than asked; on rough's plans 0.09 shorter.
   - *Not the family's load shapes (W3 step 1, validated; `skill_space.md` §10.5).* The
     frozen trackers were given in-family stances with real gaits' slow unloading (warp
     0.5). Their stances run *longer* than asked: G1 +0.039 ± 0.004 more than at the
     half-sine, NAO +0.019 ± 0.005. The foot is no more often airborne, and the change is
     an earlier touchdown, not an earlier lift-off (`transfer_probes_*.json`). Rough's
     early lift-off is not reproduced by the shape of the load transfer alone.
   - *The plans load feet they hold above the ground* (`plan_contact.json`, from
     `plan_contact.py`; measured, a correlation; `skill_space.md` §10.10).
     - The teachers' loaded feet rest at 0.000 `L` (90th percentile 0.006 flat, 0.013
       rough). The plans the G1 was given hold them at 0.018 (flat) and 0.029 (rough) at
       the 90th, and the biped's at 0.
     - The higher a plan holds the still-loaded foot before a lift-off, the earlier the
       G1 lifts off (Spearman −0.35 on rough). The higher the first loaded samples, the
       later it lands (+0.49).
     - Rough's plans hold loaded feet higher than flat's. That fits rough failing and
       flat passing, and B1 learning, from rough's plans, which field to believe.
     - W8 puts the plans' loaded feet on the ground at deployment, on the frozen trackers.
     - *Shown since, by intervention, on the G1* (W8, §5.5). Grounding the plans' loaded
       feet lowers rough's largest deviation by 0.027 ± 0.004 and flat's by
       0.019 ± 0.001. Lift-off comes later and touchdown earlier. Part of the rough deficit
       is the plans' inconsistency. On the NAO it does not carry (§2.3).
4. **The NAO lifts its feet half as high as asked, at every cadence.**
   - *On the skills.* Plans ask 0.07–0.09 `L`; the body gives 0.03–0.05; B1-NAO gives
     0.01, with grazes in half of every planned swing.
   - *On the family's probes* (`clearance_probes.json`, from `clearance_probes.py`): 0.53–
     0.64 of the asked clearance on the grid, the slowest strides (2.6 `T`) included, and
     0.37–0.47 on the corners (asked 0.055–0.129 `L`, reached 0.026–0.048). The G1 reaches
     0.93–1.06 everywhere.
   - *Not a body limit: a thin body margin at the plans' swing speed*
     (`swing_bandwidth.json`, from `swing_bandwidth.py`).
     - **Two corrections.**
       - *The first version* said "Not a body limit" because B2 (the NAO trained on the
         task's own reward) lifts 0.08–0.31 `L`. B2's mean stance and flight last longer
         than 1.55 `T` (its note in `transfer/paper/README.md`), where only the joints'
         ranges bound the lift (0.318 `L` at the walking crouch). B2's lifts therefore say
         nothing about the plans' swings, which are 2–5 control steps long in 97 % of the
         skills' swings. That version had put the whole shortfall on the tracker.
       - *The second version* said "a body limit at the plans' swing speed". The ceiling
         it rested on assumes a leg at rest at lift-off. The NAO's swings do not start at
         rest, and a fifth to a quarter of them beat that ceiling (below). That version
         affected three things: it put most of the shortfall on the body; it put the NAO's
         leg gains and velocity limit to the user as the levers; and it made the bound,
         not 1, a NAO candidate's clearance ceiling (§3). No run and no pre-registration
         used it.
       - *Its ceilings came from a square grid* that missed the flat-foot boundary. The
         range-only ceiling read 0.314 `L` instead of 0.318, and the G1's ceilings read up
         to 3 % low, below their own uncapped values. No conclusion rested on the
         difference. The grid now follows that boundary, and a test holds a ceiling never
         to drop when a limit is removed.
     - *The ceilings.* Whatever the targets, a joint at rest that must be back within a
       time moves at most as far as full effort, then full braking, allow. Effort is taken
       over the leg's own inertia, under the joint's velocity limit. Over the hip, knee
       and ankle pitch at the NAO's walking crouch (knee 1.557 rad, the foot kept flat):
       - *A leg at rest at lift-off, within the airborne time.* The sole rises at most a
         median 0.72 of the asked lift on the skills (0.77 on the probes), and 81 % of the
         skills' planned swings ask more (67 % of the probes'). The NAO beats this ceiling
         in 18 % of its swings (26 % on the probes). Its swing knee is already flexing at
         lift-off, at a median 2.87 rad/s (2.46), which is 0.45 (0.38) of its velocity
         limit.
       - *With one more control step of flexion before lift-off:* 1.01 (1.15). This
         ceiling holds; the NAO beats it in 1.6 % of swings (1.7 %).
       - *The G1:* 3.2 and 4.1 times its asked lift, beaten in 1.6 % and 0.3 % of its
         skills' swings.
     - *What the margin is.* The NAO's leg can just about make the asked lift, while the
       G1's has 3–4 times that room. The knee's velocity limit, the URDF's 6.40 rad/s, is
       what makes the margin thin: 1.02 rad per `T`, against the G1's 5.17. Without it the
       one-step ceiling would be 1.75 of asked rather than 1.01. The simulator enforces
       it: in 20 captures the NAO's swing knee never exceeds 6.51 rad/s. Dynamic
       similarity scales a plan by `L`, `m` and `g`; the joints' speed does not scale with
       them, and `z*` does not carry it.
     - *Where the shortfall sits.* The tracker reaches 0.51 of asked (0.54 on the
       probes), about half of what the leg allows. Its swing knee reaches the velocity
       limit in only 6 % of the skills' swings (9 % of the probes'), with a median peak of
       0.61 rad/`T`. Most of the shortfall is the controller's.
     - *One candidate for the controller's part.* The leg drives were sized critically
       damped against the whole body a stance leg carries (`derive_gains.py`). Against the
       leg alone, in swing, they are overdamped (damping ratio 2.0 at the hip, 4.0 at the
       knee, 15 at the ankle) and respond in 0.82–0.87 `T`. That is longer than the swing
       they must make (median 4 steps, 0.50 `T`); the G1's respond in 0.09–0.36 `T`. A
       saturated drive delivers its full effort whatever its gains, so the cost is a lead:
       the targets must sit well ahead of the leg. This fits the controller's part but has
       not been shown to cause it.
     - *Measured since* (`swing_drive.json`, from `swing_drive.py`). The NAO's swing joints
       move as isolated PD joints: one step of the PD law, against the leg's own inertia,
       predicts the next with R² 0.84 at the hip and 0.72 at the knee. Within a swing the
       hip delivers a median 0.34 of its target's excursion; the G1's hip delivers 0.60. W7
       (`skill_space.md` §10.9) tests faster drives on the frozen trackers.
     - *Shown since, by intervention* (W7, §5.4). Four times stiffer leg drives on the same
       frozen trackers raise the lift from 0.48 to 0.68 of asked. So the drives' lag is a
       cause of the controller's part, and the tracker was not the whole of it.
   - *Not a short swing* (`clearance_mechanism.json`, from `clearance_mechanism.py`). The
     NAO's feet stay in the air as long as asked (median ratio 1.00). Swings that last the
     full asked time still reach only 0.43 (corners) to 0.58 (grid) of the asked height.
   - *Part of it is where the NAO's sole point sits* (`sole_offset.json`, from
     `sole_offset.py`; skill_space.md §10.8).
     - The G1's sole point is its collision box's underside, measured in the simulator;
       the NAO's is the URDF's sole frame. A flat, loaded NAO foot reads +0.01006 `L`
       (2.494 mm, 44 captures), the G1's −0.00003 `L`. Every plan puts a stance sole at 0.
     - In the reward, that is about 6 tolerances in every stance sample (foot height is
       scored at 0.0016–0.0020 `L`), which no motion can remove. The NAO's foot-height
       error is a median 5.4–8.1 tolerances, 81–83 % of its foot-position term; the G1's
       is 0.9–1.9 (`clearance_mechanism.json`).
     - In the clearance, the shortfall splits into a low absolute peak plus this offset:
       grid 0.011 + 0.010 `L`, corners 0.037 + 0.010, both skills on S's seeds (forward
       command) 0.020–0.029 + 0.007–0.009. A tracker that met the absolute arc exactly
       would still lift 0.007–0.010 `L` short.
     - In the deployed estimator, the ground is the lowest sole corner, at the sole
       point's height, so the trunk reads 0.007–0.009 `L` low on flat (G1: 0.0001).
     - *A correction to this section's first version.* It said the shortfall "costs 15–20
       tolerances". That is the clearance gap. The reward sees the absolute peak's gap,
       6–18 tolerances, plus a constant 6 tolerances in stance. The conclusion "not a loose
       reward" stands; the reward's view of the gap was overstated.
   - *What remains.* The low absolute peak (0.011–0.037 `L`) is mostly the controller's.
     The NAO's leg, allowed the flexion it already starts before lift-off, can about make
     the asked lift; the tracker reaches half of that. The body's margin is thin, so no
     candidate should be expected to go much above 1.
     - *Candidates for the controller's part:* the overdamped swing drives, or a trade
       against balance. Within each probe, swings that lift higher carry more trunk-height
       error (Spearman +0.36 to +0.40; G1 −0.11 to −0.37). That fits the balance trade but
       does not establish it.
     - *For W6.* Its arm must lift 0.007–0.010 `L` more. On the skills the trackers reach a
       median 0.51 of the asked lift where the leg allows 1.01, so W6's target is
       physically reachable. It is not a free gain either.
   - *The low swings are the early landings* (`touchdown_lift.json`, from
     `touchdown_lift.py`; measured, a correlation).
     - On the NAO, a swing's peak correlates with its touchdown lag: Spearman +0.52 on flat
       and +0.38 on rough, under either interface. The lowest quarter of swings lands
       0.10–0.13 `T` early, about one control step.
     - The height at the second airborne sample, which a touchdown cannot have cut short,
       correlates +0.32 and +0.24.
     - On the G1 the two are unrelated (−0.03 to +0.08).
     - So the NAO's duty gap and its clearance gap look like one failure: a foot that does
       not rise comes down early, and its stance is long. Which causes which is not shown;
       a tracker that lifts more would show it.
     - *Shown since* (W7, §5.4). Faster drives raise the lift, and in the same runs the
       touchdowns come later, the stances shorten and the duty deviation halves. It is one
       failure, and the lift sets the stance.

### 2.3 Unresolved

- Why the G1's tracker lifts early on rough's plans: about a third of it is explained.
  - Family coverage of load-transfer shapes (H3) is not supported: the frozen tracker does
    not lift early on them (W3 step 1).
  - The plans' inconsistency is a cause. Grounding their loaded feet (W8, §5.5) moves rough
    0.027 of the 0.081 between the zero-shot model and B1.
  - The rest remains. Its candidates stay hypotheses: the generator's extra contact changes
    per plan (3.21–3.73 against the teacher's 2.92, `sampler_sharpness.json`), and the
    reward's resolution.
- Why grounding the plans hurts the NAO on rough (duty deviation +0.018 ± 0.001, W8).
  - *Measured* (`plan_contact.json`): a loaded NAO foot reads a median 0.012–0.013 `L`,
    because of its sole offset (§2.2, item 4). A loaded G1 foot reads 0.000–0.0005.
  - *Hypothesis:* grounded plans ask for a stance height the NAO's sole point cannot
    reach. W8's switch on W6's tracker (sole point at the resting contact) tests it.
- Why the NAO's absolute swing peak is low (0.011–0.037 `L` under the plan's), once the
  sole-point offset is set aside. W6 tests the offset's part (skill_space.md §10.8). For the
  rest, the NAO's leg can about make the asked lift once it may flex a step before
  lift-off, as it does (`swing_bandwidth.json`). Why the tracker reaches half of that is
  not explained.
- ~~Whether the NAO's low swings cause its long stances.~~ Answered by W7 (§5.4): raising
  the lift alone, through the drives, shortened the stances.
- Whether the NAO's real drives are as slow as the simulated ones. The simulated gains are
  derived for stance (`nao.py`), not identified on the robot, and the NAO's gait fidelity
  depends on them (W7). Nothing in this repository measures the real robot's drives.
- Whether W7's falls at gain 4 (13, all on rough at the fast command) are the frozen
  trackers' mismatch with a stiffer leg or a property of the stiffer body. W7b (gain 2 on
  all three trackers) and, if it passes, an RL arm on the faster body separate the two.
- Whether B1-NAO's shuffle is the interpolated plans' doing. W4 was to test it under the
  generator-space interface, and its condition (W1 passing) failed. The zero-shot trackers'
  contact events do not follow the interface, which makes that cause less likely; it was
  not run.
- The true-state path's short strides on flat (instrumentation).

## 3. Research decisions

- **The order, cheapest discriminating test first.**
  1. The measurement, because every later verdict depends on it.
  2. The interface (W1): a deploy-time swap on frozen trackers, which tests H1 alone.
  3. The generator's sampler (W2): offline.
  4. The family's load transfers (W3): an in-family probe first, then RL with three
     seeds paired with S's.
  5. B1-NAO under the fixed interface (W4), because the paper's cross-body claim rests
     on it.
- **Rejected for this study, with reasons.**
  - *`ζ` in the contract.* The NAO effect is 2–3 %, it is not on the diagnosed path, and
    a new channel needs its own leakage test.
  - *Decoder v2 again.* Studied at equal competence; it slows the walk.
  - *Sequence-level generator objectives.* Four failed for a cause B3 found.
  - *Reward re-weighting of the load field.* The log-load study already covers the
    resolution argument, and weighting is a tuning knob, not a mechanism.
  - *Changing the contact threshold.* It moves the metric, not the gait.
- **The family's load shape, a design corrected before training.**
  - *First design.* An impulse-normalised `sin(πs)^p` was built first. Printed at substep
    resolution, a peakier stance only deepened both ends into sub-threshold tails: the
    light share moved 0.064 → 0.066.
  - *What real stances do.* The teachers' stances are front-loaded (near peak in the
    first tenth, slow drain before lift-off).
  - *What replaced it.* The time warp `κ s^(κ−1) sin(π s^κ)`, which has that shape for
    κ < 1 and keeps the impulse by a change of variables.
  - *The data-hygiene cost.* The direction of the change was chosen after looking at the
    teachers' profiles; this is declared in §10.5.
- **Robustness (W5) is instrumentation.** Pushes and a degraded estimate decide nothing;
  they are added because no evaluation so far perturbed a body.
- **The NAO's sole point (W6), added after the plan was approved.** It came out of the
  clearance analysis while the queues waited for memory.
  - *Why it is a definition fix, not a strategy.* It changes where `z*` measures the foot,
    to the G1's definition (the resting contact). It does not prescribe how the body moves.
  - *How it is measured.* The offset comes from the simulator's own resting height,
    because the mesh accounts for only 0.84 of the 2.494 mm.
  - *Why it trains.* It cannot be a deploy-time swap: a frozen tracker would read its foot
    6 tolerances away from where it learned it.
  - *How it is budgeted.* One screened seed first, run after W5, so it never delays the
    approved queues. Seeds 32 and 33 run only on a passing screen.
- **The NAO's swing margin is reported; it calls for no change to the body.** It came out
  of the same wait.
  - *Why no run.* The shortfall is mostly the controller's (§2.2, item 4). The body's two
    levers are the leg gains (derived in `derive_gains.py` for stance) and the knee's
    velocity limit (the real NAO's motor). Both would widen a margin the tracker does not
    use, and either would change the body that every NAO result in this repository was
    obtained on.
  - *What it does change.* A NAO candidate's clearance is judged against 1, as before. The
    leg allows about that and not much more, so a ratio near 1 is the most a candidate can
    show, not a level to beat.
  - *A correction.* This entry first called the at-rest ceiling a body limit, put the two
    levers to the user as a decision, and made that ceiling a candidate's clearance limit
    (§2.2, item 4).
- **After W1 and W3 step 1, the RL arms they gated are not run.** That is W3's steps 3–4
  (about 13 h) and W4 (about 2.5 h). Both gates failed as their pre-registrations required:
  the interface does not reach the NAO's body, and the family's load shapes do not reproduce
  the G1's failure on rough. W5 and W6 run as queued. The freed budget goes to the user with
  the new framings (§2.3), not into a new arm chosen after seeing these results.
  - *The user's answer:* the choice is mine. *Chosen:* W7, the NAO's leg drives
    (`skill_space.md` §10.9).
  - *Why this one.* It is the most direct test of the strongest lead. Duty and clearance
    fail together on the NAO, swing by swing. The interface is ruled out. The drives
    measurably lag the targets (`swing_drive.json`).
  - *Why it is cheap.* It is deploy-time on frozen trackers (about 2 h), and its
    intervention on the lift tests whether the lift sets the stance, which no correlation
    can.
  - *What waits.* An RL arm is gated on it.
  - *The G1's rough question, offline first.* It found the plans' inconsistency (§2.2,
    item 3), and W8 (`skill_space.md` §10.10) tests it at deployment on both bodies' frozen
    trackers. It is the same kind of test as W7, and it touches no body.
  - *The order.* W7, then W8, then W6. The first two are short and decide what a later arm
    trains with. W6 had not started, and its queue was stopped while waiting and relaunched
    behind them.
- **After W7: gain 2 on the seeds (W7b), not an arm at gain 4.**
  - *What W7 gave.* It read *both*, by a wide margin, and failed only the falls criterion.
    Its pre-registration then closes the arm at gain 4, and that verdict stands.
  - *Rejected:* reading the falls as a frozen-tracker artifact and opening the arm anyway.
    That would change a criterion after seeing the data.
  - *Rejected:* stopping at W7. The dose (the incumbent at gain 2) kept most of the gain
    and did not fall, and the two seeds at gain 2 cost about 1 h.
  - *Chosen:* W7b. W7's criteria, unchanged, at gain 2 over the same three trackers,
    pre-registered with its in-sample third declared (`skill_space.md` §10.9).
  - *Where it runs.* After W8, which was already waiting, and before W5b and W6. It
    decides whether an RL arm runs, which neither of those does.
- **After W8: a G1 candidate recorded, not adopted; the NAO's miss tested through W6.**
  - *What the rule gives.* The G1's grounded plans meet `incumbents.json`'s adoption rule:
    the target is better by 6.5 SE; pytest, ruff and the chains pass; flat L4 and the
    walk are better.
  - *Why it is not written into `incumbents.json`.* The assignment forbids overwriting the
    original incumbents, and the user decides adoption at review. The candidate goes to
    `transfer/results/candidates.json`, a new file.
  - *Rejected:* making `ground` the G1's default. Old commands would then silently
    reproduce different numbers.
  - *The NAO.* Its loaded feet read 0.012–0.013 `L` because of the sole offset, so a
    grounded plan asks for what its sole point cannot do. Rather than a new NAO run now,
    W8's switch is added to W6's ladder, pre-registered before W6 runs. W6's tracker is
    the one whose sole point rests at 0.

## 4. Experiment log

| When | Commit | What | Result | File |
| --- | --- | --- | --- | --- |
| 2026-11-04 | 3ad68ab | Contact-transition metrics on every L4 result | Section 2.2 | `contact_fidelity_l4.json`, `contact_fidelity_table.json` |
| 2026-11-04 | a463723 | The at-rate verdict recomputed without bias | Section 2.1 | `body_rate_verdicts.json` |
| 2026-11-04 | 03cdd5b | W1 offline: the generator chained at each body's rate, both spaces | Prediction: NAO asked duty −0.022 to −0.030; G1 identical | `chain_at_rate.json` |
| 2026-11-04 | b6c0930 | W1 queued on the NAO's frozen trackers (control re-run, then 3 trackers × 2 skills + S0 walk) | waiting for the machine | `logs/contract/w1_resample.status` |
| 2026-11-04 | b0787fd | W2 offline: integrators and generator variants against the teacher's futures | Hypothesis fails: the blur is learned (light 0.105 at every integrator vs the teacher's 0.032) | `sampler_sharpness.json` |
| 2026-11-04 | b9e7c4f | The family's load-transfer warp (code, tests; unshaped bit-identical to HEAD) | κ = 0.5 raises light contact 0.065 → 0.097 at duty 0.6, impulse kept | `tests/test_procedural_shape.py` |
| 2026-11-04 | 3572bb7 | W3 and W4 pre-registered and queued behind W1 | waiting for W1 | `logs/contract/w3_w4.status` |
| 2026-11-04 | 8dd2368 | W5 robustness probes, queued behind W3/W4 | waiting | `logs/contract/w5_robust.status` |
| 2026-11-04 | d466b6b | Clearance asked vs reached on the family's probes | NAO 0.37–0.64 of asked at every cadence; G1 0.93–1.06 | `clearance_probes.json` |
| 2026-11-04 | 0a8c1a0 | Where the NAO's sole point rests; why its swings are low | Flat foot +0.01006 `L` (G1 −0.00003); a quarter to a half of the clearance gap; swings not short; W6 pre-registered and queued after W5 | `sole_offset.json`, `clearance_mechanism.json`, `logs/contract/w6_sole.status` |
| 2026-11-05 | 2bd4f12 | What each body's swing leg can do in the plans' time | NAO bound 0.72 of the asked lift on the skills (asked above it in 81 % of swings; the knee's 6.40 rad/s binds, and is enforced); tracker 0.51; G1 bound 3.2; "not a body limit" corrected. *Corrected in the next row:* the bound assumes a leg at rest at lift-off and does not bind | `swing_bandwidth.json` |
| 2026-11-05 | 3fe7631 | The swing ceiling's assumptions checked against the captures | At-rest ceiling beaten in 18 % (skills) / 26 % (probes) of NAO swings, the knee at 0.45 / 0.38 of its limit at lift-off; one-step ceiling 1.01 / 1.15 holds (beaten in 1.6 % / 1.7 %); G1 3.2 / 4.1. A thin margin, not a body limit; the shortfall is mostly the controller's. Square grid fixed (range-only 0.314 → 0.318 `L`) | `swing_bandwidth.json` |
| 2026-11-05 | 6a96bb8 | W1 in the simulator: control re-run, then 3 frozen NAO trackers × 2 skills × 3 commands, and S0's walk, under `HUMANOID_RESAMPLE=generator` (10:01–11:43) | Control reproduces (≤ 0.00097). Asked duty −0.029 / −0.024, as predicted; body duty +0.006 / +0.003. Criterion fails (+0.0088 ± 0.0010). W4 not run | `w1_resample.json`, `w1_contact.json`, `body_rate_verdicts_w1.json` |
| 2026-11-05 | cce652f | Swing by swing: lift against touchdown lag | NAO: low swings land early (Spearman +0.38 to +0.52; lowest quarter 0.10–0.13 `T` early); G1 unrelated | `touchdown_lift.json` |
| 2026-11-05 | 2100ac2 | W3 step 1: both frozen incumbents on the transfer probes (11:43–11:48) | Prediction fails on both bodies, with the sign reversed: at warp 0.5 the stance runs longer (G1 +0.039 ± 0.004, NAO +0.019 ± 0.005), by an earlier touchdown. Steps 3–4 and W4 not run; the queue closed at 11:48 | `transfer_probes_*.json`, `logs/contract/w3_w4.status` |
| 2026-11-05 | 8d52199 | W7 pre-registered: the NAO's leg drives measured, then faster drives (`K × 4`, `D × 2`) on the frozen trackers, queued after W5 and before W6 | Drives' `D / K` 0.83–0.92 `T` (G1 0.10); NAO hip delivers 0.34 of its target's excursion (G1 0.60); open-loop replay at `g` = 4: planar rise 0.22 → 0.39 of asked | `swing_drive.json`, `logs/contract/w7_drives.status` |
| 2026-11-05 | 551f797 | W8 pre-registered: where the plans' loaded feet are, and the G1's events against it; the plans' loaded feet put on the ground on both bodies' frozen trackers, queued after W7 and before W6 | Teachers' loaded feet at 0 (p90 0.006 / 0.013 `L`), the G1's plans' at p90 0.018 / 0.029, the biped's at 0; early lift-off with the plan's height before it (Spearman −0.35 rough), late touchdown with it after (+0.49) | `plan_contact.json`, `logs/contract/w8_plan_contact.status` |
| 2026-11-05 | 24f5e10 | W7 in the simulator: `HUMANOID_NAO_LEG_GAIN=4` on 3 frozen NAO trackers × 2 skills × 3 commands and S0's walk, then the incumbent at gain 2 (12:53–14:39); W7b pre-registered and queued after W8 | Reads *both*: lift +0.203 ± 0.003, duty deviation −0.065 ± 0.005, gate C 0 → 10 of 18 rows. Fails no regression on falls (+13, all `rough_b4` fast). RL arm not opened. Gain 2 on the incumbent: no falls, lift 0.620, duty 0.080 | `w7_drives.json`, `logs/contract/w7b_drives.status` |
| 2026-11-05 | (next commit) | W8 in the simulator: `HUMANOID_PLAN_CONTACT=ground` on S's three G1 seeds and the three NAO trackers, L4 on both skills and S0's walk (14:41–17:10); S0's chains, pytest and ruff for the G1 candidate | G1 passes: rough largest −0.027 ± 0.004, flat −0.019 ± 0.001, no falls (seed 42's three gone), walk deviations all smaller. NAO misses: rough duty +0.018 ± 0.001. Chains byte-identical. G1 candidate recorded, `incumbents.json` unchanged | `w8_plan_contact.json`, `candidates.json` |
| 2026-11-05 | f7919b5 | W5: pushes 0.1 / 0.2 `V` and estimate noise ×2 / ×4 on both incumbents and the NAO's generator-space variant (11:49–12:51) | Pushes: no effect. Noise ×2: flat falls 262/294 (G1), 272/304 (NAO); rough 0/64 even at ×4. W5b (`flat_b4`) queued after W8 | `w5_robust.json` |

## 5. Comparative evaluation

### 5.1 W1: the NAO's frozen trackers under the generator-space interface (validated)

`transfer/results/w1_resample.json` (from `w1_compare.py`). The figures are the duty
deviation from the teacher's own grid: the mean over three commands, two skills and both
feet. Deployed path.

| tracker | linear | generator | gate C passed, own grid | falls |
| --- | --- | --- | --- | --- |
| `r1_3498` (incumbent) | 0.124 | 0.132 | 0 → 0 of 6 | 0 → 0 |
| S seed 32 | 0.134 | 0.142 | 0 → 0 | 0 → 0 |
| S seed 33 | 0.138 | 0.149 | 0 → 0 | 0 → 0 |
| paired difference | | +0.0088 ± 0.0010 | | |

- **The pre-registered criterion fails.** The deviation had to fall by more than 2 SE; it
  rose.
- **What did not regress:** speed, falls, period and S0's walk.
- **What did:** stride length, by 0.0015 ± 0.0006.
- **Gate C under the other two references:** at the body's rate, unbiased, 0 → 0; at the
  body's rate as first computed, 12 → 11 of 18 (`body_rate_verdicts_w1.json`).
- **Not adopted.**

### 5.2 W3 step 1: the frozen incumbents on in-family load transfers (validated)

`transfer/results/transfer_probes_{g1_warm_r3_logload_sym,nao_r1_3498_sym_v3}.json`, from
`transfer_probe_effect.py`. Twelve walking schedules, each written with the half-sine and
with warps 0.5 and 2. The figures are warp 0.5 minus the half-sine, left foot, mean ± SE
of the twelve pairs.

| body | stance, body − asked | airborne where light is asked | touchdown lag | predicted |
| --- | --- | --- | --- | --- |
| G1 | +0.039 ± 0.004 | −0.084 ± 0.050 | −0.039 ± 0.003 `T` | negative, positive |
| NAO | +0.019 ± 0.005 | −0.006 ± 0.073 | −0.025 ± 0.006 `T` | negative, positive |

- **The prediction fails on both bodies,** with the stance moving the other way.
- **Steps 3 and 4 (the shaped family's RL arms, about 13 h) are not run,** as
  pre-registered.

### 5.3 W5: pushes and a degraded estimate (instrumentation)

`transfer/results/w5_robust.json` (from `w5_table.py`; forward command; `skill_space.md`
§10.7).

- **Pushes of up to 0.2 `V` change nothing** on either body or skill: no falls, the largest
  gate C deviation within 0.007 of nominal.
- **Estimate noise splits the skills, not the bodies.**
  - *Flat (`flat_dagger2`).* At twice the estimator's precision it falls in nine episodes of
    ten: G1 262/294, NAO 272/304. At four times both bodies walk backwards.
  - *Rough (`rough_b4`).* No falls at four times, on either body.
  - *Where it lies.* The tracker is the same in both rows, so the gap is the generator's.
    `rough_b4` was trained across noise levels, reading the level; `flat_dagger2` at one
    level, reading none.
  - *The follow-up.* W5b runs `flat_b4`, trained as `rough_b4` was, under the same
    perturbations.

### 5.4 W7: faster NAO leg drives on the frozen trackers (validated; body-model candidate)

`transfer/results/w7_drives.json` (from `w7_compare.py`; `skill_space.md` §10.9). Derived
gains → `K × 4`, `D × 2`. Mean over two skills × three commands, deployed path; gate C on
the teacher's own grid.

| tracker | lift (reached / asked) | duty deviation | gate C passed | falls |
| --- | --- | --- | --- | --- |
| `r1_3498` (incumbent) | 0.489 → 0.688 | 0.124 → 0.063 | 0 → 4 of 6 | 0 → 1 |
| S seed 32 | 0.491 → 0.692 | 0.134 → 0.075 | 0 → 3 | 0 → 6 |
| S seed 33 | 0.455 → 0.663 | 0.138 → 0.062 | 0 → 3 | 0 → 6 |
| paired difference | +0.203 ± 0.003 | −0.065 ± 0.005 | | +13 |
| incumbent at `K × 2`, `D × √2` (dose) | 0.620 | 0.080 | 4 of 6 | 0 |

- **Lift and duty both pass, by 76 and 12 SE.** The touchdowns come later
  (+0.011 ± 0.002 `T`), and the feet graze less in swing (−0.034 ± 0.003).
- **No regression fails, on falls.** All 13 are on `rough_b4` at the fast command (1, 6
  and 6 of about 64 episodes). Speed holds at 0.99 or above, and S0's walk passes for all
  three.
- **Not adopted, and the RL arm at gain 4 does not open,** as pre-registered.
- **Why this is a body-model candidate, not a tracker result.** The intervention changes the
  simulated NAO's drives. Every NAO result in this repository, the paper's included, was
  obtained with the derived gains, and none is replaced.
- **W7b** runs the same criteria at gain 2 over the three trackers.

### 5.5 W8: the plans' loaded feet on the ground (validated; G1 candidate)

`transfer/results/w8_plan_contact.json` (from `w8_compare.py`; `skill_space.md` §10.10).
As drawn → grounded, frozen trackers, mean over three commands, deployed path.

| body | rough, largest | flat, largest | duty deviation, both skills | falls |
| --- | --- | --- | --- | --- |
| G1, seeds 41 / 42 / 43 | 0.148 / 0.161 / 0.150 → 0.128 / 0.128 / 0.120 | 0.089 / 0.090 / 0.082 → 0.070 / 0.071 / 0.065 | 0.102 / 0.098 / 0.093 → 0.084 / 0.082 / 0.078 | 3 → 0 |
| G1, paired difference | −0.027 ± 0.004 | −0.019 ± 0.001 | −0.017 ± 0.001 | |
| NAO, `r1_3498` / 32 / 33 | 0.154 / 0.164 / 0.169 → 0.193 / 0.209 / 0.212 | 0.174 / 0.187 / 0.209 → 0.173 / 0.186 / 0.210 | 0.124 / 0.134 / 0.138 → 0.133 / 0.145 / 0.148 | 0 → 0 |
| NAO, paired difference | +0.042 ± 0.002 | −0.000 ± 0.001 | +0.010 ± 0.001 | |

- **G1: the target is met** (rough better by 6.5 SE, flat better instead of worse), with
  no regression.
  - Rough closes 0.027 of the 0.081 to B1 and still passes gate C nowhere. Flat passes at
    all nine rows (eight before).
  - S0's walk: every gate C deviation smaller; speed ratio 1.004–1.009 → 1.064–1.067.
- **NAO: the target is missed.** Rough's duty deviation rises by 0.018 ± 0.001; flat does
  not move.
- **G1 candidate.** It meets `incumbents.json`'s rule (S0's chains byte-identical, pytest
  and ruff pass). It is recorded in `candidates.json`, and `incumbents.json` is not changed.

## 6. Conclusions

*(pending)*

## 7. Recommendation

*(pending)*
