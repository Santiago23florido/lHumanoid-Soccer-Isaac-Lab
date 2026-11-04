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
2. **The interface lengthens the NAO's asked stances.** The same linear blend as 2.1, in
   `plan_interpolation` and `TimeGrid.interpolation`: the NAO is asked 0.022–0.030 more
   duty than the generator plans (predicted by `chain_at_rate.py`, whose linear chain
   reproduces every recorded capture's asked duty within 0.007).
3. **The trackers make contact binary.**
   - *G1, zero-shot.* Where light contact is asked, the foot is in the air 0.47–0.53 of
     the time. The left foot lifts 0.067 `T` early and lands 0.058 `T` late, so the
     stance is 0.06–0.09 short of asked on rough.
   - *G1, B1* (task-trained): airborne 0.30–0.35, and the duty is right.
   - *The family.* It asks for light contact 5.5 % of the time.
   - *Real and procedural plans are treated differently.* On procedural plans the
     zero-shot stance runs 0.02 *longer* than asked; on rough's plans 0.09 shorter.
4. **The NAO lifts its feet half as high as asked.**
   - *Clearance.* Plans ask 0.07–0.09 `L`; the body gives 0.03–0.05; B1-NAO gives 0.01,
     with grazes in half of every planned swing.
   - *Hypotheses.* Joint-speed feasibility in body time: the NAO's hip pitch is
     1.02 rad/`T` against the G1's 8.27. Or a reward local optimum. Not yet tested.

### 2.3 Unresolved

- Why the trackers lift early at light loads: family coverage (hypothesis H3) or reward
  resolution.
- Whether the NAO's clearance is physically feasible at the teacher's cadence.
- Whether B1-NAO's shuffle is the interpolated plans' doing (W4).
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

## 4. Experiment log

| When | Commit | What | Result | File |
| --- | --- | --- | --- | --- |
| 2026-11-04 | 3ad68ab | Contact-transition metrics on every L4 result | Section 2.2 | `contact_fidelity_l4.json`, `contact_fidelity_table.json` |
| 2026-11-04 | a463723 | The at-rate verdict recomputed without bias | Section 2.1 | `body_rate_verdicts.json` |
| 2026-11-04 | 03cdd5b | W1 offline: the generator chained at each body's rate, both spaces | Prediction: NAO asked duty −0.022 to −0.030; G1 identical | `chain_at_rate.json` |
| 2026-11-04 | b6c0930 | W1 queued on the NAO's frozen trackers (control re-run, then 3 trackers × 2 skills + S0 walk) | waiting for the machine | `logs/contract/w1_resample.status` |
| 2026-11-04 | (this commit) | W2 offline: integrators and generator variants against the teacher's futures | Hypothesis fails: the blur is learned (light 0.105 at every integrator vs the teacher's 0.032) | `sampler_sharpness.json` |

## 5. Comparative evaluation

*(to be filled as the pre-registered runs complete)*

## 6. Conclusions

*(pending)*

## 7. Recommendation

*(pending)*
