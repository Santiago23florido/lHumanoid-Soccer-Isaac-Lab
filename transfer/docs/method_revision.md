# Method revision: a contact-consistent contract

This document proposes a change to how the contract `z*` is defined, and reads twelve
reference papers for what applies. It builds on the contact-fidelity study
([`research_report.md`](research_report.md)). Its first test, W9, is pre-registered in
[`skill_space.md`](skill_space.md) §10.11. Each claim is marked **measured** (computed from
existing captures; the script is named), **validated** (a pre-registered simulator test,
read from its file) or **hypothesis**. The papers are in `references/`, which is
gitignored; they are cited by name.

## 1. The problem at the level of the definition

The study left two failures open:
- the G1's rough deficit, about two thirds of the gap to B1;
- the NAO's short, low swing, which training on faster drives does not fix (W7c).

Both concern contact. The contract describes each foot by three independent continuous
fields:
- a load share `ℓ` (`foot_load`, in `m g`);
- a sole position from the CoM, whose vertical part with `height` gives the sole height
  `h`;
- a sole velocity `v` (`foot_velocity`, world velocity in the heading frame, in `V`).

**The physics couples them; the contract does not.** On rigid ground with friction:
- a foot carries load only on the ground: `0 ≤ ℓ ⊥ h ≥ 0`;
- a loaded foot sticks: `ℓ > 0 ⇒ v = 0`, within the friction cone;
- a stance foot keeps its world position for the whole stance.

The teachers' data obey all three. Nothing in the generator does: it models the 27 fields
of a 20-sample future jointly, but the couplings hold only as well as the flow happens to
learn them. The tracker then scores each field separately against the plan. Where the
plan breaks a coupling, it asks for a state no body can reach, and every field the body
gets right still pays for the one it cannot.

## 2. What the plans do (measured)

`transfer/results/plan_slip.json` (from `plan_slip.py`) gives the loaded feet's motion, as
medians in `V`:

| Loaded foot (load above the contact threshold) | Horizontal speed | Firm load (above 25 % of weight) | Sole height, p90 (`L`) |
| --- | --- | --- | --- |
| Teachers: flat / rough / walk | 0.0020 / 0.0022 / 0.0017 | 0.0019 / 0.0019 / 0.0016 | 0.006 / 0.013 / 0.009 |
| Procedural plans (what the trackers practise, 70 %) | 0.0000 | 0.0000 | 0.000 |
| Walk contract's plans (the other 30 %), G1 / NAO | 0.0135 / 0.0090 | 0.0118 / 0.0081 | 0.010 / 0.014 |
| Skill plans on the G1: flat / rough | 0.111 / 0.125 | 0.104 / 0.115 | 0.018 / 0.029 |
| Skill plans on the NAO: flat / rough | 0.097 / 0.108 | 0.088 / 0.098 | 0.023 / 0.031 |
| G1 bodies: flat / rough | 0.004 / 0.006 | 0.004 / 0.005 | 0.004 / 0.015 |
| NAO bodies: flat / rough | 0.042 / 0.046 | 0.040 / 0.043 | 0.023 / 0.030 |

- *The skill plans slide their stance feet* at about fifty times the teacher's speed,
  under firm load as well.
- *The trackers never practised this.* The family holds a loaded foot exactly still, and
  the walk contract's plans move it at a tenth of the skills' speed.
- *The G1's body does not follow the slide.* The NAO's partly does: its stance foot drags
  at 0.04 `V` on the skills, against 0.016 on the walk.
- *W8 fixed the height half alone.* Grounded plans keep their slide unchanged (0.111 /
  0.126 on the G1). Height alone moved rough's largest deviation by −0.027 ± 0.004 and
  flat's by −0.019 ± 0.001 on the G1 (validated, `w8_plan_contact.json`).

**The slide is the generator's own** (`transfer/results/stick_offline.json`, from
`stick_offline.py`). Plans drawn from the teachers' own play histories, with no body in
the loop, slide just the same: 0.114 (flat) and 0.124 (rough). Further:
- *It is systematic, not a bad draw.* The most consistent of eight draws per window still
  slides at 0.100 and 0.110.
- *The positions drift too.* A loaded foot's world position, from the plan's own CoM
  velocity and foot positions, moves a median 0.018 (flat) and 0.023 (rough) `L` per
  stance, against the teacher's 0.0035 and 0.0077: three to five times the teacher's
  drift. Zeroing the velocity does not touch it.
- *Closeness to the realised future is not the argument.* Against the future the teacher
  actually walked from each window, zeroing the stance velocity lowers the foot-velocity
  error only 3–4 % (2.39 → 2.31 flat, 2.68 → 2.57 rough, normalised units). Grounding
  raises the foot-height error (2.19 → 2.30, 2.76 → 2.94), because the plan's stance timing
  differs from the teacher's. W8 helped in the simulator anyway. What the change restores
  is the plan's internal consistency, the property the procedural family has and the
  tracker learned under.

## 3. What the references contribute

| Paper | Idea | For this method | Taken |
| --- | --- | --- | --- |
| DexGraspFlow (2026) | Link-level SE(3) flow matching, a topology bias, training-free steering of the flow during integration. Limitation it names: constraints are left to a post-hoc IK; future work is to put them inside the integration | Our grounding, like their IK, corrects after sampling. Steering the flow toward the contact manifold keeps the other fields consistent with the corrected ones | **Yes** (realisation 2) |
| SERNF (2026) | Exact-likelihood normalizing-flow policy; a chunk-level critic; best-of-N chunk selection | Best-of-N on the contact residual is body-blind and stays on the generator's manifold. Offline, it barely lowers the slide (§2), so it is weak here. Chunk-level credit supports scoring contact events per committed window | Partly (§5, objective) |
| GOTT (2026) | Contact acquisition as its own closed-loop primitive with a hysteresis contact predictor; a canonical masked representation; training across several hands transfers zero-shot (72.1 % against an oracle's 72.2 %) | Contact transitions are a first-class mode change, not a by-product of tracking a trajectory | Yes, as the hybrid view (§4) |
| XHugWBC (ICML 2026) | One policy for 12 simulated and 7 real humanoids; physics-consistent morphological randomization (pseudo-inertia Cholesky); gait phase and stance in the command | An explicit contact schedule in the interface. Randomization over drive parameters is the alternative to identifying the NAO's drives exactly | The first as support; the second deferred |
| CrossBFM (2026) | A shared latent behavior space distilled from one robot's forward-backward model; PPO trackers with feet-slip and contact-force terms | Its encoder aligns robots through each robot's retargeted motion corpus. A corpus without the transferred gaits would be allowed (correction below), but its latent comes from one source robot's behaviours. Unseen-robot transfer recovers 12.8–89 %, best for similar morphologies | No, as a latent (§3 below); its multi-robot training is noted in §6 |
| LAD (2025) | Contrastive alignment of embodiment-specific action encoders; diffusion in the shared latent | Its policy learns from demonstrations of the task on every embodiment, the target's included. On the NAO that is the task's data in the target's learner, which the rule forbids | No |
| LAC-WM (ICML 2026) | Latent actions from an inverse-dynamics model, effect-based, shared across embodiments | Supports defining the interface by effect in a shared space, as `z*` already is | Support only |
| DexJoCo-X (2026) | Controlled comparison: function-aligned action slots 47.7 %, native 47.0 %, a learned latent 33.1 % | An explicit, function-aligned interface beats a learned latent | Support: keep `z*` explicit |
| CrossDex (ICLR 2025) | A universal action space from human eigengrasps, retargeted per hand; embodiment randomization helps unseen hands | Our analogue is the dimensionless `z*` with a learned per-body tracker. Nothing new for fidelity on two known bodies | No |
| Mask2Real-WM (2026) | Dynamics in an intermediate space (masks) trained in simulation, rendering apart; simulation data helps the split model and hurts the monolithic one | Supports the split between generator and tracker | Support only |
| Proprioceptive Transformer (2026) | History Transformer with an auxiliary reconstruction of hidden state; joint-side sensing beats motor-side | A better load and contact estimator, and a reason to measure the NAO's joint response at the joint | Deferred (estimator) |
| ORCA (IROS 2025) | Hardware; a protocol for joint tracking accuracy and latency (sine and step commands) | A protocol for identifying the real NAO's drives (`research_report.md` §7, item 2) | Deferred (identification) |

**What zero-shot excludes here.** The rule is `skill_space.md` §9's: no gradient step of a
body's encoder or tracker uses a task's data. It does not limit what the generator learns
from, and it does not forbid data from the target body on other motion. The NAO's tracker
already trains on the procedural family and the walk, and its loads are learned on its own
practice.
- *Allowed.* A generator trained on several source robots' demonstrations of the task,
  each in `z*`. It stays zero-shot for any robot not among them.
- *Not allowed.* The task's trajectories reaching the target's encoder or tracker: its own
  demonstrations, or the teacher's retargeted to it (B1 and B2 in
  `transfer/paper/README.md`).
- *Whether an expert exists does not decide it.* What decides it is what the target's
  learners see.

**What the learned-latent papers say together.**
- *What they need.* LAD and LAC-WM learn from task data on the target (demonstrations, or
  fine-tuning on the new robot). CrossBFM aligns robots through each one's retargeted motion
  corpus.
- *Why the revision still keeps `z*` explicit.* DexJoCo-X's controlled comparison favours
  explicit, function-aligned coordinates (47.7 % against 33.1 % for a learned latent). This
  project's own history agrees: a learned action code leaked the body (`skill_space.md`
  §4–6). `z*`'s alignment is analytic (dimensional analysis) and needs no correspondence
  data.
- *What changes.* What `z*` is allowed to say about contact.

*Correction (2026-11-06).* The first version of this section rejected all three because
"they need data from the target body, which the zero-shot rule forbids", and CrossBFM's row
said it "violates the zero-shot rule". That misstated the rule: data from the target is
allowed except the task's, and the rule says nothing about the generator's sources. It
affected the reasons given for rejecting LAD, LAC-WM and CrossBFM, here and in the discussion
that followed, and it omitted multi-source generators from §6. No run or result depended on it.

## 4. The modification

**Define the plan space as the contact-consistent manifold `M`.**
- *Modes.* A foot is in stance where `ℓ > threshold`, otherwise in swing.
- *Stance:* `h = 0`, `v = 0`, and a world position fixed from touchdown to lift-off.
- *Swing:* `ℓ = 0` and `h ≥ 0`.
- *The rule:* every plan the tracker is scored against lies on `M`. The procedural family
  already does by construction, and the teachers' data do to within their measurement.

This is a hybrid definition: a discrete contact mode per foot, and a continuous state on
the mode's manifold. It defines the task's plan more precisely. It does not prescribe how a
body moves: a plan on `M` asks for what a load already implies.

**Three realisations, cheapest first.** Each is run only if the previous one passes.
1. **Projection at deployment** (`HUMANOID_PLAN_CONTACT=stick`;
   `latent/plan_contact.stick_loaded_feet`).
   - *What it does.* It grounds every loaded foot and sets its velocity to zero. It is
     body-blind and needs no training, and the default stays `as_drawn`.
   - *What it does not do.* It leaves the stance drift of the position (§2), which needs
     the plan's positions rewritten.
   - *Prediction.* It improves on W8's height-only grounding on the G1 (W9).
2. **Constraint-guided sampling** (from DexGraspFlow).
   - *What it does.* Each flow step adds `−λ(t) ∇ ½‖C(z)‖²` to the velocity, where `C`
     holds `h · s(ℓ)`, `v · s(ℓ)` and the stance drift, with `s` a smooth stance indicator.
     The network then re-predicts every other field given the corrected ones.
   - *Prediction.* At equal consistency it is at least as good as projection.
   - *What would falsify it.* Guidance helps no more than projection: then the inconsistency
     is local to the contact fields.
3. **A hybrid parameterisation of `z*`** (the full change).
   - *Per foot, a signed contact coordinate `σ`.* `ℓ = max(σ, 0)` and `h = κ max(−σ, 0)`,
     with `κ` a dimensionless scale. The stance foot is expressed by its anchor: its world
     position at touchdown, its velocity identically zero.
   - *What follows.* Complementarity holds by construction. Contact events are the
     zero-crossings of `σ`, and the duty factor is the share of time with `σ > 0`: the
     quantity gate C reads becomes a direct function of one coordinate, not a threshold on
     a blurred load.
   - *What it costs.* It retrains every generator and tracker, and the leakage and symmetry
     tests of `z*` are rerun. It gets its own pre-registration after 1 and 2.

**Predictions this makes about the open failures** (hypotheses).
- *The G1 on rough.* The tracker lifts off early where the plan holds a loaded foot high
  (W8 measured it). If it also lifts where the plan moves a loaded foot, a still stance
  should remove more of the early lift-off.
- *The NAO.* Its stance foot follows part of the slide (0.04 `V`). A plan that holds the
  stance still should reduce the drag. Whether the swing then grows is not predicted: W6
  and W7c say the swing's shortfall is the tracker's objective, not the stance's.

## 5. On the objective side (recorded, not run)

The plan defines the walking task, so scoring the body against the plan's own contact
events is tracking the task. Examples are a likelihood of the lift-off and touchdown
times within each committed window (SERNF's chunk-level credit; GOTT's contact stage).
Rewards that prescribe a gait, such as feet air time, clearance bonuses or the usual
legged-locomotion contact terms, are not allowed: the project forbids a condition on *how* the
walking task is done (`plan.md`). CrossBFM's feet-slip and contact-force terms are on that side of the
line and are not taken. With plans on `M` the line is easy to keep: "do not slide a loaded
foot" is then part of what the plan asks.

## 6. The protocol agreed on 2026-11-06: training may see several bodies, inference is zero-shot

The target design: a generator, with a learned latent beside `z*`, trained on
about five base tasks, whose training may see more than one body. What must hold is
zero-shot at inference.

**Training may use:**
- the base tasks as the G1's experts perform them (experts exist only on the G1);
- the base tasks as other bodies execute them, each through its own decoder, which is
  trained once and is the same for every task. The NAO may be one of these bodies;
- the procedural family and the walk, as the decoders already do.

**Inference on a new task `T` must have:**
- no gradient step anywhere (generator, latent, decoder, estimator) on data of `T` executed
  by the target body;
- the target's decoder as trained once, unchanged;
- the generator learning `T` only from the G1's expert, or not at all if `T` is held out
  (below).

**The two claims, and their tests.**
- *Task zero-shot on a seen body.* Leave one base task out: train on four, evaluate on the
  fifth. Measures:
  - offline, calibration against the held-out teacher and contact consistency (§2's
    metrics);
  - in the simulator, L4 on both bodies with their frozen decoders.

  What task zero-shot costs is the gap to a generator that saw the task. A base task the
  target executed in training is not a zero-shot result on that target.
- *Body zero-shot on an unseen body.* Leave one body out of the generator's training. It
  needs a third body and is not yet possible here.

**The latent's condition.** A latent beside `z*` must carry no identity of the body. Its
leakage test is pre-registered with it: how well a classifier, or a regression onto the
body's joints, recovers the body from the latent, against a shuffled control. That is the
failure of `ζ` and F5 (`research_report.md` §1.2). With several bodies in training, the
test can be run on every one of them.

**Open.** About five base tasks need G1 experts beyond the two published
ones (flat, rough) and the walk. `skill_space.md` §9 limits teachers to Isaac Lab's
published G1 policies, so where the others come from is an open decision.

## 7. Order of work

1. **W9** (`skill_space.md` §10.11). Realisation 1 on the frozen trackers of both bodies,
   paired against W8's grounded captures. About 2.5 h of simulator time, no training.
2. If W9 passes: realisation 2, pre-registered the same way.
3. If 1 or 2 shows the mechanism: realisation 3, with its own budget and its own
   approval.
4. **The multi-task generator with a learned latent** (§6), after the consistency work:
   leave-one-task-out first. Its base tasks wait as decided on where the G1's
   extra experts come from.
5. **Open, not yet proposed: a multi-source generator.** It would be trained on the task
   as several source humanoids perform it, all in `z*`, the target excluded. It stays
   zero-shot. GOTT (72.1 % on unseen variants from three training hands, against 17.2 %
   from one) and XHugWBC suggest diversity of sources helps an unseen body. Two things
   hold it back:
   - *a second teacher.* `skill_space.md` §9 limits pretrained teachers to Isaac Lab's
     published G1 policies, so another source needs that rule revisited first;
   - *what it does not fix.* It does not address the inconsistency of §2, which is how the
     generator models contact, not which bodies it learned from.
6. Not proposed: a learned latent on its own in place of `z*` (§3), multi-morphology tracker training (not this
   project's failure mode), and the estimator Transformer (robustness, not fidelity).
