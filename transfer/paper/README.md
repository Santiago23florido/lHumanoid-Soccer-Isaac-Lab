# Results for the paper

Everything the paper's tables and figures need, produced by scripts in this
repository. A number here is quoted with the script and the result file it comes
from. The prose of the paper is not written here; this folder holds what it
reports.

The experiments run from two queues: [`transfer/scripts/paper_queue.sh`](../scripts/paper_queue.sh)
(S, B1) and, after it, [`transfer/scripts/paper_baselines.sh`](../scripts/paper_baselines.sh)
(B2, B3). Each stage skips itself when its results already exist, so either queue
can be resumed after an interruption. [`transfer/scripts/paper_tables.py`](../scripts/paper_tables.py)
rebuilds every table below from the result files, and
[`transfer/scripts/paper_figures.py`](../scripts/paper_figures.py) draws the figures
from the same files into `transfer/paper/figures/` (gitignored; rerun it): the gait
signature against gate C, the decomposition, the log-load seeds, and the baselines
once they exist.

## The claim

A locomotion skill learned on one humanoid (the Unitree G1: Isaac Lab's published
`Isaac-Velocity-Flat-G1-v0` and `Isaac-Velocity-Rough-G1-v0`) runs on another (the
NAO H25, 9× lighter and 2.3× shorter) **zero-shot**. Neither body's estimator or
tracker ever sees the skill's data. The skill crosses as a contract, the short
future of a 27-number dimensionless state `z*`. A new skill costs one training,
on the G1, however many bodies run it: **N + M trainings, not N × M**.

## Experiments

| Id | What | Question it answers | Status |
| --- | --- | --- | --- |
| M | The adopted model (`transfer/results/incumbents.json`) on both bodies, both skills | The headline | done |
| S | Seeds of each body's final training stage, with L4 | How much the headline varies with training | done |
| B1 | A tracker trained on the skill's own plans (N × M), each body | What zero-shot costs, in the same architecture | done |
| B2 | Oracle: the NAO trained on each skill's own task, Isaac Lab's G1 recipe in the NAO's units | What the NAO can do when it sees the task | done; rough (seed 42) under review: it settled on standing on one leg, see B2's note |
| B3 | Joint retargeting: the NAO imitating the G1's joint trajectories on its own clock, three value maps | The classic baseline (the joint-target signal) | implemented (`nao/tasks/nao_imitate`), actions about the reference; queued after B2's second seed |
| T | Decomposition of the transfer error into generator, tracker and estimator | The theory section's measured terms | done (`paper_decomposition.py`) |
| A | Ablations: estimator, sole height, generator, log-space loads, symmetrization, procedural family | What each piece contributes | done from existing runs |
| D | Decoder v2 (physics prior, deep-learning core) against the distilled v1 control | Whether a stronger decoder buys competence | done, not adopted |

Metrics are those of the ladder, `transfer/scripts/zero_shot_ladder.sh`:

- **L4:** the published skill at three commands (forward 0.25 m/s; fast 0.6 m/s; turning 0.3 m/s with 0.3 rad/s, and 0.2 m/s lateral on flat), in the G1's units and Froude-scaled onto the NAO. Each command is 32 environments × 900 control steps after 100 to settle; its episodes are the 32 plus every reset. A command *passes* the teacher's gait signature (gate C, `evaluate_latent.py`) when the speed is at least 0.90 of the teacher's, the fall rate at most 0.02 above it, and the stride length, stride period and both duty factors within 10 %. The deployed path is the autonomous reference with the kinematic estimator.
- **Competence map:** the share of the family's probes a tracker masters, on 61 grid probes and 36 corners.
- **S0's walk:** autonomous reference.

## How each experiment is built

- **S.** Each body's last training stage repeated with other seeds from the same start: on the G1 the log-load continuation of `r3_6194` (seeds 41, 42, 43; 41 is the incumbent), on the NAO the round-1 continuation (31, 32, 33). Every seed is laddered as the incumbents are, symmetrized.
- **B1.** The N × M alternative in the same architecture: each body's incumbent tracker continued for 1000 iterations on one skill's own plans (the skill's generator with that body's encoder; no procedural family), then laddered on that skill. It sees the task; the zero-shot tracker does not.
- **B2, the oracle.** `nao/tasks/nao_velocity`: Isaac Lab's recipe for `Isaac-Velocity-{Flat,Rough}-G1-v0` (rewards, commands, terrain, PPO and iterations) on the NAO. Every dimensioned constant is converted with `UnitSystem.convert_to`, each reward weight by its term's dimension times a time, so a dynamically similar motion earns the same return. The control rate, actions, exploration noise, ground and fall definition are the zero-shot trackers'. The policy observes its base velocity, as the published G1 policies do. It is scored against the G1's teacher like the trackers, through `capture_velocity.py`, which `paper_baselines.sh` first holds to `capture_rollouts.py` on the published rough policy.
  - *The rough oracle stands on one leg.* Seed 42 stepped on both feet at iteration 2000 and then, by iteration 3000, settled on its left leg, the right one raised and the right arm held out, while its air-time reward doubled. Isaac Lab's `feet_air_time_positive_biped` pays single stance every step up to its threshold, so the posture earns that term in full; at a slow command, standing still costs little of the tracking kernel. That is the likely reason, not a measured one. It does the same on its training terrain, so it is the policy, not the plane: on the plane it takes no stride at the forward command (table B). Reproduced by `capture_velocity.py --body nao --task-id NaoVelocityRough-v0` with the run's `model_2999` and, on `NaoVelocityRoughOnFlat-Play-v0`, `model_2000`. Whether it is the recipe on this body or the seed is open.
- **B3, joint retargeting.** `transfer/retarget.py` maps the teacher's joint angles through `correspondence.py`'s names and signs, by each of three value maps (copy; offset about each robot's nominal posture; fraction of range), clamped to the NAO's limits, on the NAO's clock (one NAO step reads the teacher 1.625 steps on). `nao/tasks/nao_imitate` imitates them DeepMimic-style (reference state initialisation; pose 0.65, joint rates 0.1, base velocity 0.1, yaw rate 0.1) on the trackers' body. Its actions are the trackers' offsets of 0.25 rad, about the reference's next row instead of the crouch: the zero action plays the reference back and the policy learns the feedback around it (residual policy learning, Silver et al. 2018; on a biped's reference motion, Xie et al. 2018), so the baseline gets the joint signal in its most favourable form. About the crouch, the first design, the pose reward had no gradient: the retargeted references sit so far from the zero action that its kernel is ~0 there. It trains on the teacher's DART capture and is tested on the teacher's own play capture at each L4 command.
- **T.** `paper_decomposition.py`: for each L4 command, the gait's signed deviation from the teacher split exactly into the generator's part (what the plans asked − teacher) and the tracker's (body − asked), and the estimate's effect (deployed path − true-state path).
  - *Reading the last term on flat.* It is large there (G1 0.19–0.36), and the deficit is the true-state path's, not the deployed one's. On the G1 the true-state path's stride period is 0.60–0.64 `T` against the teacher's 0.98–1.03, with 7–16 falls per command; the deployed path's is 0.95–1.01 `T`, with none. On the NAO the true-state path's is 0.77–0.81 `T`, with 1–6 falls. The plans are not what changes: on both paths they ask for 0.94–0.99 `T` on the G1 and 0.96–1.01 `T` on the NAO. It is the zero-shot tracker that, fed the true state, executes the shorter stride; B1's task-specific tracker, fed the same state, falls on neither body and executes 0.96–1.01 `T` on the G1 (`zero_shot_<body>_<tag>_flat_dagger2_auto_kin_l4_*.json`, runs `encoder` and `truth`, fields `asked` and `signature`).
- **A, D.** Rows read from runs that differ by the one change the row names (`ABLATIONS` in `paper_tables.py`).

## Tables

Regenerated by `paper_tables.py`; until a stage has run, its rows are absent.

<!-- tables:start -->

### M — the adopted model, zero-shot, deployed path

Falls over the episodes of 3 commands × 64, commands that pass the teacher's gait signature, and speed over the teacher's. Source: `zero_shot_<body>_<tag>_<generator>_auto_kin_l4_*.json`, tags `warm_r3_logload_sym`, `r1_3498_sym_v3`.

| Body | Skill | L4 |
| --- | --- | --- |
| G1 | rough | 0/192, passes 0/3, speed 1.00–1.05 |
| G1 | flat | 0/192, passes 3/3, speed 1.06–1.14 |
| NAO | rough | 0/192, passes 0/3, speed 1.00–1.04 |
| NAO | flat | 0/192, passes 0/3, speed 1.04–1.12 |

### S — seeds of each body's final training stage, symmetrized

Walk: S0's check, autonomous reference, falls and speed over the teacher's. Source: `paper_queue.sh` stages `s_g1`, `s_nao`.

| Body | Seed | Rough L4 | Flat L4 | Walk |
| --- | --- | --- | --- | --- |
| G1 | 41 | 0/192, passes 0/3, speed 1.00–1.05 | 0/192, passes 3/3, speed 1.06–1.14 | 0/64, 1.01 |
| G1 | 42 | 3/192, passes 0/3, speed 0.97–1.05 | 0/192, passes 2/3, speed 1.06–1.15 | 0/64, 1.01 |
| G1 | 43 | 0/192, passes 0/3, speed 1.00–1.05 | 0/192, passes 3/3, speed 1.06–1.13 | 0/64, 1.00 |
| NAO | 31 | 0/192, passes 1/3, speed 1.00–1.04 | 0/192, passes 0/3, speed 1.04–1.17 | 0/64, 0.99 |
| NAO | 32 | 0/192, passes 0/3, speed 1.00–1.04 | 0/192, passes 0/3, speed 1.04–1.11 | 0/64, 1.00 |
| NAO | 33 | 0/191, passes 0/3, speed 1.00–1.04 | 0/192, passes 0/3, speed 1.04–1.12 | 0/64, 1.00 |

### B1 — zero-shot against a tracker trained on the skill's own plans (N × M)

Source: `paper_queue.sh` stages `b1_nao`, `b1_g1`.

| Body | Skill | Zero-shot (adopted) | Task-specific tracker |
| --- | --- | --- | --- |
| G1 | rough | 0/192, passes 0/3, speed 1.00–1.05 | 0/192, passes 3/3, speed 1.01–1.05 |
| G1 | flat | 0/192, passes 3/3, speed 1.06–1.14 | 0/192, passes 3/3, speed 1.06–1.13 |
| NAO | rough | 0/192, passes 0/3, speed 1.00–1.04 | 0/192, passes 0/3, speed 1.00–1.04 |
| NAO | flat | 0/192, passes 0/3, speed 1.04–1.12 | 0/192, passes 0/3, speed 1.04–1.13 |

### B2 — zero-shot against the oracle: the NAO trained on the skill's own task

The oracle is Isaac Lab's G1 recipe in the NAO's units (`nao/tasks/nao_velocity`), scored against the G1's teacher like the zero-shot tracker. Its gait is its own: it was rewarded for the task, not the G1's gait. Source: `paper_baselines.sh` stage `b2`.

| Body | Skill | Zero-shot (adopted) | Oracle |
| --- | --- | --- | --- |
| NAO | rough | 0/192, passes 0/3, speed 1.00–1.04 | 0/192, passes 0/3, speed -0.02–0.96 |
| NAO | flat | 0/192, passes 0/3, speed 1.04–1.12 | 0/192, passes 0/3, speed 0.95–1.18 |

### B — the NAO's gait against the teacher's, by method

Mean over the three L4 commands of each gate C feature's |relative deviation| (the gate allows 0.10). A method that fails gate C fails it here or on speed and falls, in the tables above. A command on which the NAO takes no stride has no stride length or period; the cell names it.

| Body | Skill | Method | Stride length | Stride period | Duty, left | Duty, right |
| --- | --- | --- | --- | --- | --- | --- |
| NAO | rough | zero-shot | 0.04 | 0.03 | 0.15 | 0.03 |
| NAO | rough | B1, task tracker | 0.07 | 0.07 | 0.33 | 0.06 |
| NAO | rough | B2, oracle | no stride on forward | no stride on forward | 0.65 | 0.66 |
| NAO | flat | zero-shot | 0.04 | 0.03 | 0.14 | 0.17 |
| NAO | flat | B1, task tracker | 0.19 | 0.13 | 0.67 | 0.47 |
| NAO | flat | B2, oracle | 7.32 | 7.32 | 0.24 | 0.14 |

### T — the gait's deviation from the teacher, decomposed

Mean over the three commands of the |signed relative deviation|. Signed, total = generator + tracker exactly; estimation = deployed path − true-state path. Source: `paper_decomposition.py`, `transfer/paper/data/decomposition.json`.

| Body/skill | Term | speed | stride_length | stride_period | duty_left | duty_right |
| --- | --- | --- | --- | --- | --- | --- |
| g1/rough | total | 0.040 | 0.045 | 0.014 | 0.145 | 0.140 |
| g1/rough | generator | 0.033 | 0.070 | 0.099 | 0.021 | 0.026 |
| g1/rough | tracker | 0.007 | 0.100 | 0.089 | 0.166 | 0.114 |
| g1/rough | estimator | 0.001 | 0.008 | 0.007 | 0.033 | 0.006 |
| g1/flat | total | 0.081 | 0.053 | 0.026 | 0.089 | 0.035 |
| g1/flat | generator | 0.077 | 0.036 | 0.038 | 0.042 | 0.035 |
| g1/flat | tracker | 0.016 | 0.020 | 0.013 | 0.131 | 0.070 |
| g1/flat | estimator | 0.190 | 0.276 | 0.363 | 0.034 | 0.017 |
| nao/rough | total | 0.031 | 0.041 | 0.028 | 0.154 | 0.025 |
| nao/rough | generator | 0.027 | 0.063 | 0.039 | 0.069 | 0.033 |
| nao/rough | tracker | 0.005 | 0.039 | 0.041 | 0.086 | 0.030 |
| nao/rough | estimator | 0.002 | 0.008 | 0.009 | 0.024 | 0.008 |
| nao/flat | total | 0.073 | 0.040 | 0.031 | 0.143 | 0.174 |
| nao/flat | generator | 0.061 | 0.042 | 0.018 | 0.121 | 0.098 |
| nao/flat | tracker | 0.015 | 0.011 | 0.013 | 0.022 | 0.075 |
| nao/flat | estimator | 0.031 | 0.174 | 0.185 | 0.105 | 0.067 |

### A — ablations

Rough L4 on the deployed path, one change at a time, unless the row says otherwise. The tags are in `paper_tables.py`'s `ABLATIONS`.

| Body | Change | Before | After |
| --- | --- | --- | --- |
| G1 | learned encoder → kinematic estimator | 163/278, passes 0/3, speed 0.49–0.88 | 43/187, passes 0/3, speed 0.93–1.00 |
| NAO | learned encoder → kinematic estimator | 0/192, passes 0/3, speed 1.01–1.05 | 0/192, passes 0/3, speed 1.01–1.05 |
| G1 | height at the sole point → the sole's lowest corner | 43/187, passes 0/3, speed 0.93–1.00 | 4/190, passes 0/3, speed 1.02–1.07 |
| G1 | generator B3 → B4 (noise-conditional) | 4/190, passes 0/3, speed 1.02–1.07 | 0/192, passes 0/3, speed 1.00–1.05 |
| NAO | generator B3 → B4 (noise-conditional) | 0/192, passes 0/3, speed 1.01–1.05 | 0/192, passes 0/3, speed 1.00–1.04 |
| G1 | loads scored linearly → in log space (same continuation, seed 41) | 0/192, passes 0/3, speed 1.00–1.05 | 1/192, passes 0/3, speed 1.00–1.05 |
| NAO | loads scored linearly → in log space (same continuation, seed 31) | 1/192, passes 1/3, speed 1.00–1.04 | 0/192, passes 0/3, speed 1.00–1.04 |
| G1 | unsymmetrized → symmetrized deployment | 1/192, passes 0/3, speed 1.00–1.05 | 0/192, passes 0/3, speed 1.00–1.05 |
| G1 | map + corners passed: loads linear → log space, 3 seeds each | 67, 62, 66 | 71, 72, 74 (+7.33 ± 2.67) |
| NAO | map passed: walk plans only → 70 % procedural family | grid 6/57, manoeuvre 0/6, twin 0/3 | grid 33/57, manoeuvre 6/6, twin 1/3 |

### D — decoder v2 at equal competence (G1, warm-started from the same incumbent)

Grid and corners: probes mastered. Source: `warm_study.sh`, `core_and_nao.sh`.

| Decoder | Rough L4 | Grid | Corners |
| --- | --- | --- | --- |
| v1, distilled control | 62/225, passes 0/3, speed 0.69–1.05 | 37/61 | 30/36 |
| v2, physics prior | 0/192, passes 0/3, speed 1.00–1.06 | 36/61 | 26/36 |
| v2, deep-learning core | 0/192, passes 0/3, speed 1.01–1.05 | 39/61 | 30/36 |

<!-- tables:end -->
