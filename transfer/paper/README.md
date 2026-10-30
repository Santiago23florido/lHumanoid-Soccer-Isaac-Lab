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
| B2 | Oracle: the NAO trained on each skill's own task, Isaac Lab's G1 recipe in the NAO's units | What the NAO can do when it sees the task | done, rough with two seeds; rough under review: both seeds settle on standing on one leg, see B2's note |
| B3 | Joint retargeting: the NAO imitating the G1's joint trajectories on its own clock, three value maps | The classic baseline (the joint-target signal) | done, three maps per skill; none passes gate C (B3's note) |
| B2b | The oracle with one change: its air-time term pays no phase held past its threshold | Whether the oracle's failure is the recipe's air-time term on this body | done, both skills, seeds 42 and 43 (B2b's note) |
| B3s | B3's copy map with seeds 43 and 44, both skills | Whether copy's tie with the zero-shot model on rough holds across seeds | queued after B2b |
| SE | Bootstrap standard errors over environments for every L4 comparison of the NAO's methods | Which differences between methods are larger than their sampling error | done for the runs so far (table SE): environments vary far less than seeds, so a comparison needs seeds; B3 copy's waits for B3s |
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
  - *The rough oracle stands on one leg.* Seed 42 stepped on both feet at iteration 2000 and then, by iteration 3000, settled on its left leg, the right one raised and the right arm held out, while its air-time reward doubled. Isaac Lab's `feet_air_time_positive_biped` pays single stance every step up to its threshold, so the posture earns that term in full; at a slow command, standing still costs little of the tracking kernel. That is the likely reason, not a measured one. It does the same on its training terrain, so it is the policy, not the plane: on the plane it takes no stride at the forward command (table B). Reproduced by `capture_velocity.py --body nao --task-id NaoVelocityRough-v0` with the run's `model_2999` and, on `NaoVelocityRoughOnFlat-Play-v0`, `model_2000`. Seed 43 does the same on the other leg: at the forward command its right foot is in contact 99.8 % of the time and its left 3.3 %, at a speed of -0.02 of the teacher's, and it reaches 0.72 at the fast command and 0.30 turning (`oracle_nao_rough_s43_l4_*.json`, run `oracle`, `signature.duty_factor`). Its air-time term rose the same way, from 0.012 at iteration 2000 to 0.030 at 2999, against seed 42's 0.020 to 0.041 (`logs/contract/train_oracle_rough{,_s43}.log`, `Episode_Reward/feet_air_time`). Two seeds make it the recipe on this body, not the seed; the cause is still the hypothesis above. Seed 43 first crashed on runaway actions; it is the run retrained under ddfbb72's action bound, whose Isaac check reproduced seed 42's forward result.
- **B3, joint retargeting.** `transfer/retarget.py` maps the teacher's joint angles through `correspondence.py`'s names and signs, by each of three value maps (copy; offset about each robot's nominal posture; fraction of range), clamped to the NAO's limits, on the NAO's clock (one NAO step reads the teacher 1.625 steps on). `nao/tasks/nao_imitate` imitates them DeepMimic-style (reference state initialisation; pose 0.65, joint rates 0.1, base velocity 0.1, yaw rate 0.1) on the trackers' body. Its actions are the trackers' offsets of 0.25 rad, about the reference's next row instead of the crouch: the zero action plays the reference back and the policy learns the feedback around it (residual policy learning, Silver et al. 2018; on a biped's reference motion, Xie et al. 2018), so the baseline gets the joint signal in its most favourable form. About the crouch, the first design, the pose reward had no gradient: the retargeted references sit so far from the zero action that its kernel is ~0 there. It trains on the teacher's DART capture and is tested on the teacher's own play capture at each L4 command.
  - *What the maps do.* Every map stands on both skills (0/192), and every one steps at the teacher's stride period, by construction: the reference is on the NAO's clock. None passes gate C. On flat, copy and offset reach 0.88–1.02 of the teacher's speed and miss on the split of stance and swing: their duty factors are 0.14–0.48 off, against the zero-shot model's 0.14 and 0.17 mean. On rough, copy is the zero-shot model's equal: speed 1.01–1.06, duty factors 0.12 and 0.14 off on the mean (zero-shot 0.15 and 0.03), and a mean largest deviation of 0.147 against 0.154, inside the spread of the zero-shot model's own seeds (table SE; copy's seeds are B3s); offset is slower (0.87–1.04) and further off (0.15, 0.24). Range walks at 0.38–0.42 of the teacher's speed on flat and 0.41–0.69 on rough. A fraction of range scales each joint's swing by the ratio of the two robots' spans, and for the hip pitch that ratio is 2.02/5.40 = 0.374 (the NAO's URDF; `g1/body/body_card.json`, `pos_limits`). The NAO executes the shorter stride it is given: 0.32–0.62 short, except at flat's turning command (0.08). That the pose term (0.65) outweighs the velocity terms (0.1 each) is the likely reason it follows the joints rather than the speed, not a measured one. Source: `retarget_nao_<skill>_<map>_l4_*.json`, run `imitate`, `verdict`; the largest deviation as `paper_figures.py` reads it.
- **B2b, pre-registered on 2026-10-30, before any of its runs.** B2's oracle settled on long stances: on rough both seeds stand on one leg, and on flat its stride period is 5.6–11.2 `T` against the teacher's 1.0, its mean stance and flight both past the air-time threshold's 1.55 `T` (`oracle_nao_<skill>_l4_*.json`, run `oracle`, `signature`). Isaac Lab's `feet_air_time_positive_biped` pays a single stance held indefinitely its threshold at every step. `NaoVelocity{Flat,Rough}Bounded-v0` is the recipe with one change: the term pays a single stance only while neither foot has held its contact or flight longer than the threshold (`nao/tasks/nao_velocity/rewards.py`, `bounded`). On a gait whose every phase is shorter, as the teacher's are on average (its longest mean stance or flight is 0.62 `T`; duty factor times stride period in the same files, run `teacher`), it is the original term, step for step. The protocol is fixed now: both skills, seeds 42 and 43, the same iterations, scored as B2 is; the result is reported as it comes, with no further change to the recipe. It tests the hypothesis that the air-time term is why the oracle does not walk the teacher's way; it does not make the oracle walk the teacher's gait, which nothing in its reward asks for. Source: `ORACLE_VARIANT=bounded SEED=<seed> paper_baselines.sh b2`.
  - *What seed 42 did.* On flat the bounded oracle shortens its phases to the threshold, not to the teacher's: its stride period is 3.3–3.5 `T` (the original's 5.6–11.2, the teacher's 0.98–1.03) and its mean stance and flight 1.58–1.78 `T`, just past the 1.55 `T` where the term stops paying. It falls at no command, walks at 1.22–1.23 of the teacher's speed, and its duty factors are within 10 % at forward and turning (0.05–0.08 off; fast's left 0.12); its stride length and period are 2.2–3.1 off. On rough it steps on both feet at fast and turning, duty factors 0.51/0.56 and 0.59/0.48 against the teacher's 0.48/0.50 and 0.55/0.58 (the original's 0.72/0.34 and 0.89/0.20), at 0.88 and 0.54 of the teacher's speed, stride period 3.3 and 4.5 `T`. At forward it still stands mostly on its right foot: duty factors 0.13/0.90, stride period 12.1 `T`, speed 0.33, where the original took no stride at all. No falls on rough either. Its air-time reward ended training at 0.022, against the original seeds' 0.041 and 0.030 (`Episode_Reward/feet_air_time`, iteration 2999). Source: `oracle_nao_<skill>_bounded_l4_*.json`, run `oracle`, `signature` and `verdict`; phases are duty factor times stride period.
  - *What seed 43 did.* On flat, the same: stride period 3.38–3.44 `T`, mean stance and flight 1.62–1.79 `T`, 1.15–1.18 of the teacher's speed, no falls, stride length and period 2.3–2.9 off, duty factors at most 0.11 off. On rough it walks on both feet at fast, duty factors 0.59/0.52, speed 0.86, stride period 3.4 `T`. At forward it stands still on both feet (duty factors 0.99/0.99, speed -0.03), where the original seed 43 stood on its right leg (0.03/1.00); at turning it shuffles in long double support (0.82/0.93, stride period 2.1 `T`, speed 0.23). No falls. Its air-time reward ended training at 0.014, the original seed 43's at 0.030. Source: `oracle_nao_<skill>_bounded_s43_l4_*.json`.
  - *Both seeds, read against the hypothesis.* Bounding the term changes the oracle's rhythm on flat, the same way with both seeds, to the threshold's and not the teacher's. On rough, at fast, both seeds' duty factors are balanced (0.51/0.56, 0.59/0.52; the teacher's 0.48/0.50), where the original seeds' were lopsided (0.72/0.34, 0.38/0.69); at forward neither walks, one standing mostly on one foot and the other on both. On these two seeds the bound sets the length of the oracle's phases on flat and evens its stance at fast on rough; it does not make the rough oracle walk at forward.
- **B3s.** B3's copy map, the one level with the zero-shot model on rough, trained again with seeds 43 and 44 on both skills, everything else as B3. Source: `SEED=<seed> RETARGET_MAPS=copy paper_baselines.sh b3`.
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

The oracle is Isaac Lab's G1 recipe in the NAO's units (`nao/tasks/nao_velocity`), scored against the G1's teacher like the zero-shot tracker. Its gait is its own: it was rewarded for the task, not the G1's gait. B2b is the recipe with its air-time term bounded (`rewards.py`, `bounded`). One column per training seed. Source: `paper_baselines.sh` stage `b2`, with `SEED=<seed>` past the first and `ORACLE_VARIANT=bounded` for B2b.

| Body | Skill | Zero-shot (adopted) | Oracle, seed 42 | Oracle, seed 43 | B2b, seed 42 | B2b, seed 43 |
| --- | --- | --- | --- | --- | --- | --- |
| NAO | rough | 0/192, passes 0/3, speed 1.00–1.04 | 0/192, passes 0/3, speed -0.02–0.96 | 0/192, passes 0/3, speed -0.02–0.72 | 0/192, passes 0/3, speed 0.33–0.88 | 0/192, passes 0/3, speed -0.03–0.86 |
| NAO | flat | 0/192, passes 0/3, speed 1.04–1.12 | 0/192, passes 0/3, speed 0.95–1.18 | — | 0/192, passes 0/3, speed 1.22–1.23 | 0/192, passes 0/3, speed 1.15–1.18 |

### B3 — zero-shot against joint retargeting: the NAO imitating the G1's joints

The teacher's joint trajectories through the correspondence, on the NAO's clock, imitated DeepMimic-style (`nao/tasks/nao_imitate`); one column per value map (`transfer/retarget.py`). At test time each imitates the teacher's own play capture at the command. Source: `paper_baselines.sh` stage `b3`.

| Body | Skill | Zero-shot (adopted) | Retarget, copy | Retarget, offset | Retarget, range |
| --- | --- | --- | --- | --- | --- |
| NAO | rough | 0/192, passes 0/3, speed 1.00–1.04 | 0/192, passes 0/3, speed 1.01–1.06 | 0/192, passes 0/3, speed 0.87–1.04 | 0/192, passes 0/3, speed 0.41–0.69 |
| NAO | flat | 0/192, passes 0/3, speed 1.04–1.12 | 0/192, passes 0/3, speed 0.88–1.02 | 0/192, passes 0/3, speed 0.89–1.00 | 0/192, passes 0/3, speed 0.38–0.42 |

### B — the NAO's gait against the teacher's, by method

Mean over the three L4 commands of each gate C feature's |relative deviation| (the gate allows 0.10). A method that fails gate C fails it here or on speed and falls, in the tables above. A command on which the NAO takes no stride has no stride length or period; the cell names it.

| Body | Skill | Method | Stride length | Stride period | Duty, left | Duty, right |
| --- | --- | --- | --- | --- | --- | --- |
| NAO | rough | zero-shot | 0.04 | 0.03 | 0.15 | 0.03 |
| NAO | rough | B1, task tracker | 0.07 | 0.07 | 0.33 | 0.06 |
| NAO | rough | B2, oracle, seed 42 | no stride on forward | no stride on forward | 0.65 | 0.66 |
| NAO | rough | B2, oracle, seed 43 | 1.12 | 2.84 | 0.60 | 0.50 |
| NAO | rough | B2b, bounded, seed 42 | 1.88 | 5.34 | 0.30 | 0.26 |
| NAO | rough | B2b, bounded, seed 43 | 0.82 | 2.25 | 0.51 | 0.42 |
| NAO | rough | B3, copy | 0.02 | 0.02 | 0.12 | 0.14 |
| NAO | rough | B3, offset | 0.10 | 0.00 | 0.15 | 0.24 |
| NAO | rough | B3, range | 0.46 | 0.00 | 0.12 | 0.13 |
| NAO | flat | zero-shot | 0.04 | 0.03 | 0.14 | 0.17 |
| NAO | flat | B1, task tracker | 0.19 | 0.13 | 0.67 | 0.47 |
| NAO | flat | B2, oracle, seed 42 | 7.32 | 7.32 | 0.24 | 0.14 |
| NAO | flat | B2b, bounded, seed 42 | 2.95 | 2.41 | 0.10 | 0.04 |
| NAO | flat | B2b, bounded, seed 43 | 2.80 | 2.41 | 0.10 | 0.03 |
| NAO | flat | B3, copy | 0.10 | 0.00 | 0.33 | 0.21 |
| NAO | flat | B3, offset | 0.04 | 0.00 | 0.38 | 0.21 |
| NAO | flat | B3, range | 0.43 | 0.00 | 0.21 | 0.07 |

### SE — the NAO's methods against the zero-shot model, by seed

Each command's largest gate C deviation, averaged over the three commands, as the figure reads it. A method trained with several seeds: the mean over its seeds ± their standard deviation over the root of their number, and its difference from the zero-shot model's seeds (S's; the adopted model is not among them) ± both errors combined. A method trained with one seed has only the bootstrap error of its 32 environments, far smaller than the spread between seeds (compare the zero-shot model's rows): it can only be read against the range of the zero-shot model's seeds. Source: `paper_bootstrap.py`, 1000 draws, `transfer/paper/data/bootstrap.json`.

| Body | Skill | Method | Seeds | Largest deviation | Minus zero-shot |
| --- | --- | --- | --- | --- | --- |
| NAO | rough | zero-shot, S's seeds | 3 | 0.149 ± 0.017 (seeds 0.115–0.169) | — |
| NAO | rough | B2, oracle | 2 | 2.651 ± 0.319 (seeds 2.332–2.970) | +2.502 ± 0.319 |
| NAO | rough | B2b, bounded | 2 | 3.798 ± 1.545 (seeds 2.253–5.343) | +3.648 ± 1.545 |
| NAO | rough | zero-shot, adopted | 1 | 0.154 ± 0.003 (environments) | one seed |
| NAO | rough | B1, task tracker | 1 | 0.326 ± 0.002 (environments) | one seed |
| NAO | rough | B3, copy | 1 | 0.147 ± 0.002 (environments) | one seed |
| NAO | rough | B3, offset | 1 | 0.238 ± 0.001 (environments) | one seed |
| NAO | rough | B3, range | 1 | 0.464 ± 0.001 (environments) | one seed |
| NAO | flat | zero-shot, S's seeds | 3 | 0.186 ± 0.014 (seeds 0.162–0.209) | — |
| NAO | flat | B2b, bounded | 2 | 2.874 ± 0.078 (seeds 2.797–2.952) | +2.688 ± 0.079 |
| NAO | flat | zero-shot, adopted | 1 | 0.174 ± 0.002 (environments) | one seed |
| NAO | flat | B1, task tracker | 1 | 0.674 ± 0.003 (environments) | one seed |
| NAO | flat | B2, oracle, seed 42 | 1 | 7.618 ± 0.003 (environments) | one seed |
| NAO | flat | B3, copy | 1 | 0.325 ± 0.001 (environments) | one seed |
| NAO | flat | B3, offset | 1 | 0.376 ± 0.001 (environments) | one seed |
| NAO | flat | B3, range | 1 | 0.436 ± 0.001 (environments) | one seed |

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
