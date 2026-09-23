# Humanoid Soccer Isaac Lab

Isaac Lab research project on **cross-embodiment humanoid skill transfer**.

A skill is learned on a robot capable enough to learn it — a 29-DOF Unitree G1 —
and the research question is what has to happen for that skill to end up on a
robot that is not: the SoftBank Robotics / Aldebaran **NAO H25 V5.0**, which is
limited for physical rather than algorithmic reasons.

The balance work below is the foundation that question rests on. It is what
established, for the NAO specifically, *what is physically possible* — and
therefore which parts of a teacher's behaviour are worth imitating and which
are asking for something no controller on that body could do.

<p align="center">
  <img src="docs/img/nao_stand.png" alt="NAO H25 V5.0 in its nominal standing posture in Isaac Sim" width="640">
</p>
<p align="center">
  <sub>NAO H25 V5.0 in Isaac Sim, held in the nominal posture every controller
  in this repository is referenced to: hips −0.35 rad, knees +0.70, ankles
  −0.35, giving a vertical torso and flat soles with the centre of mass 0.2689 m
  above the ground. Reproduce with <code>scripts/render_nao.py</code>.</sub>
</p>

---

## Research status

The current problem is **zero-step balance under perturbation**: hold the
standing posture against external pushes *without changing the contact
configuration*. A foot that lifts or slides more than 50 mm ends the episode.

That constraint is what makes the problem well posed. Without it the robot
solves the task by stepping, which is a different problem governed by different
theory, and the zero-step capturability bound the reward is built on stops
applying.

### Headline result

Four controllers, one environment instance, one protocol: pushes at 90 % of the
direction-dependent capturability bound, **n = 2048** environments, 600 control
steps.

| Controller | Fall-free | 95 % CI | Ankle torque | Saturation |
| --- | --- | --- | --- | --- |
| **PPO policy** | **82.3 %** | [80.6, 84.0] | 44.0 % | 4.2 % |
| PPO, arms frozen | 83.0 % | [81.4, 84.6] | 47.9 % | 4.9 % |
| Joint PD | 77.6 % | [75.8, 79.4] | 34.3 % | 2.6 % |
| Capture-point PD | 27.8 % | [25.9, 29.7] | 30.2 % | 3.6 % |

| Contrast | Δ | SE | z | Verdict |
| --- | --- | --- | --- | --- |
| PPO − joint PD | +4.7 | 1.25 | 3.76 | **significant** |
| Arms frozen − PPO | +0.7 | 1.18 | 0.59 | not significant |
| Joint PD − capture-point PD | +49.8 | 1.35 | 36.8 | **significant** |

Raw data: [`docs/results/comparison_directional.json`](docs/results/comparison_directional.json).

### What has been established

1. **The learned policy beats an analytically derived joint PD by 4.7 points**
   (z = 3.76). The mechanism is visible in the torque column: the policy uses
   44.0 % of the ankle budget against the PD's 34.3 %. A joint PD has no
   representation of its centre of pressure, so it never *tries* to drive it to
   the edge of the foot; the policy exploits authority the capturability bound
   permits and the PD leaves unused.

2. **The arms contribute nothing measurable** (z = 0.59). This contradicts the
   usual explanation for why learned controllers beat model-based ones, which
   appeals to angular-momentum strategies of the arms and trunk. Under this
   zero-step constraint, that contribution is indistinguishable from zero.

3. **Zero-step capturability is anisotropic by a factor of 1.87** on this robot
   — 0.443 m/s backward against 0.827 m/s diagonal — because the support polygon
   is asymmetric (103 mm ahead of the ankle axis, 61 mm behind) and the centre
   of mass sits 12.6 mm forward of it. Perturbation curricula are scaled by
   `ω₀·d(θ)` rather than by an absolute speed for this reason.

4. **The inertia that sizes the leg gains is the whole body about each joint
   axis, not the distal subtree** — larger by a factor of **543** at the ankle.
   Using the open-chain value is a three-order-of-magnitude error with no
   symptom other than the robot falling over.

5. **PPO's entropy coefficient is not scale-free.** Adding the zero-step
   constraint cut the mean return from 116 to 25; the entropy bonus `c_H·H` does
   not scale with the return, so buying entropy became more profitable than
   balancing. The policy standard deviation climbed monotonically 0.50 → 1.25
   and performance decayed 61 % *at constant curriculum difficulty*. Nothing
   raised an error. `entropy_coef` is now 0.001 and `Mean action std` is the
   one-line diagnostic.

6. **Sample size gates every claim here.** At n = 128 the standard error on a
   proportion near 0.85 is 3.2 points, which is larger than any effect this task
   produces. Results measured at n = 128 were not reproducible; n ≳ 1800 is
   required to resolve a 2.3-point difference.

### Open questions

- **Single seed.** The +4.7 margin characterises one policy, not the method.
  Three independent training seeds are needed before it describes the algorithm.
- **The capture-point controller underperforms badly** (27.8 %). The likely
  cause is per-foot saturation of the pressure-centre command limiting lateral
  authority, but this has not been verified.
- **No hardware.** Everything here is simulation. Inertials are approximate,
  collision geometry is convex hulls, and there is no measured actuator model.

---

## Repository map

Where to find each thing, and what it is responsible for.

```text
.
├── assets/
│   ├── robots/nao/
│   │   ├── urdf/nao.urdf              tracked, BSD, verbatim upstream
│   │   ├── meshes/  texture/          UNTRACKED (CC BY-NC-ND), fetched locally
│   │   └── README.md                  asset provenance
│   └── generated/nao/                 UNTRACKED build output (derived URDF, USD)
├── configs/                           YAML planning configs for later phases
├── docs/                              see the documentation index below
├── scripts/                           every entry point (see table below)
├── source/humanoid_soccer_lab/        the installable Isaac Lab extension
├── tests/                             run without Isaac Sim
└── third_party/nao/                   upstream licenses and attribution
```

### The extension — `source/humanoid_soccer_lab/humanoid_soccer_lab/`

Split by **embodiment**, not by software layer: a robot's model, controllers and
tasks live together because they are coupled through numbers derived from that
machine and nothing else uses them.

```text
humanoid_soccer_lab/
├── common/      theory that mentions no robot
├── nao/         the constrained target embodiment
├── g1/          the capable source embodiment
├── transfer/    teaching one through the other
├── soccer/      the long-horizon target, inactive
└── tasks.py     registers every gym id
```

| Path | Responsibility |
| --- | --- |
| `common/capturability.py` | LIPM, DCM and zero-step capturability for **any** legged robot: takes a CoM height, a support polygon and a point inside it. NumPy only. |
| `nao/assets/nao_kinematics.py` | **All NAO-derived numbers.** Forward kinematics, centre of mass, composite inertia, support polygon. NumPy only, so tests verify it directly. |
| `nao/assets/nao.py` | `NAO_CFG` (passive, zero drives) and `NAO_STAND_CFG` (actuator groups, nominal posture, soft limits). |
| `nao/assets/nao_usd.py`, `nao_paths.py` | URDF-to-USD pipeline; filesystem layout. |
| `nao/controllers/dcm_balance.py` | Capture-point controller. Plain torch, testable without Isaac Sim. |
| `nao/tasks/nao_stand/` | The balance environment, its configuration and its PPO setup. |
| `g1/assets/g1.py` | Source-robot selection. `SOURCE_ROBOT` swaps between G1-29DOF, G1-23DOF and H1 in one line. |
| `g1/tasks/g1_walk/` | Teacher walking task, built on Isaac Lab's `G1FlatEnvCfg` with commands capped at the Froude-matched speed. |
| `transfer/correspondence.py` | Which joint on one robot stands for which on the other — and what the map cannot express. |
| `transfer/feasibility.py` | What the student can physically do, per direction, so the teacher can be ignored where it asks for the impossible. |
| `transfer/distillation.py` | The student objective and the three candidate teacher signals. |
| `soccer/` | Scaffold for the eventual soccer task. Intentionally unimplemented. |

### Entry points — `scripts/`

| Script | What it does |
| --- | --- |
| `fetch_nao_meshes.py` | One-time licensed asset bootstrap. Run this first. |
| `view_nao.py` | Phase-1 smoke test: loads the passive articulation and lets it fall. Validates geometry, mass and gravity. |
| `render_nao.py` | Renders the standing posture to `docs/img/nao_stand.png`. |
| `derive_gains.py` | Reproduces the joint gain table from the URDF. Tests assert the shipped gains match it. |
| `stand_nao.py` | The model-based baselines. `--controller {joint_pd,dcm}`. |
| `train.py` | Trains a policy. Thin wrapper that registers this extension's task ids, then delegates to Isaac Lab. |
| `play.py` | Replays a checkpoint and exports TorchScript to `<run>/exported/policy.pt`. |
| `compare_controllers.py` | **Scores every controller on one environment instance.** The source of the headline table. |
| `sweep_thresholds.py` | Largest push survived, per direction, against the theoretical bound. |
| `diagnose_recovery.py` | Checks the four assumptions a capturability comparison needs, including whether the robot actually stepped. |
| `check_nao_stand_env.py` | Environment instantiation and stepping check. |
| `stand_metrics.py` | Metric export shared by the baseline scripts. |
| `list_tasks.py` | Lists registered task ids. |
| `diagnose_isaac_startup.py` | Isolates Isaac Sim startup failures. |

### Documentation — `docs/`

| Document | Covers |
| --- | --- |
| [`transfer.md`](docs/transfer.md) | **The research plan**: the embodiment gap, the hypothesis, the three candidate teacher signals, order of work and threats to the result. |
| [`task_nao_stand.md`](docs/task_nao_stand.md) | **The balance task**: full specification, observations, reward table, curriculum, training and evaluation procedure. |
| [`stand_nao.md`](docs/stand_nao.md) | The joint PD baseline and its measurements. |
| [`stand_nao_dcm.md`](docs/stand_nao_dcm.md) | The capture-point controller, its derivation and its failure modes. |
| [`view_nao.md`](docs/view_nao.md) | The asset smoke test, options and troubleshooting. |
| [`architecture.md`](docs/architecture.md) | Project layering and task boundaries. |
| [`asset_pipeline.md`](docs/asset_pipeline.md) | URDF → USD conversion and asset policy. |
| `results/` | Measured results as JSON. |
| `img/` | Figures. |

---

## Setup

### 1. Isaac Lab

Install Isaac Lab separately. Developed against a source checkout at
`C:\IsaacLab`.

| Component | Version |
| --- | --- |
| Isaac Lab | 2.3.2 (`isaaclab` package 0.54.4) |
| Isaac Sim | 5.1.0 |
| rsl-rl-lib | 5.0.1 |
| PyTorch | 2.7.0+cu128 |
| Tested on | Windows 11, RTX 4070 Laptop |

### 2. This extension

```powershell
python -m pip install -e source\humanoid_soccer_lab
```

### 3. NAO meshes, once

The NAO geometry is CC BY-NC-ND 4.0 and upstream permits redistribution only
through an installer that obtains explicit assent, so **this repository ships no
mesh file**:

```powershell
python scripts\fetch_nao_meshes.py
```

It prints the license and requires you to type `I ACCEPT`. It downloads the
official ROS Noetic `nao_meshes` package, verifies its SHA-256 and extracts it
using only the Python standard library — native Windows, no WSL, no ROS.

### 4. Verify

```powershell
python -m pytest                       # needs neither Isaac Sim nor rendering
python scripts\view_nao.py --headless --max-steps 1
```

`view_nao.py` intentionally lets the robot collapse: that fall is what validates
the articulation, its collision geometry and gravity.

---

## Reproducing the results

```powershell
# derived parameters and the gain table
python scripts\derive_gains.py

# model-based baseline
python scripts\stand_nao.py --headless --controller joint_pd

# train (about 81 minutes for 700 iterations on an RTX 4070 Laptop)
python scripts\train.py --task NaoStand-Direct-v0 --headless `
    --num_envs 4096 --max_iterations 700 --seed 1 --run_name my_run

# export TorchScript from a checkpoint
python scripts\play.py --task NaoStand-Direct-v0 --headless --num_envs 8 `
    --checkpoint logs\rsl_rl\nao_stand\<run>\model_699.pt

# the headline table
python scripts\compare_controllers.py --headless --num-envs 2048 --steps 600 `
    --protocol directional --ablate-arms `
    --policy logs\rsl_rl\nao_stand\<run>\exported\policy.pt
```

Use `--num-envs 2048`. At 128 the binomial error swamps every effect this task
produces.

### Rendering on this machine

Offscreen capture crashes in the RTX renderer under `--headless` on this
hybrid-graphics laptop (the failure is in `rtx.scenedb` during Hydra engine
creation). `render_nao.py` therefore runs with a window:

```powershell
python scripts\render_nao.py --width 1280 --height 960
```

Training and evaluation are unaffected; they need no renderer.

---

## Licensing and attribution

This repository contains **no original NAO model**. Full provenance, upstream
commit SHAs and the licensing reasoning are in
[`third_party/nao/README.md`](third_party/nao/README.md).

| Component | Upstream | Commit | License |
| --- | --- | --- | --- |
| URDF (tracked, verbatim) | [ros-naoqi/nao_robot](https://github.com/ros-naoqi/nao_robot) | `6747646` | BSD 3-Clause, © 2009–2013 A. Hornung, University of Freiburg |
| Meshes and texture (fetched, untracked) | [ros-naoqi/nao_meshes](https://github.com/ros-naoqi/nao_meshes) | `7c5b9f3` | CC BY-NC-ND 4.0, © Aldebaran / SoftBank Robotics |

Two consequences are load-bearing:

- **Meshes are never committed.** Redistribution is permitted only via an
  installer that obtains the user's assent, so they are fetched locally.
- **The generated USD is never committed.** It embeds the licensed geometry and
  counts as Adapted Material, which under CC BY-NC-ND 4.0 §2(a)(1)(B) may be
  produced for non-commercial use but not shared.

The top-level [`LICENSE`](LICENSE) covers **only this project's own code**.
**Non-commercial only**: commercial use would require removing the NAO geometry
or a separate license from SoftBank Robotics.

---

## Known limitations

**Method**

- Single training seed; the measured margin describes one policy, not the method.
- Simulation only. No hardware transfer attempted or evaluated.
- The support polygon is an axis-aligned bounding box, which overestimates the
  contact region under foot roll or edge contact.
- No contact-mode state machine; controllers run without explicit knowledge of
  single versus double support.
- Actuator gains are an analytical design from URDF inertias, not hardware
  identification.

**Model**

- **The URDF zero pose is unreachable.** `LElbowRoll` is limited to
  `[-1.545, -0.035]` and `RElbowRoll` to `[0.035, 1.545]`; a NAO elbow cannot
  straighten. Defaults are clamped into the URDF's own limits.
- **Finger links carry placeholder inertials** (2e−06 kg). They are `mimic`
  joints imported as PhysX mimic constraints with zero drive gains; stable, but
  the least trustworthy part of the model.
- **Four links have no geometry** — `LElbow`, `RElbow`, `l_gripper`,
  `r_gripper` — producing harmless "unresolved reference" warnings.
- **Collision geometry is convex hulls.** Adequate for foot contact; revisit
  before manipulation.
- **`torso` does not exist in the simulator.** Fixed-joint merging folds it into
  `base_link`. Read body names from the loaded model, not from the URDF.

---

## Roadmap

1. ~~NAO asset integration and validation~~ — done.
2. ~~Actuator model and analytically derived joint gains~~ — done.
3. ~~Model-based balance baselines (joint PD, capture point)~~ — done.
4. ~~Zero-step balance RL task and trained policy~~ — done.
5. **Multiple seeds and an explanation for the capture-point result** — next.
6. Soccer field, ball asset, reset distributions.
7. Locomotion and ball approach.
8. Multi-agent play once the single-agent task is stable.
