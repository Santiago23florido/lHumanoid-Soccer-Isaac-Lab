# Humanoid Soccer Isaac Lab

Isaac Lab project for humanoid robot soccer research, built on the SoftBank
Robotics / Aldebaran **NAO H25 V5.0**.

The previous Franka Lagrangian MBRL work is preserved on:

```text
archive/franka-lagrangian-mbrl-2026-07-25
```

## Current development status

Three controllers now stand the NAO up, and all three are scored on one shared
protocol — 128 environments, the same reset distribution, omnidirectional
pushes to 0.548 m/s — so the numbers mean the same thing.

Stepping ends the episode: a foot more than 50 mm from where it started fails
the run. That makes this zero-step balance, and it is what lets a controller be
compared against the capturability bound at all.

| Controller | What it is | Fall-free | Ankle torque |
| --- | --- | --- | --- |
| **PPO, arms frozen** | The policy with its 8 arm joints held at nominal. | **88.3 %** | 55.9 % |
| PPO policy | 19 joint offsets, asymmetric actor-critic. | 84.4 % | 56.8 % |
| Joint PD | Holds the nominal posture. One line of control. | 80.5 % | 33.5 % |
| Capture-point PD | Feedback on the divergent component, ankle/hip/arm strategies. | 28.9 % | 27.3 % |

An earlier version of this table read 99.2 % for the policy, against a task
that did not forbid stepping. It was stepping: 20 of 20 survivors of a backward
push moved a foot, up to 762 mm. With that removed the margin over the joint PD
falls from +17.2 points to **+3.9**, and the drop is the measurement of what
stepping was worth.

Two things that follow, both recorded rather than smoothed over:

- The policy does **not** win by using less effort. It uses 56.8 % of the ankle
  torque budget against the PD's 33.5 %, and saturates three times as often.
- **Freezing its arms improves it.** On the stepping task the arms were the
  backward recovery mechanism; here they are a liability. Unexplained.

Reproduce any row with:

```powershell
python scripts\compare_controllers.py --headless --num-envs 128 --steps 600 `
       --ablate-arms --policy logssl_rl
ao_stand\<run>\exported\policy.pt
```

See [`docs/stand_nao.md`](docs/stand_nao.md) for the joint PD,
[`docs/stand_nao_dcm.md`](docs/stand_nao_dcm.md) for the capture-point
controller, [`docs/nao_stand.md`](docs/nao_stand.md) for the learning task and
[`docs/nao_stand_plan.md`](docs/nao_stand_plan.md) for what beating these
baselines is allowed to mean.

`scripts/train.py` and `scripts/play.py` are real entry points now, delegating
to Isaac Lab's RSL-RL scripts after registering this extension's task ids.

The original passive asset/viewer described below remains available. Its zero
drives are distinct from the standing configuration's active PD drives.

## Phase 1 — original passive NAO asset integration

The NAO is imported from its upstream URDF, converted to USD, and loads as a
valid free-floating PhysX articulation with correct masses, inertias, joint
limits and collision geometry.

Verified on Isaac Sim 5.1.0 / Isaac Lab 2.3.2, Windows 11, RTX 4070:

| Property | Value |
| --- | --- |
| Robot | `NaoH25V50` (NAO H25, version 5.0) |
| Bodies | 43 |
| Joints / DOFs | 42 (25 independently actuated + 17 upstream `mimic`) |
| Articulation root | `base_link`, **not** fixed to the world |
| Total mass | 5.3054 kg, matching the URDF exactly |
| Fixed frames merged | 36 sensor frames (cameras, sonars, FSRs, bumpers, IMU, tactile) |

**The original Phase 1 asset contains no reinforcement learning.** No rewards, observations,
actions, soccer, ball, locomotion, balance controller, training code, PPO,
policy networks or multi-agent environments. The robot has no controller and
collapses under gravity when the simulation starts — that fall is precisely
what validates the articulation, its collision geometry and gravity.

## Setup

### 1. Install Isaac Lab

Install Isaac Lab separately (this project is developed against a source
checkout at `C:\IsaacLab`, Isaac Lab 2.3.2 / Isaac Sim 5.1.0).

### 2. Install this extension

```powershell
python -m pip install -e source\humanoid_soccer_lab
```

### 3. Fetch the NAO meshes (one time)

The NAO geometry is licensed CC BY-NC-ND 4.0 and upstream permits
redistribution only through an installer that obtains the user's explicit
assent, so **this repository ships no mesh file**. Fetch them onto your machine:

```powershell
python scripts\fetch_nao_meshes.py
```

The script prints the license and requires you to type `I ACCEPT` before
downloading anything; pass `--accept-license` to skip the prompt in automation.
It downloads the official ROS Noetic `nao_meshes` package, verifies its SHA-256,
and extracts it using only the Python standard library — **native Windows, no
WSL, no Ubuntu, no ROS, no 7-Zip**.

## Smoke test — the one command

```powershell
& "C:\Users\USER\miniconda3\shell\condabin\conda-hook.ps1"
conda activate env_isaaclab
cd C:\IsaacLab
.\isaaclab.bat -p "C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_nao.py" --device cuda:0 --rendering_mode performance --kit_args=--/app/vulkan=false
```

`--rendering_mode performance --kit_args=--/app/vulkan=false` forces Direct3D 12
and is required on this machine, where the RTX/Vulkan path crashes at GUI
startup. Headless runs do not need it:

```powershell
.\isaaclab.bat -p "C:\Users\USER\Documents\FrugalStage\lagrangian-mbrl-franka\scripts\view_nao.py" --device cuda:0 --headless --max-steps 1
```

The command does everything automatically: on first run it generates the derived
URDF, converts it to USD (~10 s, cached in `assets/generated/nao/`), builds the
scene and loads the robot. **No manual import through the Isaac Sim GUI.**

### Expected behavior

1. Isaac Sim opens showing a NAO hovering ~2.7 cm above a ground plane.
2. Diagnostics print to the terminal: robot name, body count, joint count, DOF
   count read back from the articulation, per-body masses, the full joint limit
   table with effort and velocity limits, total mass, and the articulation root.
3. Checks report `PASS` for geometry/scale and for physics/gravity.
4. Physics starts and the robot collapses to the ground and stays there.

See [`docs/view_nao.md`](docs/view_nao.md) for options and troubleshooting.

## Tests

```powershell
python -m pytest
```

They need neither Isaac Sim nor rendering. Mesh-dependent tests skip themselves
if the bootstrap has not been run.

## Source attribution and licensing

This repository contains **no original NAO model**. Full provenance, upstream
commit SHAs, the exact files copied, and the licensing decisions are documented
in [`third_party/nao/README.md`](third_party/nao/README.md).

| Component | Upstream | Commit | License |
| --- | --- | --- | --- |
| URDF (`assets/robots/nao/urdf/nao.urdf`, verbatim) | [ros-naoqi/nao_robot](https://github.com/ros-naoqi/nao_robot) | `6747646` | BSD 3-Clause, © 2009-2013 A. Hornung, University of Freiburg |
| Meshes and texture (fetched, untracked) | [ros-naoqi/nao_meshes](https://github.com/ros-naoqi/nao_meshes) | `7c5b9f3` | CC BY-NC-ND 4.0, © Aldebaran / SoftBank Robotics |

The top-level [`LICENSE`](LICENSE) covers **only this project's own code**. It
does not apply to, and does not relicense, any upstream NAO asset.

Two consequences are load-bearing, and are argued in full in
`third_party/nao/README.md`:

- **Meshes are never committed.** Upstream allows redistribution only via an
  installer that obtains the user's assent, so they are fetched locally.
- **The generated USD is never committed.** It embeds the licensed geometry, so
  under CC BY-NC-ND 4.0 §2(a)(1)(B) it may be produced for non-commercial use
  but not shared. `assets/generated/` is untracked.

**Non-commercial only.** Any commercial use of this project would require
removing the NAO geometry or a separate license from SoftBank Robotics.

## Repository layout

```text
.
|-- assets/
|   |-- robots/nao/
|   |   |-- urdf/nao.urdf        # tracked, BSD, verbatim upstream
|   |   |-- meshes/ texture/     # UNTRACKED, CC BY-NC-ND, fetched locally
|   |   `-- README.md
|   `-- generated/nao/           # UNTRACKED build artifacts (derived URDF, USD)
|-- configs/                     # planning configs
|-- docs/                        # architecture, asset pipeline, viewer guide
|-- scripts/
|   |-- fetch_nao_meshes.py      # one-time licensed asset bootstrap
|   `-- view_nao.py              # canonical Phase 1 smoke test
|-- source/humanoid_soccer_lab/  # Isaac Lab external extension
|   `-- humanoid_soccer_lab/
|       |-- assets/
|       |   |-- nao.py           # NAO_CFG ArticulationCfg
|       |   |-- nao_usd.py       # derived URDF + USD conversion
|       |   `-- nao_paths.py     # layout, stdlib only
|       `-- tasks/               # RL scaffold, intentionally unimplemented
|-- third_party/nao/             # attribution and upstream licenses
`-- tests/
```

## Known limitations

- **Passive viewer.** `view_nao.py` intentionally lets the robot fall. Use
  `stand_nao.py` for the active PD baseline.
- **Meshes must be fetched** once per checkout; they cannot be redistributed.
- **The URDF zero pose is unreachable.** `LElbowRoll` is limited to
  `[-1.545, -0.035]` and `RElbowRoll` to `[0.035, 1.545]`, because a NAO elbow
  cannot fully straighten. Defaults are clamped into the URDF's own limits.
- **Finger links carry placeholder inertials** of 2e-06 kg with 1.1e-09 inertia
  in the upstream URDF. They are imported as-is. Because they are `mimic`
  joints they are imported as PhysX mimic constraints; with the drives left at
  zero the articulation is stable, but these links remain the least trustworthy
  part of the model and residual finger jitter is visible.
- **Two actuator configurations.** `NAO_CFG` keeps passive drives;
  `NAO_STAND_CFG` provides posture gains. Their gains are documented in the
  standing guide and are an approximate design, not hardware identification.
- **Four links have no geometry.** `LElbow`, `RElbow`, `l_gripper` and
  `r_gripper` carry inertia but no visual or collision mesh upstream, producing
  harmless "unresolved reference" warnings.
- **Collision geometry is convex hulls** of the upstream collision STLs. Good
  enough for foot contact; revisit before fine manipulation.
- **Windows GUI needs the Direct3D 12 flags** shown above.
- **Isaac Sim emits deprecation warnings** while merging the 36 fixed sensor
  frames. Merging is required: 27 of those links have no inertial at all.

## Next phases

Development sequence, following the standing baseline:

1. Actuator model and joint gains: implemented and verified for the PD baseline.
2. Soccer field, ball asset, contact sensors, reset logic.
3. Complete and validate the standing RL draft, then add locomotion and ball approach.
4. Reward terms for balance, gait regularity, ball control, shooting.
5. Multi-agent play once the single-agent task is stable.
