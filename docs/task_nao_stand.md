# `NaoStand-Direct-v0` — zero-step balance under perturbation

A Direct RL environment in which the NAO must hold its nominal standing posture
against external pushes **without changing its contact configuration**. A foot
that lifts or slides more than 50 mm ends the episode.

That constraint is the point of the task. Without it the robot solves the
problem by stepping, which is a different problem with different theory behind
it, and the zero-step capturability bound that the reward is built on no longer
applies.

- Task package: [`source/humanoid_soccer_lab/humanoid_soccer_lab/tasks/direct/nao_stand/`](../source/humanoid_soccer_lab/humanoid_soccer_lab/tasks/direct/nao_stand/)
- Registered ids: `NaoStand-Direct-v0`, `NaoStand-Direct-Play-v0`
- Robot configuration: `NAO_STAND_CFG` in [`assets/nao.py`](../source/humanoid_soccer_lab/humanoid_soccer_lab/assets/nao.py)

## Specification

| Property | Value | Set in |
| --- | --- | --- |
| Control rate | 100 Hz (`sim.dt = 1/200`, `decimation = 2`) | `nao_stand_env_cfg.py` |
| Episode length | 20 s, truncated (`is_finite_horizon = False`) | `nao_stand_env_cfg.py` |
| Action space | 19 (11 leg + 8 arm), joint offsets | `nao_stand_env_cfg.py` |
| Action scale / clip | 0.25 rad / ±3.0 | `nao_stand_env_cfg.py` |
| Actor observation | 65 | `_get_observations` |
| Critic observation | 65 + 17 privileged = 82 | `_get_observations` |
| Parallel environments | 4096 | `nao_stand_env_cfg.py` |

The control rate follows from the plant: the open-loop unstable pole is
5.54 rad/s and the error doubling time is 115 ms, so 100 Hz gives about 11.5
actions per doubling. It also matches the real NAO's 10 ms control cycle.

### Action parameterisation

```
q_desired = q_nominal + 0.25 * clip(action, -3, 3)
```

applied as a position target to the implicit PD. `action = 0` is therefore
exactly the joint-PD baseline, which already stands: the policy learns a
correction on top of a working controller rather than balance from scratch.

### Observations

The actor sees only what the physical NAO can measure. The critic additionally
receives simulator state, which reduces value-estimation variance without
biasing the policy gradient.

| Actor (`policy`, 65) | Dim |
| --- | --- |
| Projected gravity in base frame | 3 |
| Base angular velocity | 3 |
| Joint position offset from nominal | 19 |
| Joint velocity | 19 |
| Previous action | 19 |
| Binary foot contact | 2 |

| Critic extra (`critic`, 17) | Dim |
| --- | --- |
| Base linear velocity | 3 |
| CoM offset from polygon centroid | 2 |
| CoM velocity | 2 |
| Divergent component of motion | 2 |
| Foot normal forces, normalised by weight | 2 |
| Applied push | 3 |
| Base height | 1 |
| CoM height | 1 |
| DCM margin to polygon edge | 1 |

Configured through `obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}`.

### Termination

| Cause | Condition | Logged as |
| --- | --- | --- |
| Fall | base height < 0.20 m (nominal 0.321) | `Episode_Termination/fall` |
| Topple | projected gravity z > −0.7 (tilt > 45°) | `Episode_Termination/fall` |
| Step | max foot displacement > 0.05 m | `Episode_Termination/step` |
| Timeout | 20 s elapsed | `Episode_Termination/time_out` |

Falls and steps are logged separately on purpose. During training the two
answer different questions — a run where every termination is a step is
constrained by the contact requirement, not by balance — and that split is what
made the reward and hyperparameter problems diagnosable.

Timeout is truncation, not termination, so PPO bootstraps `V(s_T)`. Treating it
as termination would bias the value function downward on every state that
survives a full episode.

### Reward

All terms are scaled by the control timestep.

| Term | Form | Scale |
| --- | --- | --- |
| `alive` | +1 per step | +2.0 |
| `dcm` | `exp(-‖ξ_xy‖²/σ²)`, σ = 0.03 | +3.0 |
| `upright` | `-‖g_b,xy‖²` | −2.0 |
| `com_height` | `exp(-(z_com − 0.2689)²/σ²)`, σ = 0.02 | +1.0 |
| `posture` | `exp(-‖q − q̄‖²/σ²)`, σ = 0.5 | +0.5 |
| `joint_torque` | `-‖τ‖²` | −1e−3 |
| `joint_accel` | `-‖q̈‖²` | −2.5e−7 |
| `joint_vel` | `-‖q̇‖²` | −1e−3 |
| `action_rate` | `-‖aₜ − aₜ₋₁‖²` | −0.01 |
| `foot_slip` | `-Σ‖v_foot,xy‖² · 1[loaded]` | −0.5 |
| `undesired_contact` | `-Σ 1[non-plantar contact]` | −2.0 |
| `foot_lift` | `-Σ 1[foot unloaded]` | −1.0 |
| `foot_displacement` | `-Σ‖p_foot − p_foot,0‖` | −4.0 |
| `joint_limit` | `-Σ ReLU(|q| − q_soft)` | −1.0 |

The `dcm` term is derived rather than tuned: keeping the divergent component of
motion near the support-polygon centroid is exactly the zero-step recoverability
condition (see [`docs/stand_nao_dcm.md`](stand_nao_dcm.md)).

`foot_lift` and `foot_displacement` price the zero-step constraint; the hard
termination above is what actually enforces it. A penalty alone was measured to
be insufficient.

### Perturbation curriculum

Pushes are applied every 2 s as a direct velocity write on the articulation
root, with direction uniform on [0, 2π). The magnitude is **not** absolute — it
is a fraction of what is recoverable along the sampled heading:

```
v_push(θ, t) = α(t) · ω₀ · d(θ),      α: 0.15 → 0.90
```

ramped over 24e6 consumed environment steps. `d(θ)` comes from
`nao_kinematics.capturable_velocity_in_direction`.

This matters because the bound is anisotropic by a factor of 1.75 on this robot
(0.443 m/s backward, 0.775 m/s diagonal). A curriculum fixed at the forward
bound would ask for 124 % of what is recoverable backward, which is not a hard
episode but an impossible one.

### Domain randomisation

| Parameter | When | Range |
| --- | --- | --- |
| Static friction | startup | [0.6, 1.2] |
| Dynamic friction | startup | [0.5, 1.0] |
| Restitution | startup | [0.0, 0.1] |
| Torso mass | startup | ±0.3 kg on 1.05 kg |
| Joint position | reset | ±0.05 rad |
| Joint velocity | reset | ±0.1 rad/s |
| Base x, y | reset | ±0.01 m |
| Base roll, pitch | reset | ±0.05 rad |
| Base yaw | reset | [−π, π] |
| Base linear velocity | reset | ±0.1 m/s |
| Base angular velocity | reset | ±0.2 rad/s |

Full-range yaw randomisation is required so the policy cannot memorise a heading.

## Training

```powershell
python scripts/train.py --task NaoStand-Direct-v0 --headless `
    --num_envs 4096 --max_iterations 700 --seed 1 --run_name mi_corrida
```

About 81 minutes for 700 iterations (68.8e6 environment steps) on an RTX 4070
Laptop.

### The one diagnostic to watch

`Mean action std` must **flatten**, not climb. PPO's entropy bonus `c_H · H` has
a magnitude independent of the return, so when the reward scale drops the bonus
becomes relatively larger and the optimiser buys entropy at the cost of
performance. On this task that shows up as the policy standard deviation rising
monotonically from 0.50 past 1.2, and episode length decaying for hundreds of
iterations at constant curriculum difficulty.

`entropy_coef = 0.001` is correct for the current reward scale (mean return
≈ 66). It is **not** scale-free and must be revisited whenever reward terms
change.

Healthy run, for reference:

| Iteration | Episode length | Action std | Return | α |
| --- | --- | --- | --- | --- |
| 1 | 11 | 0.50 | 0.4 | 0.15 |
| 101 | 1660 | 0.53 | 91.9 | 0.46 |
| 251 | 1226 | 0.55 | 67.7 | 0.90 |
| 401 | 1327 | 0.58 | 72.9 | 0.90 |
| 700 | 1231 | 0.63 | 66.5 | 0.90 |

Difficulty saturates around iteration 250; the slope from there to 700 is
+0.12 steps per iteration, i.e. flat.

## Evaluation

```powershell
# export TorchScript from a checkpoint
python scripts/play.py --task NaoStand-Direct-v0 --headless --num_envs 8 `
    --checkpoint logs/rsl_rl/nao_stand/<run>/model_699.pt

# score every controller on one environment instance
python scripts/compare_controllers.py --headless --num-envs 2048 --steps 600 `
    --protocol directional --ablate-arms `
    --policy logs/rsl_rl/nao_stand/<run>/exported/policy.pt
```

Use `--num-envs 2048`. The metric is binomial; at n = 128 the standard error on
a proportion near 0.85 is 3.2 percentage points, which is larger than every
effect this task produces.

Current results are in [`docs/results/comparison_directional.json`](results/comparison_directional.json)
and summarised in the repository README.

## Known limitations

- Single training seed. The measured margin characterises one policy, not the
  method.
- The support polygon is an axis-aligned bounding box, which overestimates the
  contact region under foot roll or edge contact.
- No contact-mode state machine; controllers run without explicit knowledge of
  single versus double support.
- Simulation only. No hardware transfer has been attempted or evaluated.
