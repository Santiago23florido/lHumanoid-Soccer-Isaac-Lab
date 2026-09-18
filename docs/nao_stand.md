# Experimental NAO standing RL task

`NaoStand-Direct-v0` and `NaoStand-Direct-Play-v0` register Claude's unfinished
standing-balance task. This is an experimental scaffold, not a validated or
trained controller. The verified controller is the joint-PD baseline described
in [stand_nao.md](stand_nao.md).

## Intended design

The actor commands 19 position offsets, scaled by 0.25 radians and added to the
nominal posture. The underlying implicit PD gains remain fixed. The proposed
actor observation has 65 values; the critic receives those plus 17 privileged
simulator values. The configuration uses PPO, random horizontal velocity
increments and domain randomization. No trained checkpoint or improvement over
the PD baseline is established.

## Runtime validation status

`scripts/check_nao_stand_env.py` now instantiates the task and exercises it, so
the issues below are split into those that have been closed against a running
simulation and those that remain open from static review.

Closed by runtime testing:

* **The scene builds and steps.** Observation widths (65 actor, 17 critic),
  action width (19), joint order, foot bodies and undesired-contact bodies all
  resolve as configured.
* **A zero action holds the posture.** 128 environments, 400 control steps, zero
  falls. This is the load-bearing claim of the residual-action design: an
  untrained policy starts from the stabilising baseline, not from a rag doll.
* **The reward separates good from bad.** A random policy returns +18.3 against
  +24.7 for the zero action and falls in 42% of environments.
* **The `torso` body does not exist.** The URDF-to-USD conversion merges it into
  `base_link`, so the mass-randomization event and the undesired-contact list
  both raised at scene creation. Names used in the config must be articulation
  body names, which are not always URDF link names.
* **CoM height frame (was issue 1).** It was measured from the ankle body
  origins, 45.11 mm above the soles, against a target defined at the soles --
  more than two kernel widths, which drove the height reward to 0.00015. It now
  measures from the ground plane and reads 0.149.

Fixed, each with a test that fails if it comes back:

3. **The reset discarded its own randomization.** `_reset_idx` applied the reset
   events and then wrote the defaults over them. The nominal state is now
   written first, so the events perturb it instead of being erased. This was the
   one that mattered: the domain randomization did not exist, and the only
   symptom was that toggling it changed nothing.
4. **The post-reset observation described the previous episode.** The derived
   values were computed in `_get_dones`, which runs before `_reset_idx`.
   `_get_observations` now refreshes them, so a reset environment reports the
   pose it restarted from rather than the one it fell into.
5. **The six held joints had no owner.** They now get their nominal targets
   explicitly, at reset and every control step.
8. **Actions were unbounded.** Clipped at three standard deviations, with the
   resulting targets clamped into the soft joint limits.
10. **`train.py` and `play.py` were placeholders.** They now delegate to Isaac
    Lab's RSL-RL scripts after registering this extension's task ids. The
    extension also installs, which it never did: `pyproject.toml` pointed
    `readme` outside the package directory and setuptools refused it.

Still open:

2. CoM mass weights use `default_mass` even after the startup event changes
   the actual torso mass in PhysX. Every CoM and DCM value is therefore slightly
   wrong, per environment and invisibly.
6. The quantity named foot normal force is a historical maximum of force
   magnitude rather than the current vertical component. It feeds the critic and
   reads as a normal load.
7. Resolved in the evaluation harness, which now threads the environment's own
   observation through, but `_get_observations` still writes the previous-action
   buffer as a side effect.
9. The nominal double-support rectangle does not track which feet are actually
   loaded, and the forward capturability bound is used for every push direction
   even though the backward bound is 0.443 m/s.

The curriculum consumes 24 million aggregate environment steps. With 4096
environments and 24 steps per iteration, that is approximately 244 iterations,
not the 2000 mentioned in a configuration comment.

## What the PD baseline leaves for the policy

128 environments, 600 control steps, pushes sampled uniformly in direction at
up to 0.548 m/s:

| Controller | Fall-free | Falls |
| --- | --- | --- |
| Joint PD (zero action) | **80.5 %** | 27 |
| Capture-point PD | 45.3 % | 75 |
| Random action | 32.8 % | 117 |

**80.5 % is the number to beat.** Without pushes the joint PD never falls, so
the 19.5 % is entirely the perturbation protocol.

### This number replaced a wrong one

An earlier measurement put the joint PD at 85.2 %. That run predated the reset
fix, so the domain randomization was being discarded and the controller was
being scored on an easier problem than the one it is supposed to solve.
Publishing 85.2 % would have set the bar below the real baseline.

The capture-point controller scoring *worse* is the other result worth
carrying. On a single forward push it beats the joint PD outright, recovering
0.55 m/s where the joint PD falls at 0.50. Its hip and arm gains were only ever
validated in the sagittal plane, and on omnidirectional pushes that costs it
more than the capture-point feedback wins. See
[stand_nao_dcm.md](stand_nao_dcm.md).

Not all 19.5 % is winnable without stepping. Pushes are uniform in direction
while the capturable bound is not — 0.548 m/s forward, 0.443 m/s backward,
0.620 m/s lateral — so the hardest backward pushes need a step. But the bound
itself is not a hard ceiling either: it assumes constant centroidal angular
momentum, and a hip or arm strategy breaks that assumption deliberately.

### Where each controller fails, per direction

Fixed push heading relative to the robot's facing, exact magnitude, 64
environments, 250 steps. The largest push each controller survives in more than
half the environments:

| Direction | LIPM bound | Joint PD | PPO policy | Ratio |
| --- | --- | --- | --- | --- |
| Forward | 0.548 m/s | 0.40 | **0.70** | **1.28x** |
| Backward | 0.443 m/s | 0.30 | **0.80** | **1.81x** |
| Lateral | 0.620 m/s | 0.50 | 0.60 | 0.97x |

The joint PD falls short of the bound in every direction, which is what a
controller with no representation of its own pressure centre should do: it
never tries to reach the edge of the foot.

The policy exceeds the bound forward and backward, and sits exactly *at* it
laterally. That contrast is the result. The hip and shoulder joints it commands
are pitch joints, so angular momentum is available in the sagittal plane and
not in the frontal one — and lateral pressure-centre motion comes from load
sharing between the two feet rather than from either ankle, which is the same
physical constraint the capture-point controller ran into.

The bound assumes constant centroidal angular momentum. The policy beats it
exactly where it can break that assumption.

### Which mechanism, measured

Freezing the eight shoulder and elbow joints at nominal and re-running the
sweep isolates what the arms contribute. Hip pitch stays active, so this
ablates the arms rather than all angular momentum.

| Backward push | Policy | Arms frozen |
| --- | --- | --- |
| 0.70 m/s | 98.4 % | **68.8 %** |
| 0.80 m/s | 76.6 % | **9.4 %** |
| 0.90 m/s | 34.4 % | **0.0 %** |

Forward at 0.80 m/s the drop is 9.4 % to 3.1 %; laterally at 0.70 m/s it is
32.8 % to 21.9 %. Both small.

**The arms are the backward recovery mechanism**, and close to irrelevant
forward and laterally. That matches the geometry: backward is the tightest
direction, with the heel 60.7 mm from the ankle against 103.3 mm of toe, so it
is where the ankle runs out first and where momentum is worth most. The arms
are pitch joints, so they have leverage in the sagittal plane and none in the
frontal one.

Forward the policy reaches 1.28x the bound with or without arms, so whatever
takes it past the limit there is the hip strategy or better use of the ankle,
not the arms.

**Read the per-magnitude curves, not the threshold.** On a 0.1 m/s grid the
50 % crossing resolves to one step and it understates this badly: freezing the
arms moves the backward crossing by a single step while taking the rate at
0.80 m/s from 77 % to 9 %. The joint PD's backward crossing also moved between
two runs, 0.30 and 0.40, which is the scale of the measurement noise at 64
environments.

Reproduce with:

```powershell
python scripts\sweep_thresholds.py --headless --num-envs 64 --ablate-arms `
       --policy logs
sl_rl
ao_stand\<run>\exported\policy.pt
```

Score a trained policy on exactly this protocol with:

```powershell
python scripts\compare_controllers.py --headless --num-envs 128 --steps 600 `
       --policy logs\rsl_rl\nao_stand\<run>\exported\policy.pt
```

which runs all three controllers against one environment instance, so the only
difference between them is the control law.
