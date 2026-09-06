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

Still open from static review:

2. CoM mass weights use `default_mass` even after the startup event changes
   the actual torso mass in PhysX.
3. `_reset_idx` calls the base reset events and subsequently overwrites their
   randomized states with nominal joint and root states. **Confirmed by
   measurement**: enabling and disabling reset randomization produced identical
   fall-free rates, because the randomization never survived to the first step.
4. Derived CoM/DCM/contact values can describe the previous episode when
   observations are assembled immediately after a reset.
5. The six held-joint targets need explicit initialization and a check that
   zero policy action reproduces the complete PD baseline.
6. The quantity named foot normal force is a historical maximum of force
   magnitude, rather than the current vertical force.
7. `_get_observations` changes the previous-action buffer; its indexing and
   repeated-read behavior need verification.
8. Actions and position targets have no explicit clipping in the environment.
   A Gaussian action distribution is not bounded by the 0.25 scaling factor.
9. The nominal double-support rectangle does not track actual loaded contacts.
   The forward capturability bound is also used for all push directions.
10. `scripts/train.py` and `scripts/play.py` remain placeholders. Extension
    installation, asset resolution, rollout, saving and loading need end-to-end
    validation. The play configuration still has sources of randomness.

The curriculum consumes 24 million aggregate environment steps. With 4096
environments and 24 steps per iteration, that is approximately 244 iterations,
not the 2000 mentioned in a configuration comment.

## What the PD baseline leaves for the policy

Measured with `scripts/check_nao_stand_env.py --pushes`, 128 environments,
600 control steps, pushes sampled uniformly in direction at up to 0.548 m/s:

| Controller | Fall-free | Falls | Mean return |
| --- | --- | --- | --- |
| Zero action (pure PD) | 85.2 % | 21 | +35.7 |
| Random action | 32.8 % | 117 | +26.9 |

Without pushes the PD never falls, so **14.8 % is the headroom a policy has to
win**, and it is the number any training run should be judged against. Part of
that headroom is not winnable by an ankle strategy at all: pushes are uniform in
direction while the capturable bound is not (0.548 m/s forward, 0.443 m/s
backward, 0.620 m/s lateral), so a backward push at full magnitude is beyond
what any non-stepping controller can absorb. Closing the rest requires the arm
and hip strategies the PD structurally cannot express.

Before a training campaign, resolve the open issues above, then perform a short
PPO update/save/load check. Evaluate learned behavior on the same perturbation
protocol as the PD. Existing unit tests and PD simulations do not certify this
RL environment.
