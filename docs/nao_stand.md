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

## Work required before training

Static review identified the following integration and measurement issues:

1. CoM height is measured relative to ankle body origins, but its target is
   relative to the soles. The nominal difference is 45.11 mm.
2. CoM mass weights use `default_mass` even after the startup event changes
   the actual torso mass in PhysX.
3. `_reset_idx` calls the base reset events and subsequently overwrites their
   randomized states with nominal joint and root states.
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

Before a training campaign, validate resets and observations, run the environment
with zero and known small actions, then perform a short PPO update/save/load
check. Evaluate learned behavior on the same perturbation protocol as the PD.
Existing unit tests and PD simulations do not certify this RL environment.
