# `g1/` — the source embodiment

**Unitree G1, 29 actuated joints.** The robot a skill is learned on first,
because it can actually perform it: roughly 35 kg, 1.3 m tall, and actuators
with the torque density to walk dynamically.

## Why this robot

Isaac Lab ships the model, tuned actuator groups and a working velocity
locomotion task. Nothing here has to re-derive a robot from a URDF the way the
NAO track does, and nothing carries the redistribution constraints the NAO
meshes do — the USD downloads from Isaac Lab's Nucleus server on first use.

It is deliberately swappable. `SOURCE_ROBOT` in
`source/humanoid_transfer/humanoid_transfer/g1/assets/g1.py` selects which
shipped humanoid plays teacher:

| Value | DOF | Notes |
| --- | --- | --- |
| `g1_29dof` | 29 | Default. Widest gap to the NAO, hardest transfer. |
| `g1` | 23 | Same platform, reduced joint set. |
| `h1` | 19 | Larger, simpler. Matches the NAO's 19 commanded joints exactly — a control condition where joint counts agree and only scale differs. |

Changing it changes the size of the embodiment gap, which is the variable the
transfer study most wants to vary.

## The teacher task

`G1Walk-Teacher-v0` is Isaac Lab's `G1FlatEnvCfg` with one change: the command
range is capped at the Froude-matched speed.

Dynamic similarity between legged systems requires matched Froude numbers,
`Fr = v²/(g·l)`. With 0.74 m against the NAO's 0.2689 m, a student walking at
0.15 m/s corresponds to a teacher at **0.25 m/s**. The default configuration
samples to 1.0 m/s in both directions plus lateral and turning commands, and
every episode spent there is experience the transfer discards.

```powershell
python scripts\train.py --task G1Walk-Teacher-v0 --headless --num_envs 4096
```

See [`docs/teacher_task.md`](docs/teacher_task.md) for the derivation and the
assumption it rests on.

## What is here

| Path | Contents |
| --- | --- |
| `docs/` | Teacher task documentation. |
| `scripts/` | G1-specific entry points. |

Code lives in `source/humanoid_transfer/humanoid_transfer/g1/`. There are no
assets: the model is Isaac Lab's.
