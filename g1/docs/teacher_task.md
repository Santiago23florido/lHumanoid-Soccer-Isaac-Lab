# `G1Walk-Teacher-v0`

Flat-ground walking on the source robot, used to produce the teacher policy.

## What it is

Isaac Lab's `G1FlatEnvCfg`, subclassed. The shipped configuration already walks;
reproducing it would add risk without adding anything the transfer study needs.
The contribution of this project is not a better way to make a G1 walk.

## The one change

The command distribution, and it follows from Froude scaling. Two legged systems
are dynamically similar when

```
Fr = v² / (g · l)
```

matches. With the teacher's 0.74 m base height against the student's 0.2689 m
centre-of-mass height:

```
v_teacher = v_student · √(0.74 / 0.2689) = v_student · 1.66
```

| Quantity | Value |
| --- | --- |
| Student target speed | 0.15 m/s |
| Froude-matched teacher speed | ≈ 0.25 m/s |
| Isaac Lab default range | ±1.0 m/s, plus lateral and turning |

0.15 m/s for the student is chosen well inside its weakest capturable direction
(0.443 m/s), because a gait has to stay recoverable *between* steps rather than
only at the boundary.

Lateral and angular commands are zeroed. They have no student counterpart yet,
and every episode spent there is experience the transfer discards.

## Assumption worth attacking

Froude matching is a **necessary** condition for dynamic similarity, not a
sufficient one. It says nothing about whether the NAO's actuators can deliver
the torques the scaled gait requires. Torque feasibility is a separate question
and is not addressed anywhere in this repository yet.

## Configuration

| Property | Value | Source |
| --- | --- | --- |
| Environment | `G1FlatEnvCfg` | Isaac Lab |
| PPO config | `G1FlatPPORunnerCfg` | Isaac Lab, unmodified |
| Robot | `G1_CFG` (23 body DOF + 14 hand) | Isaac Lab, selected by `SOURCE_ROBOT` |
| `lin_vel_x` | `(0.0, 0.25)` | this task |
| `lin_vel_y`, `ang_vel_z`, `heading` | zeroed | this task |

## Running it

```powershell
python scripts\train.py --task G1Walk-Teacher-v0 --headless `
    --num_envs 4096 --max_iterations 1500

python scripts\play.py --task G1Walk-Teacher-Play-v0 --headless --num_envs 32
```

The play variant fixes the command at the matched speed and disables the push
events, so recorded rollouts are comparable to each other.

The G1 USD downloads from Isaac Lab's Nucleus server on first use, so the first
run needs network access.

## What this task is not

It is not a contribution and it is not tuned. If the teacher fails to walk, the
fix is to look at Isaac Lab's configuration rather than at this file.
