# Architecture

The project is an Isaac Lab external extension: a repository-level project with
`scripts/`, `source/` and documentation at the root, and one installable
extension under `source/humanoid_soccer_lab`.

## Organising principle

The package is split by **embodiment**, not by software layer. A robot's model,
its controllers and its tasks live together, because they are coupled through
numbers derived from that specific machine and nothing else uses them.

```text
humanoid_soccer_lab/
├── common/      theory that mentions no robot
├── nao/         the constrained target embodiment
├── g1/          the capable source embodiment
├── transfer/    teaching one through the other
├── soccer/      the long-horizon target, inactive
└── tasks.py     registers every gym id
```

### `common/`

Anything true of legged robots in general: the linear inverted pendulum, the
divergent component of motion, zero-step capturability. Expressed in terms of a
centre-of-mass height, a support polygon and a point inside it — never in terms
of a particular robot.

Pure NumPy, no Isaac Lab. That constraint is what lets the tests check the
theory directly rather than through a simulator, and it is the same reason the
NAO's kinematics module is written that way.

### `nao/` — the target

The robot a skill is meant to end up on. Small, weak, and limited for physical
rather than algorithmic reasons, which is what creates the research question.

- `assets/` — model, derived parameters, URDF-to-USD pipeline
- `controllers/` — model-based baselines that a learned policy has to beat
- `tasks/` — reinforcement learning environments

Every number in `assets/nao_kinematics.py` is derived from the vendored URDF and
covered by a test. That module is the authority; nothing else re-derives.

### `g1/` — the source

The robot a skill is learned on first, because it can perform it. Isaac Lab
ships the model, the actuator groups and a working locomotion task, so this
package selects and configures rather than deriving.

`SOURCE_ROBOT` in `assets/g1.py` picks which shipped humanoid plays teacher.
Changing it changes the size of the embodiment gap, which is the variable the
transfer study most wants to vary.

### `transfer/` — the research

Joint correspondence, feasibility masking and the student objective. Depends on
`common/` for the theory and on neither robot's URDF: the correspondence is
stated at the level of kinematic roles, and the student's envelope is passed in
as geometry.

See [`transfer.md`](transfer.md) for the plan and the open questions.

### `soccer/` — the long horizon

Kept intact and deliberately inactive. Balance and transfer have to hold before
a soccer task means anything.

## Boundaries

**Assets do not contain environments.** A test enforces this. A robot
configuration that knows about rewards has made itself unusable for anything
except the task it was written for.

**Theory does not contain robots.** `common/` takes geometry as arguments. The
moment it imports a URDF it stops being reusable for the second embodiment,
which is the whole reason it exists.

**Embodiments do not import each other.** `transfer/` is the only place the two
robots meet. Where it needs a constant from the other side — the NAO's
centre-of-mass height, the G1's leg length — it restates it with a comment
naming the source, rather than reaching across.

**Tasks live with their robot.** `tasks.py` at the package root imports them so
the gym ids register; it owns nothing.

## Task registration

Isaac Lab discovers tasks by importing `isaaclab_tasks`, which knows nothing
about an external extension. `scripts/train.py` and `scripts/play.py` therefore
import `humanoid_soccer_lab.tasks` before delegating, which registers:

| Id | Robot | Purpose |
| --- | --- | --- |
| `NaoStand-Direct-v0` | NAO | Zero-step balance under perturbation |
| `NaoStand-Direct-Play-v0` | NAO | Replay and TorchScript export |
| `G1Walk-Teacher-v0` | G1 | Teacher gait at the Froude-matched speed |
| `G1Walk-Teacher-Play-v0` | G1 | Rollout capture |

Registration is guarded: on a machine without gymnasium or Isaac Lab it degrades
to `REGISTERED = False` rather than raising, which is what lets the test suite
import the package anywhere.
