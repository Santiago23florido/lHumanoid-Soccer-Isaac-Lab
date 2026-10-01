# Architecture

The project is an Isaac Lab external extension: a repository-level project with
`scripts/`, `source/` and documentation at the root, and one installable
extension under `source/humanoid_transfer`.

## Organising principle

The package is split by **embodiment**, not by software layer. A robot's model,
its controllers and its tasks live together, because they are coupled through
numbers derived from that specific machine and nothing else uses them.

```text
humanoid_transfer/
├── common/      theory that mentions no robot
├── nao/         the constrained target embodiment
├── g1/          the capable source embodiment
├── transfer/    teaching one through the other
├── soccer/      the long-horizon target, inactive
└── tasks.py     registers every gym id
```

Outside the extension, `g1/policy/` holds the trained teacher: the only derived
binary this repository tracks, and the one artifact the transfer study cannot
regenerate. See [`g1/policy/README.md`](g1/policy/README.md).

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

See [`transfer.md`](transfer/docs/plan.md) for the plan and the open questions.

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

**The standing task is not an input to the walking work.** A test enforces this,
and it is the newest boundary because it is the one that was crossed.

The NAO's balance work — `nao/tasks/nao_stand` and `nao/controllers` — is
**kept, registered and documented**, and is **frozen out of the walking
development path**. What is shared is the robot's *body*: `nao/assets` and
`common/`, which would read exactly the same if the standing study had never
happened.

| Shared with the walking work | Frozen out of it |
| --- | --- |
| `nao/assets/` — URDF, USD, kinematics, actuator model | `nao/tasks/nao_stand/` |
| `common/` — LIPM, DCM, capturability as functions of geometry | `nao/controllers/` |
| | `nao/results/`, `nao/docs/baseline_*.md`, `nao/docs/task_balance.md` |

The reason is not tidiness. `nao_stand` terminates on a step, and that is
correct: "recover without stepping" *is* the task it poses, and the zero-step
capturable envelope is its definition. **A condition may define a task.** The
walking task poses the opposite objective — track the command, do not fall —
so the same criterion there stops defining and starts prescribing: it measures
how close to standing still a state is, which for a gait is a statement about
*how* to walk rather than about what the body can do.

That has already cost two rewrites, which is why it is a test and not a note:

- `transfer/feasibility.py` weighted the teacher by zero-step capturability. It
  down-weighted 0 of 12 800 real samples, on the double-stance polygon, for a
  robot in single support most of a cycle. Replaced by joint reachability.
- `g1/tasks/g1_walk` justified the student's 0.15 m/s target by the 0.443 m/s
  capturable bound — "stay near a third of it". The speed is a free parameter of
  the experiment; the margin a learned gait keeps is a result, not its reason.

Capturability is not deleted anywhere. It is reported
(`transfer.feasibility.standing_margin`) rather than required, which keeps the
same number a finding instead of an assumption.

Enforced by [`tests/test_development_baseline.py`](tests/test_development_baseline.py),
which also asserts the standing task still registers and still carries its
termination — a guard that passed because the code was deleted would say nothing.

## Task registration

Isaac Lab discovers tasks by importing `isaaclab_tasks`, which knows nothing
about an external extension. `scripts/train.py` and `scripts/play.py` therefore
import `humanoid_transfer.tasks` before delegating, which registers:

| Id | Robot | Purpose | State |
| --- | --- | --- | --- |
| `G1Walk-Teacher-v0` | G1 | Teacher gait at the Froude-matched speed | **Trained.** Weights pinned in `g1/policy/` |
| `G1Walk-Teacher-Play-v0` | G1 | Rollout capture | Trained |
| `NaoStand-Direct-v0` | NAO | Zero-step balance under perturbation | **Kept, not an input.** See the boundary above |
| `NaoStand-Direct-Play-v0` | NAO | Replay and TorchScript export | Kept |
| NAO walking | NAO | Open objective: track the command, do not fall | **Not started.** The critical path |

Registering a task is not the same as building on it. The NAO ids stay
registered so the standing result remains reproducible; the walking work starts
from the robot's body.

Registration is guarded: on a machine without gymnasium or Isaac Lab it degrades
to `REGISTERED = False` rather than raising, which is what lets the test suite
import the package anywhere.
