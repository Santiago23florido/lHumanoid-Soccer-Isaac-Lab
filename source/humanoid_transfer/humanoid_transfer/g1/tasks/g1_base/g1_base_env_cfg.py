"""Base tasks on the source robot, for a generator trained across several tasks.

``transfer/docs/method_revision.md`` §6 sets the target design: a generator trained on about
five base tasks, each performed by an expert on the G1, evaluated by leaving one task out.
The repository had three: the walk (``G1Walk-Teacher-v0``) and Isaac Lab's published flat
and rough policies. These are three more.

Each is Isaac Lab's ``G1FlatEnvCfg``, as the walk teacher is, with **only the command
distribution changed**. A command range defines a task; it does not say how to walk.
Rewards, terrain, observations and the PPO recipe (``G1FlatPPORunnerCfg``, 1500 iterations)
are Isaac Lab's. Since 2026-11-06 extra experts are trained this way.

All three are on flat ground. ``z*`` carries no terrain, and even the rough task runs on a
plane (``G1RoughOnFlat``, ``transfer/scripts/held_out_task.sh``).

Two candidates are left out, for reasons in the recipe:
- *Turning in place.* Isaac Lab's ``feet_air_time_positive_biped`` pays nothing when the
  linear command is under 0.1 m/s, so the recipe asks for no stepping there.
- *Stairs.* Terrain is outside ``z*``.

The ranges are free parameters of the experiment, declared as such. Whether an expert walks
is decided by its pre-registered acceptance check (``skill_space.md`` §10.12), not assumed.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)

BACKWARD_SPEED = (-0.5, 0.0)
"""m/s. Half the published forward cap (1.0 m/s), walking backwards only."""

SIDESTEP_SPEED = (-0.5, 0.5)
"""m/s, lateral. The published flat task's own lateral range, with no forward part."""

JOG_SPEED = (1.0, 1.5)
"""m/s. Above the published cap: a faster gait, possibly with flight phases."""

PLAY_COMMANDS = {
    "backward": {"slow": (-0.2, 0.0, 0.0), "mid": (-0.35, 0.0, 0.0), "fast": (-0.5, 0.0, 0.0)},
    "sidestep": {
        "left": (0.0, 0.25, 0.0),
        "left_fast": (0.0, 0.45, 0.0),
        "right": (0.0, -0.35, 0.0),
    },
    "jog": {"slow": (1.0, 0.0, 0.0), "mid": (1.25, 0.0, 0.0), "fast": (1.5, 0.0, 0.0)},
}
"""The fixed commands (m/s, m/s, rad/s) each expert's play rollouts are captured at."""


def _set_ranges(cfg, lin_vel_x, lin_vel_y) -> None:
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = lin_vel_x
    ranges.lin_vel_y = lin_vel_y
    ranges.ang_vel_z = (0.0, 0.0)
    ranges.heading = (0.0, 0.0)


def _play(cfg) -> None:
    """As ``G1WalkTeacherEnvCfg_PLAY``: few environments, no corruption, no pushes."""
    cfg.scene.num_envs = 32
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class G1BaseBackwardEnvCfg(G1FlatEnvCfg):
    """Walking backwards on flat ground."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_ranges(self, BACKWARD_SPEED, (0.0, 0.0))


@configclass
class G1BaseBackwardEnvCfg_PLAY(G1BaseBackwardEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)


@configclass
class G1BaseSidestepEnvCfg(G1FlatEnvCfg):
    """Stepping sideways on flat ground."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_ranges(self, (0.0, 0.0), SIDESTEP_SPEED)


@configclass
class G1BaseSidestepEnvCfg_PLAY(G1BaseSidestepEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)


@configclass
class G1BaseJogEnvCfg(G1FlatEnvCfg):
    """Fast forward gait on flat ground."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_ranges(self, JOG_SPEED, (0.0, 0.0))


@configclass
class G1BaseJogEnvCfg_PLAY(G1BaseJogEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)
