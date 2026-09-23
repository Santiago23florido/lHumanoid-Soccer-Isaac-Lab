"""Teacher walking task on the source robot.

Built on Isaac Lab's ``G1FlatEnvCfg`` rather than written from scratch. The
shipped configuration already walks, and reproducing it would add risk without
adding anything the transfer study needs.

What this configuration changes is the *command distribution*, and it changes it
for one reason. The teacher exists to be imitated by a robot that is five times
lighter and less than half as tall. A gait the G1 performs comfortably at
1.0 m/s has no counterpart on the NAO, so a teacher trained across the full
default command range spends most of its experience in a regime the student can
never enter.

The commands are therefore capped at the fastest walk that is plausibly within
the student's reach, expressed through Froude scaling (see
:data:`FROUDE_MATCHED_SPEED`). This is a modelling assumption, stated here so it
can be attacked: it is a necessary condition for dynamic similarity, not a
sufficient one, and it says nothing about whether the NAO's actuators can
deliver the required torques.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
)

GRAVITY = 9.81

STUDENT_LEG_LENGTH = 0.2689
"""NAO centre-of-mass height above the soles in the nominal posture, in metres.

Taken from ``nao.assets.nao_kinematics.com_height_above_soles``. Repeated as a
constant rather than imported because importing the NAO kinematics here would
make the teacher task depend on the student's URDF, and the two embodiments are
meant to stay separable.
"""

TEACHER_LEG_LENGTH = 0.74
"""Source-robot standing base height, in metres. See ``g1.assets.g1``."""


def froude_matched_speed(student_speed: float) -> float:
    """Teacher speed dynamically similar to ``student_speed`` on the student.

    Two legged systems are dynamically similar when their Froude numbers match::

        Fr = v^2 / (g * l)

    so equal Froude gives ``v_teacher = v_student * sqrt(l_teacher / l_student)``.

    The scaling is why a small robot looks hurried at speeds a large one strolls
    through, and why copying a joint trajectory across a scale change produces a
    gait that is wrong in a way no amount of retargeting fixes.
    """
    return student_speed * math.sqrt(TEACHER_LEG_LENGTH / STUDENT_LEG_LENGTH)


STUDENT_TARGET_SPEED = 0.15
"""Forward speed the student is eventually meant to reach, in m/s.

Chosen well inside the NAO's zero-step capturable envelope, whose weakest
direction is 0.443 m/s. A walking gait has to remain recoverable between steps,
so the target speed has to leave margin against that bound rather than approach
it.
"""

FROUDE_MATCHED_SPEED = froude_matched_speed(STUDENT_TARGET_SPEED)
"""Teacher speed corresponding to :data:`STUDENT_TARGET_SPEED`, about 0.25 m/s."""


@configclass
class G1WalkTeacherEnvCfg(G1FlatEnvCfg):
    """Flat-ground walking, restricted to the student's reachable speed range."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Forward speeds only, capped at the Froude-matched target. The default
        # configuration samples up to 1.0 m/s in both directions and includes
        # lateral and turning commands; none of that has a student counterpart
        # yet, and every episode spent there is experience the transfer discards.
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (0.0, FROUDE_MATCHED_SPEED)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        ranges.heading = (0.0, 0.0)


@configclass
class G1WalkTeacherEnvCfg_PLAY(G1WalkTeacherEnvCfg):
    """Small, deterministic variant for replay and for recording teacher rollouts."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # One fixed speed, so recorded rollouts are comparable to each other.
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (FROUDE_MATCHED_SPEED, FROUDE_MATCHED_SPEED)

        self.events.base_external_force_torque = None
        self.events.push_robot = None
