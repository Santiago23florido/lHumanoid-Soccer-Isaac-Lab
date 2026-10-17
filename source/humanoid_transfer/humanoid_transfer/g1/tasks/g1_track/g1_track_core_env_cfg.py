"""The G1 tracker reading its plan with decoder v2's deep-learning core (skill_space.md §9, F4).

Everything of the plain tracker (:class:`~.g1_track_env_cfg.G1TrackEnvCfg`) or of the
physics one (:class:`~.g1_track_physics_env_cfg.G1TrackPhysicsEnvCfg`), with one change:
the policy group is laid out for the core -- the plan as tokens first, the body's
proprioception with a history, then ``ẑ*``
(:mod:`~humanoid_transfer.transfer.tasks.tracking.core_cfg`) -- and the plain group stays
as ``plain`` for an incumbent distilled into it.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from ....transfer.tasks.tracking.core_cfg import add_core_observations
from .g1_track_env_cfg import COMMAND, G1TrackEnvCfg, G1TrackEnvCfg_PLAY
from .g1_track_physics_env_cfg import G1TrackPhysicsEnvCfg, G1TrackPhysicsEnvCfg_PLAY


@configclass
class G1TrackCoreEnvCfg(G1TrackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        add_core_observations(self.observations, COMMAND)


@configclass
class G1TrackCoreEnvCfg_PLAY(G1TrackEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        add_core_observations(self.observations, COMMAND)


@configclass
class G1TrackPhysicsCoreEnvCfg(G1TrackPhysicsEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        add_core_observations(self.observations, COMMAND)


@configclass
class G1TrackPhysicsCoreEnvCfg_PLAY(G1TrackPhysicsEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        add_core_observations(self.observations, COMMAND)
