"""The G1 following a contract: the walk taken back to the body that walked it.

The same robot, actions, terminations and randomisation as the teacher's task
(:class:`~humanoid_transfer.g1.tasks.g1_walk.g1_walk_env_cfg.G1WalkTeacherEnvCfg`),
with three changes, all of them about *what* the body is told and *how* it is
judged:

**Command.** :class:`~humanoid_transfer.transfer.tasks.tracking.ReferenceCommand`
samples the teacher's velocity commands and turns them, through the walk's
contract generator, into plans of ``z*``. The tracker never sees the velocity
command: the plan is the only thing that says what to do.

**Rewards.** Every term that prescribes a motion is removed -- velocity
tracking, foot air time, foot slide, orientation, vertical and roll-pitch
velocity, hip deviation. What is left belongs to the body: effort and
smoothness costs, joint limits, the fall penalty, and a default posture for
the joints the contract leaves free (arms, fingers, torso). The task enters
through one term only, the likelihood of the reached ``z*`` under the plan.

**Observations.** The teacher's proprioception without the command, plus the
plan's view; an asymmetric critic also sees the true ``z*``. The play variant
adds a ``teacher`` group, the teacher's own 123 observations, so captures can
still be labelled and compared with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards

from ....transfer.tasks.tracking import (
    OCTAVE_WEIGHTS,
    OCTAVES,
    ReferenceCommandCfg,
    contract_left,
    contract_likelihood,
    contract_path,
    contract_privileged,
    contract_view,
)
from ..g1_walk.g1_walk_env_cfg import FROUDE_MATCHED_SPEED, G1WalkTeacherEnvCfg

REPO_ROOT = Path(__file__).resolve().parents[6]
BODY_CARD = REPO_ROOT / "g1" / "body" / "body_card.json"
DEFAULT_CONTRACT = REPO_ROOT / "logs" / "contract" / "walk_starts" / "generator.pt"
CONTRACT = os.environ.get("HUMANOID_CONTRACT", str(DEFAULT_CONTRACT))
START_STATES = os.environ.get(
    "HUMANOID_START_STATES", str(REPO_ROOT / "g1" / "rollouts" / "teacher_states.npz")
)
"""The task's generator; ``HUMANOID_CONTRACT`` points the same body at another task's."""

PROCEDURAL = float(os.environ.get("HUMANOID_PROCEDURAL", "0"))
"""Share of training episodes following the point-mass biped's plans instead of the task's
(``transfer/docs/skill_space.md`` §8); 0 keeps the walk alone."""

CALIBRATION = REPO_ROOT / "transfer" / "results" / "procedural_calibration.json"
LOAD_SPACE = os.environ.get("HUMANOID_LOAD_SPACE", "linear")
"""How the reward scores the feet's loads (``ReferenceCommandCfg.load_space``)."""
LOAD_PRECISION = REPO_ROOT / "transfer" / "results" / "load_log_precision_g1.json"

BASE_BODY = "pelvis"
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
COMMAND = "base_velocity"


def _body_card() -> dict:
    return json.loads(BODY_CARD.read_text(encoding="utf-8"))


def _sole_corners(foot: dict) -> list[list[float]]:
    """The four corners of a foot's sole, in its link frame: its collision box's underside."""
    low, high = foot["bbox_min_link"], foot["bbox_max_link"]
    return [[x, y, low[2]] for x in (low[0], high[0]) for y in (low[1], high[1])]


@configclass
class TrackPolicyCfg(ObsGroup):
    """The tracker's input: the teacher's proprioception, the plan, no command."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)
    contract = ObsTerm(func=contract_view, params={"command_name": COMMAND})
    contract_path = ObsTerm(func=contract_path, params={"command_name": COMMAND})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class TrackCriticCfg(ObsGroup):
    """The critic's input: the same without noise, plus the true ``z*`` against the plan."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)
    contract = ObsTerm(func=contract_view, params={"command_name": COMMAND})
    privileged = ObsTerm(func=contract_privileged, params={"command_name": COMMAND})
    contract_path = ObsTerm(func=contract_path, params={"command_name": COMMAND})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TeacherObsCfg(ObsGroup):
    """The teacher's own 123 observations, for labels and comparison only."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": COMMAND})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TrackObservationsCfg:
    policy: TrackPolicyCfg = TrackPolicyCfg()
    critic: TrackCriticCfg = TrackCriticCfg()


@configclass
class TrackPlayObservationsCfg(TrackObservationsCfg):
    teacher: TeacherObsCfg = TeacherObsCfg()


@configclass
class G1TrackRewards(G1Rewards):
    # One likelihood, its error's scale marginalised over octaves (1 to 128 tolerances);
    # it replaced three kernels two decades apart, between which it was flat (§8.4).
    contract = RewTerm(
        func=contract_likelihood,
        weight=1.0,
        params={"command_name": COMMAND, "kernels": OCTAVES, "weights": OCTAVE_WEIGHTS},
    )


@configclass
class G1TrackEnvCfg(G1WalkTeacherEnvCfg):
    """The G1 rewarded for reaching the plans of the walk's contract."""

    rewards: G1TrackRewards = G1TrackRewards()

    def __post_init__(self) -> None:
        super().__post_init__()

        card = _body_card()
        old = self.commands.base_velocity
        self.commands.base_velocity = ReferenceCommandCfg(
            asset_name=old.asset_name,
            # Starts and stops, as often as the contract's data has them: a body that
            # only ever sees steady walking plans never learns to begin walking.
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.2,
            rel_heading_envs=old.rel_heading_envs,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            debug_vis=False,
            ranges=old.ranges,
            contract=CONTRACT,
            base_body=BASE_BODY,
            foot_bodies=list(FOOT_BODIES),
            sole_points=[list(card["foot"][name]["sole_point_link"]) for name in FOOT_BODIES],
            sole_corners=[_sole_corners(card["foot"][name]) for name in FOOT_BODIES],
            leg_length=float(card["leg_length"]),
            flow_solver="midpoint",
            flow_steps=8,
            half_precision=True,
            # Most episodes begin inside the teacher's walk (reference state
            # initialisation); the rest from the default pose, to learn to start.
            start_states=str(START_STATES),
            start_fraction=0.8,
            autonomous=True,
            # No catching up with a body that falls behind: that let it veto the task
            # again. A body that leaves the task falls, and a fall ends the episode.
            resync_error=float("inf"),
            procedural_fraction=PROCEDURAL,
            procedural_calibration=str(CALIBRATION),
            load_space=LOAD_SPACE,
            load_precision=str(LOAD_PRECISION),
        )

        self.observations = TrackObservationsCfg()

        rewards = self.rewards
        for prescribed in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "feet_air_time",
            "feet_slide",
            "flat_orientation_l2",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "joint_deviation_hip",
        ):
            setattr(rewards, prescribed, None)

        # A body that has left the task ends its episode, as a fall does.
        self.terminations.contract_left = DoneTerm(
            func=contract_left, params={"command_name": COMMAND, "lag": 0.5}
        )


@configclass
class G1TrackEnvCfg_PLAY(G1TrackEnvCfg):
    """Replay on the deployed path: the body's encoder feeds the contract."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations = TrackPlayObservationsCfg()
        self.observations.policy.enable_corruption = False

        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (FROUDE_MATCHED_SPEED, FROUDE_MATCHED_SPEED)
        command.use_encoder = True
        command.start_fraction = 0.0
        command.procedural_fraction = 0.0
        # The teacher's own play conditions, so the two captures compare.
        command.resampling_time_range = (10.0, 10.0)
        command.rel_standing_envs = 0.02

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Captures count falls only.
        self.terminations.contract_left = None
