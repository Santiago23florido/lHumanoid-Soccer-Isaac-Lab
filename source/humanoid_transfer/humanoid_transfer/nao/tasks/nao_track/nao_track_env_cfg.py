"""The NAO following the walk's contract: the second body, nothing of the task retrained.

Built from the same pieces as the G1's tracking task
(:mod:`humanoid_transfer.transfer.tasks.tracking`), on Isaac Lab's
locomotion scaffold with the NAO's own body:

* **Robot.** :func:`~humanoid_transfer.nao.assets.get_nao_stand_cfg` -- the
  derived PD gains, the nominal crouch, contact reporting. Nothing of the
  frozen standing task is imported, only the body.
* **Actions.** Offsets on the 19 actuated joints, 0.25 rad per unit, around
  the crouch, as the standing study validated; held joints keep their targets.
* **Command.** The walk's contract generator, unchanged. Its command is
  dimensionless, so the NAO's velocity range is the teacher's expressed in the
  NAO's units: 0.2488 m/s on the G1 is 0.153 m/s here, the Froude-matched
  0.15 m/s the study was designed around. Plans are read at the NAO's control
  instants: at 50 Hz a step is 0.126 ``T`` against the G1's 0.077 ``T``.
* **Rewards.** The same contract terms and weights as the G1's; for the body,
  the same kind of terms -- action rate, joint limits, the fall penalty and a
  default posture for the arms and head the contract leaves free.
* **Terminations.** A fall: the base below 0.18 m or tilted past 0.8 rad.

The generator reads the true ``z*`` plus noise the size of the G1 encoder's
error: there is no NAO encoder yet, and the plan is what the NAO must learn to
follow, whichever estimate later feeds it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from ....transfer.embodiment import G1, NAO
from ....transfer.tasks.tracking import (
    ReferenceCommandCfg,
    contract_privileged,
    contract_tracking,
    contract_view,
)
from ...assets import nao_kinematics as nk
from ...assets.nao import get_nao_stand_cfg

REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CONTRACT = REPO_ROOT / "logs" / "contract" / "walk_starts" / "generator.pt"
CONTRACT = os.environ.get("HUMANOID_CONTRACT", str(DEFAULT_CONTRACT))
START_WINDOWS = REPO_ROOT / "g1" / "rollouts" / "teacher_states.npz"

COMMAND = "base_velocity"
BASE_BODY = "base_link"
FOOT_BODIES = list(nk.FOOT_BODIES)
TEACHER_TOP_SPEED = 0.2488351441615056
"""The pinned teacher's forward command cap, m/s (``g1/policy/teacher_policy.json``)."""

TEACHER_DIMENSIONLESS_SPEED = TEACHER_TOP_SPEED / G1.units.velocity
"""The walk's top command in ``V``, the unit every body shares."""

ACTUATED = SceneEntityCfg("robot", joint_names=list(nk.ACTUATED_JOINTS), preserve_order=True)
FREE_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=list(nk.ARM_ACTUATED_JOINTS) + ["HeadYaw", "HeadPitch"],
    preserve_order=True,
)


def sole_points() -> list[list[float]]:
    """Each sole point in its foot link's frame, from the URDF (the sole frames merge away)."""
    poses = nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)
    return [
        list(map(float, (np.linalg.inv(poses[foot]) @ poses[sole])[:3, 3]))
        for foot, sole in zip(nk.FOOT_BODIES, nk.SOLE_FRAMES, strict=True)
    ]


@configclass
class NaoTrackPolicyCfg(ObsGroup):
    """What a NAO could sense: IMU, joint encoders, last action -- and the plan."""

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": ACTUATED},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED}, noise=Unoise(n_min=-1.5, n_max=1.5)
    )
    actions = ObsTerm(func=mdp.last_action)
    contract = ObsTerm(func=contract_view, params={"command_name": COMMAND})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class NaoTrackCriticCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ACTUATED})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ACTUATED})
    actions = ObsTerm(func=mdp.last_action)
    contract = ObsTerm(func=contract_view, params={"command_name": COMMAND})
    privileged = ObsTerm(func=contract_privileged, params={"command_name": COMMAND})

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class NaoTrackObservationsCfg:
    policy: NaoTrackPolicyCfg = NaoTrackPolicyCfg()
    critic: NaoTrackCriticCfg = NaoTrackCriticCfg()


@configclass
class NaoTrackRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_deviation_free = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": FREE_JOINTS}
    )
    contract = RewTerm(
        func=contract_tracking, weight=1.0, params={"command_name": COMMAND, "kernel": 1.0}
    )
    contract_medium = RewTerm(
        func=contract_tracking, weight=0.5, params={"command_name": COMMAND, "kernel": 10.0}
    )
    contract_coarse = RewTerm(
        func=contract_tracking, weight=0.25, params={"command_name": COMMAND, "kernel": 100.0}
    )


@configclass
class NaoTrackEnvCfg(LocomotionVelocityRoughEnvCfg):
    """The NAO rewarded for reaching the plans of the walk's contract."""

    rewards: NaoTrackRewards = NaoTrackRewards()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Scene: the NAO on flat ground.
        self.scene.robot = get_nao_stand_cfg(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.6
        self.scene.height_scanner = None
        self.scene.contact_forces.track_air_time = False
        self.curriculum.terrain_levels = None

        # Actions: offsets around the crouch on the actuated joints.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(nk.ACTUATED_JOINTS),
            scale=0.25,
            use_default_offset=True,
            preserve_order=True,
        )

        # Command: the walk's contract, in the NAO's units.
        speed = TEACHER_DIMENSIONLESS_SPEED * NAO.units.velocity
        self.commands.base_velocity = ReferenceCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=False,
            ranges=ReferenceCommandCfg.Ranges(
                lin_vel_x=(0.0, speed),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=(0.0, 0.0),
            ),
            contract=CONTRACT,
            base_body=BASE_BODY,
            foot_bodies=FOOT_BODIES,
            sole_points=sole_points(),
            leg_length=NAO.leg_length,
            flow_solver="midpoint",
            flow_steps=8,
            half_precision=True,
            # The NAO has no demonstrations of its own: most episodes begin inside the
            # task's motion as the contract describes it (seeded history, the
            # demonstration's command and CoM velocity in the NAO's units).
            start_windows=str(START_WINDOWS),
            start_fraction=0.8,
            autonomous=True,
            # No catching up with a body that falls behind: that let it veto the task
            # again. A body that leaves the task falls, and a fall ends the episode.
            resync_error=float("inf"),
        )

        self.observations = NaoTrackObservationsCfg()

        # Rewards: nothing that prescribes a motion.
        rewards = self.rewards
        for prescribed in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "feet_air_time",
            "undesired_contacts",
            "flat_orientation_l2",
            "dof_torques_l2",
            "dof_acc_l2",
        ):
            setattr(rewards, prescribed, None)
        rewards.action_rate_l2.weight = -0.005
        rewards.dof_pos_limits.weight = -1.0
        rewards.dof_pos_limits.params["asset_cfg"] = ACTUATED

        # Terminations: a fall.
        self.terminations.base_contact = None
        self.terminations.base_low = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.18}
        )
        self.terminations.tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

        # Events: the G1 teacher's, on the NAO's base.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }


@configclass
class NaoTrackEnvCfg_PLAY(NaoTrackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (command.ranges.lin_vel_x[1],) * 2
        command.start_fraction = 0.0
