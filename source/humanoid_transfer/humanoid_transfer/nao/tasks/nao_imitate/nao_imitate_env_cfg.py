"""The joint-target baseline (B3 of ``transfer/paper/README.md``): the NAO imitating the G1.

The G1 teacher's joint trajectories, mapped joint by joint onto the NAO and slowed to
the NAO's clock (``transfer/retarget.py``, ``retarget_reference.py``), are a reference
the NAO is trained to imitate, DeepMimic-style: reference state initialisation, and a
reward for the pose, the joint rates and the base's velocity. This is the joint signal
the zero-shot study replaces with ``z*``, built to work as well as it can.

* **Reference.** ``HUMANOID_REFERENCE``: a retargeted archive of the teacher's
  task-side data (its DART capture over the command range). At test time, the
  teacher's own play captures at the L4 commands, retargeted the same way.
* **Rewards.** DeepMimic's terms with its weights (pose 0.65, joint rates 0.1, the
  centre of mass's velocity 0.1, here the base's, with its yaw rate 0.1; no end
  effectors: the signal is the joints), its kernels as they read at the G1's size,
  converted to the NAO's units; and the zero-shot trackers' body terms -- the fall
  penalty, action rate, joint limits.
* **Body.** The zero-shot trackers' and the oracle's: 50 Hz, offsets of 0.25 rad on
  the 19 actuated joints, the same plane, the same falls. The offsets are about the
  reference's next row instead of the nominal crouch, so the zero action plays the
  reference back and the policy learns the feedback around it: the baseline gets the
  joint signal in its most favourable form. Like the oracle, the policy also observes
  its base velocity.
"""

from __future__ import annotations

import os
from pathlib import Path

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

from ....common.similarity import ANGULAR_VELOCITY, VELOCITY
from ...assets import nao_kinematics as nk
from ...assets.nao import get_nao_stand_cfg
from ..nao_velocity.nao_velocity_env_cfg import similar
from .reference import (
    ReferenceJointPositionActionCfg,
    ReferenceMotionCommandCfg,
    pose_tracking,
    rate_tracking,
    reference_ended,
    reference_joints,
)

REPO_ROOT = Path(__file__).resolve().parents[6]
REFERENCE = os.environ.get(
    "HUMANOID_REFERENCE", str(REPO_ROOT / "nao" / "rollouts" / "retarget_flat_dart_offset.npz")
)
COMMAND = "base_velocity"
ACTUATED = SceneEntityCfg("robot", joint_names=list(nk.ACTUATED_JOINTS), preserve_order=True)

DRIFT = similar(0.1, VELOCITY)
"""The oracle's base-velocity noise (``nao_velocity_env_cfg.py``); the rest are the trackers'."""
RATE_SCALE = 0.1 * similar(1.0, ANGULAR_VELOCITY) ** -2
"""DeepMimic's ``exp(-0.1 Σ Δq̇²)``, with ``Δq̇`` read at the G1's size."""


@configclass
class NaoImitatePolicyCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-DRIFT, n_max=DRIFT))
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
    reference = ObsTerm(func=reference_joints, params={"command_name": COMMAND})
    reference_velocity = ObsTerm(func=mdp.generated_commands, params={"command_name": COMMAND})

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class NaoImitateCriticCfg(NaoImitatePolicyCfg):
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class NaoImitateObservationsCfg:
    policy: NaoImitatePolicyCfg = NaoImitatePolicyCfg()
    critic: NaoImitateCriticCfg = NaoImitateCriticCfg()


@configclass
class NaoImitateRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    pose = RewTerm(func=pose_tracking, weight=0.65, params={"command_name": COMMAND, "scale": 2.0})
    rates = RewTerm(
        func=rate_tracking, weight=0.1, params={"command_name": COMMAND, "scale": RATE_SCALE}
    )
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.1,
        # DeepMimic's exp(-10 Δv²) for the centre of mass, at the G1's size.
        params={"command_name": COMMAND, "std": similar(10.0**-0.5, VELOCITY)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.1,
        # Isaac Lab's yaw-rate kernel, std 0.5 rad/s at the G1's size.
        params={"command_name": COMMAND, "std": similar(0.5, ANGULAR_VELOCITY)},
    )


@configclass
class NaoImitateEnvCfg(LocomotionVelocityRoughEnvCfg):
    """The NAO rewarded for reaching the G1's retargeted joint trajectories."""

    rewards: NaoImitateRewards = NaoImitateRewards()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Scene: the NAO on the trackers' plane.
        self.scene.robot = get_nao_stand_cfg(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.6
        self.scene.height_scanner = None
        self.scene.contact_forces.track_air_time = False
        self.curriculum.terrain_levels = None

        # Actions: the trackers' offsets, about the reference's next row.
        self.actions.joint_pos = ReferenceJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(nk.ACTUATED_JOINTS),
            scale=0.25,
            preserve_order=True,
            command_name=COMMAND,
        )
        self.commands.base_velocity = ReferenceMotionCommandCfg(
            asset_name="robot", reference=REFERENCE, debug_vis=False
        )
        self.observations = NaoImitateObservationsCfg()

        # Rewards: the reference's, and the trackers' body terms.
        rewards = self.rewards
        for prescribed in (
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

        # Terminations: a fall as the trackers define it; the clip's end is a time-out.
        self.terminations.base_contact = None
        self.terminations.base_low = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.18}
        )
        self.terminations.tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
        self.terminations.reference_ended = DoneTerm(
            func=reference_ended, params={"command_name": COMMAND}, time_out=True
        )

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
class NaoImitateEnvCfg_PLAY(NaoImitateEnvCfg):
    """Replay: each environment follows one clip from its start (``capture_velocity.py``)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.in_order = True
        self.commands.base_velocity.from_start = True
