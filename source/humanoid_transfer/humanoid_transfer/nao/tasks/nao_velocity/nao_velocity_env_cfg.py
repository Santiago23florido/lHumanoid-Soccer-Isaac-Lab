"""The oracle baseline (B2 of ``transfer/paper/README.md``): the NAO trained on the G1's tasks.

The zero-shot claim is that a skill learned on the G1 runs on the NAO without the
NAO ever seeing the task. The oracle is the opposite case: the NAO trained by RL on
the task itself, with the recipe Isaac Lab trained the published G1 policies with
(``isaaclab_tasks/.../velocity/config/g1``). It answers what the NAO can do when it
does see the task.

**The same MDP, in the NAO's units.** Each constant of the recipe that carries a
dimension is converted with :meth:`~...common.similarity.UnitSystem.convert_to`, so a
motion dynamically similar to one of the G1's is commanded, observed and rewarded as
that motion was on the G1:

* *Commands, kernels, times:* velocity ranges and the tracking kernels' widths are
  velocities; turn rates and the heading stiffness are rates; the air-time threshold,
  the episode and the resampling period are durations.
* *Reward weights:* a term ``w f`` enters the return as ``w ∫ f dt``, whose dimension
  is ``f``'s times a time. Each weight is converted by that dimension, so the integral
  over a similar motion is the same on both bodies. The fall penalty is paid once, on
  one control step of 0.02 s on both, and keeps its weight.
* *Rough terrain:* every length of the terrain generator is scaled by the leg-length
  ratio (0.379), slopes kept, and the height scan's grid and offset likewise.

**What stays the NAO's.** The control rate (50 Hz, a body property: the NAO's step is
0.126 ``T`` against the G1's 0.077), the actions (offsets of 0.25 rad on the 19
actuated joints around its nominal crouch), the PPO exploration noise (0.5), the
ground (``NaoTrack``'s plane) and the definition of a fall (the base below 0.18 m or
tilted past 0.8 rad, besides Isaac Lab's torso contact). These are the zero-shot
trackers', so the oracle and the trackers are measured on the same body.

Terms of the G1's recipe for joints the NAO does not have (fingers, torso) are dropped.
The NAO also observes its base velocity, as the published G1 policies do: an oracle.

Tasks: ``NaoVelocityFlat-v0`` (Isaac-Velocity-Flat-G1-v0's recipe) and
``NaoVelocityRough-v0`` (Isaac-Velocity-Rough-G1-v0's, on its terrains, with the height
scan). Their play variants are on the plane, as the published policies are evaluated:
``NaoVelocityRoughOnFlat-Play-v0`` keeps the scanner, which reads a constant there.
"""

from __future__ import annotations

import copy

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity import mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from ....common.similarity import (
    ANGULAR_VELOCITY,
    DIMENSIONLESS,
    LENGTH,
    TIME,
    TORQUE,
    VELOCITY,
    Dimension,
)
from ....transfer.embodiment import G1, NAO
from ...assets import nao_kinematics as nk
from ...assets.nao import get_nao_stand_cfg

BASE_BODY = "base_link"
FEET = "[lr]_ankle"
LEGS = list(nk.LEG_ACTUATED_JOINTS)
ACTUATED = SceneEntityCfg("robot", joint_names=list(nk.ACTUATED_JOINTS), preserve_order=True)
HIP_AND_KNEE = [j for j in LEGS if "Hip" in j or "Knee" in j]
ANGULAR_ACCELERATION = ANGULAR_VELOCITY / TIME
LENGTH_RATIO = NAO.leg_length / G1.leg_length

TERRAIN_LENGTHS = (
    "step_height_range", "step_width", "platform_width", "border_width", "grid_width",
    "grid_height_range", "noise_range", "noise_step", "downsampled_scale",
)  # fmt: skip
"""Fields of Isaac Lab's sub-terrain configs that are lengths, in metres."""


def similar(value: float, dimension: Dimension) -> float:
    """The NAO's value dynamically similar to the G1's ``value``."""
    return float(G1.units.convert_to(NAO.units, value, dimension))


def similar_range(values: tuple[float, float], dimension: Dimension) -> tuple[float, float]:
    return (similar(values[0], dimension), similar(values[1], dimension))


def weight(value: float, dimension: Dimension) -> float:
    """A reward weight on a term of ``dimension``, integrated over time by the return."""
    integrated = dimension * TIME
    return value * G1.units.scale(integrated) / NAO.units.scale(integrated)


def scaled_terrains():
    """Isaac Lab's rough terrains, every length scaled to the NAO."""
    terrains = copy.deepcopy(ROUGH_TERRAINS_CFG)
    terrains.size = tuple(s * LENGTH_RATIO for s in terrains.size)
    terrains.border_width *= LENGTH_RATIO
    terrains.horizontal_scale *= LENGTH_RATIO
    terrains.vertical_scale *= LENGTH_RATIO
    for sub in terrains.sub_terrains.values():
        for name in TERRAIN_LENGTHS:
            value = getattr(sub, name, None)
            if isinstance(value, tuple):
                setattr(sub, name, tuple(v * LENGTH_RATIO for v in value))
            elif isinstance(value, float):
                setattr(sub, name, value * LENGTH_RATIO)
    return terrains


def feet_air_time_biped(env, command_name: str, threshold: float, moving: float, sensor_cfg):
    """Isaac Lab's ``feet_air_time_positive_biped`` with its standing threshold a parameter.

    The original zeroes the reward below a command of 0.1 m/s, a constant of the G1's.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact > 0.0
    in_mode = torch.where(in_contact, contact, air)
    single = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single.unsqueeze(-1), in_mode, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    command = env.command_manager.get_command(command_name)[:, :2]
    return reward * (torch.norm(command, dim=1) > moving)


@configclass
class NaoVelocityRewards(RewardsCfg):
    """``G1Rewards``, the NAO's joints and the NAO's units."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=weight(1.0, DIMENSIONLESS),
        params={"command_name": "base_velocity", "std": similar(0.5, VELOCITY)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=weight(2.0, DIMENSIONLESS),
        params={"command_name": "base_velocity", "std": similar(0.5, ANGULAR_VELOCITY)},
    )
    feet_air_time = RewTerm(
        func=feet_air_time_biped,
        weight=weight(0.25, TIME),
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "threshold": similar(0.4, TIME),
            "moving": similar(0.1, VELOCITY),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=weight(-0.1, VELOCITY),
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=weight(-1.0, DIMENSIONLESS),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*AnklePitch", ".*AnkleRoll"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=weight(-0.1, DIMENSIONLESS),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LHipYawPitch", ".*HipRoll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=weight(-0.1, DIMENSIONLESS),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(nk.ARM_ACTUATED_JOINTS))},
    )


@configclass
class NaoVelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """``G1RoughEnvCfg`` on the NAO."""

    rewards: NaoVelocityRewards = NaoVelocityRewards()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Scene: the NAO on the rough terrains, scaled to it.
        self.scene.robot = get_nao_stand_cfg(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = scaled_terrains()
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.6
        scanner = self.scene.height_scanner
        scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + BASE_BODY
        scanner.pattern_cfg.resolution *= LENGTH_RATIO
        scanner.pattern_cfg.size = [s * LENGTH_RATIO for s in scanner.pattern_cfg.size]
        self.episode_length_s = similar(self.episode_length_s, TIME)

        # Actions: the zero-shot trackers' (nao_track_env_cfg.py), around the nominal crouch.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(nk.ACTUATED_JOINTS),
            scale=0.25,
            use_default_offset=True,
            preserve_order=True,
        )

        # Observations: Isaac Lab's, on the actuated joints, noise in the NAO's units.
        policy = self.observations.policy
        drift = similar(0.1, VELOCITY)
        policy.base_lin_vel.noise = Unoise(n_min=-drift, n_max=drift)
        turn = similar(0.2, ANGULAR_VELOCITY)
        policy.base_ang_vel.noise = Unoise(n_min=-turn, n_max=turn)
        policy.joint_pos.params = {"asset_cfg": ACTUATED}
        policy.joint_vel.params = {"asset_cfg": ACTUATED}
        spin = similar(1.5, ANGULAR_VELOCITY)
        policy.joint_vel.noise = Unoise(n_min=-spin, n_max=spin)
        if policy.height_scan is not None:
            policy.height_scan.params["offset"] = similar(0.5, LENGTH)
            policy.height_scan.clip = similar_range((-1.0, 1.0), LENGTH)
            scan = similar(0.1, LENGTH)
            policy.height_scan.noise = Unoise(n_min=-scan, n_max=scan)

        # Events: G1RoughEnvCfg's.
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE_BODY]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        }

        # Rewards: G1RoughEnvCfg's, each weight in the NAO's units.
        rewards = self.rewards
        rewards.lin_vel_z_l2.weight = 0.0
        rewards.undesired_contacts = None
        rewards.ang_vel_xy_l2.weight = weight(-0.05, ANGULAR_VELOCITY * ANGULAR_VELOCITY)
        rewards.flat_orientation_l2.weight = weight(-1.0, DIMENSIONLESS)
        rewards.action_rate_l2.weight = weight(-0.005, DIMENSIONLESS)
        rewards.dof_acc_l2.weight = weight(-1.25e-7, ANGULAR_ACCELERATION * ANGULAR_ACCELERATION)
        rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=HIP_AND_KNEE)
        rewards.dof_torques_l2.weight = weight(-1.5e-7, TORQUE * TORQUE)
        rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEGS)

        # Commands: G1RoughEnvCfg's ranges, in the NAO's units.
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = similar_range((0.0, 1.0), VELOCITY)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = similar_range((-1.0, 1.0), ANGULAR_VELOCITY)
        command.resampling_time_range = similar_range(command.resampling_time_range, TIME)
        command.heading_control_stiffness = similar(
            command.heading_control_stiffness, ANGULAR_VELOCITY
        )
        command.debug_vis = False

        # Terminations: Isaac Lab's torso contact, and a fall as the trackers define it.
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE_BODY
        self.terminations.base_low = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.18}
        )
        self.terminations.tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class NaoVelocityFlatEnvCfg(NaoVelocityRoughEnvCfg):
    """``G1FlatEnvCfg`` on the NAO."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        rewards = self.rewards
        rewards.track_ang_vel_z_exp.weight = weight(1.0, DIMENSIONLESS)
        rewards.lin_vel_z_l2.weight = weight(-0.2, VELOCITY * VELOCITY)
        rewards.action_rate_l2.weight = weight(-0.005, DIMENSIONLESS)
        rewards.dof_acc_l2.weight = weight(-1.0e-7, ANGULAR_ACCELERATION * ANGULAR_ACCELERATION)
        rewards.feet_air_time.weight = weight(0.75, TIME)
        rewards.dof_torques_l2.weight = weight(-2.0e-6, TORQUE * TORQUE)
        rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=HIP_AND_KNEE
        )

        command = self.commands.base_velocity
        command.ranges.lin_vel_y = similar_range((-0.5, 0.5), VELOCITY)


def _play(cfg) -> None:
    """Isaac Lab's play variants: few environments, no noise, no pushes."""
    cfg.scene.num_envs = 32
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class NaoVelocityFlatEnvCfg_PLAY(NaoVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _play(self)


@configclass
class NaoVelocityRoughOnFlatEnvCfg_PLAY(NaoVelocityRoughEnvCfg):
    """The rough oracle on the plane, scanner and all, as ``G1RoughOnFlat-Play-v0``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        _play(self)
