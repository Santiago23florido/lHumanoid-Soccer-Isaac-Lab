"""A contract tracker for any registered body (``transfer/bodies/bodies.json``).

The NAO's tracker (``nao/tasks/nao_track``) written once for every body whose
robot Isaac Lab ships. Everything body-specific comes from the body's measured
card and its registry entry:

* **Robot and actions.** Isaac Lab's articulation (with its own actuator models),
  joint-position offsets around the posture, the scale the body's published
  expert uses.
* **Posture.** ``crouch``: the sagittal chains bent until the CoM stands at the
  contract's mean height in leg lengths, as the NAO was
  (:func:`~humanoid_transfer.transfer.latent.pact.kinematics.sagittal_crouch`);
  ``default``: the body's own default pose, for a body whose legs cannot reach
  the contract's height (Cassie walks at 0.76 L where the G1 walks at 0.95 L).
* **Command.** The task's contract in the body's units; most episodes start inside
  the G1 teacher's motion, seeded from its ``z*`` windows.
* **Rewards.** The contract likelihood and the body's own costs (action rate,
  joint limits, falls, a default posture for the joints ``z*`` leaves free). Nothing
  prescribes a motion.
* **Terminations.** A fall (a contact on the fall body, or tilt past 0.8 rad) and a
  body that has left the task.

The same environment variables as the G1's and NAO's trackers select the
contract, the task set and the training conditions.
"""

from __future__ import annotations

import importlib
import json
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

from ..transfer.embodiment import G1
from ..transfer.latent.pact.kinematics import CardKinematics, sagittal_crouch
from ..transfer.latent.task_set import task_specs, task_weights
from ..transfer.tasks.tracking import (
    OCTAVE_WEIGHTS,
    OCTAVES,
    ReferenceCommandCfg,
    contract_left,
    contract_likelihood,
    contract_path,
    contract_privileged,
    contract_view,
)
from ..transfer.tasks.tracking.contact_reward import contact_schedule

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY = REPO_ROOT / "transfer" / "bodies" / "bodies.json"
DEFAULT_CONTRACT = REPO_ROOT / "logs" / "contract" / "walk_dagger3e" / "generator.pt"
CONTRACT = os.environ.get("HUMANOID_CONTRACT", str(DEFAULT_CONTRACT))
POSTURE_CONTRACT = os.environ.get("HUMANOID_POSTURE_CONTRACT", CONTRACT)
START_WINDOWS = os.environ.get(
    "HUMANOID_START_WINDOWS", str(REPO_ROOT / "g1" / "rollouts" / "teacher_states.npz")
)
PROCEDURAL = float(os.environ.get("HUMANOID_PROCEDURAL", "0"))
CALIBRATION = REPO_ROOT / "transfer" / "results" / "procedural_calibration.json"
LOAD_SPACE = os.environ.get("HUMANOID_LOAD_SPACE", "log")
RESAMPLE_SPACE = os.environ.get("HUMANOID_RESAMPLE", "linear")
PLAN_CONTACT = os.environ.get("HUMANOID_PLAN_CONTACT", "as_drawn")
OBSERVE_UNITS = os.environ.get("HUMANOID_OBSERVE_UNITS") or None
LOAD_SHAPE = tuple(float(v) for v in os.environ.get("HUMANOID_LOAD_SHAPE", "1,1").split(","))
PROCEDURAL_DUTY = (
    tuple(float(v) for v in os.environ["HUMANOID_PROCEDURAL_DUTY"].split(","))
    if os.environ.get("HUMANOID_PROCEDURAL_DUTY")
    else None
)
"""The procedural biped's duty range (``ReferenceCommandCfg.procedural_duty``); unset keeps
the family's, "0.55,0.95" restricts it to walks."""
CONTACT_REWARD = float(os.environ.get("HUMANOID_CONTACT_REWARD", "0"))
"""Weight of the plan's contact schedule in the reward (``tracking/contact_reward.py``); 0: off."""
CURRICULUM = os.environ.get("HUMANOID_CURRICULUM", "0") == "1"
"""Draw the procedural plans by the tracker's contact disagreement (``latent/curriculum.py``);
its record goes to ``HUMANOID_CURRICULUM_LOG`` (an ``.npz``) when that is set."""
CURRICULUM_LOG = os.environ.get("HUMANOID_CURRICULUM_LOG") or None
LOAD_PRECISION = REPO_ROOT / "transfer" / "results" / "load_log_precision_g1.json"
"""A new body has no estimator of its own yet: its reward reads loads at the G1's
measured precision, in the dimensionless log space every body shares."""
TASKS = os.environ.get("HUMANOID_TASKS", "")
TASK_WEIGHTS = task_weights(os.environ.get("HUMANOID_TASK_WEIGHTS", ""))
COMMAND = "base_velocity"
TEACHER_TOP_SPEED = 0.2488351441615056
"""The pinned teacher's forward command cap, m/s (``g1/policy/teacher_policy.json``)."""


def registry_entry(body: str) -> tuple[dict, dict]:
    """The body's registry entry and its measured card."""
    entry = json.loads(REGISTRY.read_text("utf-8"))[body]
    card = json.loads((REPO_ROOT / entry["card"]).read_text("utf-8"))
    return entry, card


def _robot_cfg(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def _sole_corners(foot: dict) -> list[list[float]]:
    low, high = foot["bbox_min_link"], foot["bbox_max_link"]
    return [[x, y, low[2]] for x in (low[0], high[0]) for y in (low[1], high[1])]


def body_posture(body: str, contract: str = POSTURE_CONTRACT) -> dict[str, float]:
    """The joint angles the body tracks around (see the module docstring)."""
    entry, card = registry_entry(body)
    tracker = entry["tracker"]
    names = card["joints"]["names"]
    posture = dict(zip(names, card["joints"]["default_pos"], strict=True))
    if tracker["posture"] == "crouch":
        import torch

        saved = torch.load(contract, map_location="cpu", weights_only=False)
        target = float(saved["generator"]["skill_norm.mean"][0])
        chains = [tuple(chain) for chain in tracker["sagittal_chains"]]
        posture.update(sagittal_crouch(card, chains, target, leg_length=card["leg_length"]))
    return posture


def base_height(body: str, posture: dict[str, float]) -> float:
    """Height of the root above the ground with both soles on it, in this posture."""
    _, card = registry_entry(body)
    kinematics = CardKinematics(card)
    q = [posture[name] for name in kinematics.joint_names]
    return float(-kinematics.soles(q)[:, 2].min())


@configclass
class BodyTrackPolicyCfg(ObsGroup):
    """What the body could sense: IMU, joint encoders, last action -- and the plan."""

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
class BodyTrackCriticCfg(ObsGroup):
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
class BodyTrackObservationsCfg:
    policy: BodyTrackPolicyCfg = BodyTrackPolicyCfg()
    critic: BodyTrackCriticCfg = BodyTrackCriticCfg()


@configclass
class BodyTrackRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_deviation_free = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={})
    contract = RewTerm(
        func=contract_likelihood,
        weight=1.0,
        params={"command_name": COMMAND, "kernels": OCTAVES, "weights": OCTAVE_WEIGHTS},
    )
    contact_schedule = (
        RewTerm(func=contact_schedule, weight=CONTACT_REWARD, params={"command_name": COMMAND})
        if CONTACT_REWARD
        else None
    )


@configclass
class BodyTrackEnvCfg(LocomotionVelocityRoughEnvCfg):
    """A registered body rewarded for reaching the plans of a task's contract."""

    body: str = "h1"
    rewards: BodyTrackRewards = BodyTrackRewards()

    def __post_init__(self) -> None:
        super().__post_init__()
        entry, card = registry_entry(self.body)
        tracker = entry["tracker"]
        leg_length = float(entry.get("leg_length_unit", card["leg_length"]))
        feet = list(entry["foot_bodies"])

        # Scene: the body on flat ground, in its tracking posture.
        posture = body_posture(self.body)
        robot = _robot_cfg(tracker["robot"]).replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.init_state.joint_pos = posture
        robot.init_state.pos = (0.0, 0.0, base_height(self.body, posture) + 0.01)
        self.scene.robot = robot
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.6
        self.scene.height_scanner = None
        self.scene.contact_forces.track_air_time = False
        self.curriculum.terrain_levels = None

        # Actions: offsets around the posture, at the body's published scale.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=float(tracker["action_scale"]),
            use_default_offset=True,
        )

        # Command: the task's contract, in this body's units.
        speed = TEACHER_TOP_SPEED / G1.units.velocity * (9.81 * leg_length) ** 0.5
        self.commands.base_velocity = ReferenceCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=0.2,
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
            base_body=entry["base_body"],
            foot_bodies=feet,
            sole_points=[list(card["foot"][f]["sole_point_link"]) for f in feet],
            sole_corners=[_sole_corners(card["foot"][f]) for f in feet],
            leg_length=leg_length,
            flow_solver="midpoint",
            flow_steps=8,
            half_precision=True,
            start_windows=str(START_WINDOWS),
            start_fraction=0.8,
            autonomous=True,
            resync_error=float("inf"),
            procedural_fraction=PROCEDURAL,
            procedural_calibration=str(CALIBRATION),
            load_space=LOAD_SPACE,
            resample_space=RESAMPLE_SPACE,
            plan_contact=PLAN_CONTACT,
            observe_units=OBSERVE_UNITS,
            procedural_load_shape=LOAD_SHAPE,
            procedural_duty=PROCEDURAL_DUTY,
            procedural_curriculum=CURRICULUM,
            procedural_curriculum_log=CURRICULUM_LOG,
            load_precision=str(LOAD_PRECISION),
            tasks=task_specs(TASKS, REPO_ROOT, own_states=False) if TASKS else None,
            task_weights=TASK_WEIGHTS,
        )

        self.observations = BodyTrackObservationsCfg()

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
        if tracker["free_joints"]:
            rewards.joint_deviation_free.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(tracker["free_joints"])
            )
        else:
            rewards.joint_deviation_free = None

        # Terminations: a fall, or a body that has left the task.
        self.terminations.base_contact.params["sensor_cfg"].body_names = tracker["fall_contact"]
        self.terminations.tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
        self.terminations.contract_left = DoneTerm(
            func=contract_left, params={"command_name": COMMAND, "lag": 0.5}
        )

        # Events: no pushes or mass changes; resets in the tracking posture.
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
class BodyTrackEnvCfg_PLAY(BodyTrackEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        command = self.commands.base_velocity
        command.ranges.lin_vel_x = (command.ranges.lin_vel_x[1],) * 2
        command.resampling_time_range = (10.0, 10.0)
        command.rel_standing_envs = 0.02
        command.start_fraction = 0.0
        command.procedural_fraction = 0.0
        self.terminations.contract_left = None


@configclass
class H1TrackEnvCfg(BodyTrackEnvCfg):
    body: str = "h1"


@configclass
class H1TrackEnvCfg_PLAY(BodyTrackEnvCfg_PLAY):
    body: str = "h1"


@configclass
class CassieTrackEnvCfg(BodyTrackEnvCfg):
    body: str = "cassie"


@configclass
class CassieTrackEnvCfg_PLAY(BodyTrackEnvCfg_PLAY):
    body: str = "cassie"
