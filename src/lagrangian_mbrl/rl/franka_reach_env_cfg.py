"""Franka Panda *reach* task — the RL problem definition (robot + MDP).

This is the "exact robot design ready for simulation" for the model-free RL
track. It specializes Isaac Lab's tested :class:`ReachEnvCfg` for the
**Franka Emika Panda** and tunes the scene for a single 8 GB GPU (e.g. an
RTX 4070 Laptop), where the stock ``num_envs=4096`` would not fit.

The MDP, at a glance
--------------------
- **Robot:** Franka Panda 7-DoF arm (``FRANKA_PANDA_CFG`` — the standard,
  physically-calibrated articulation: joint limits, drive gains, masses).
- **Action (7-dim):** scaled joint-position targets on ``panda_joint.*`` around
  the default pose. Joint-position control is the well-behaved starting point
  for model-free RL; a joint-torque variant is provided for parity with the
  Lagrangian-dynamics (MBRL) track.
- **Observation:** joint positions & velocities (relative to default), the
  commanded end-effector pose, and the previous action.
- **Reward:** end-effector position tracking (coarse L2 + fine-grained tanh) and
  orientation tracking, with small action-rate and joint-velocity penalties
  (curriculum-annealed). Higher is better.
- **Termination:** episode time-out only (a continuing reach-and-hold task).

Why subclass instead of re-deriving the scene
----------------------------------------------
Nobody hand-writes the Panda USD / actuator model; the canonical, calibrated
config ships with Isaac Lab as ``FRANKA_PANDA_CFG``. Subclassing the tested
``ReachEnvCfg`` keeps us on the maintained code path while making every
Franka-specific choice explicit and easy to edit below.
"""

from __future__ import annotations

import math

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# Pre-defined, physically-calibrated Franka Panda articulation.
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

# End-effector body used for the reach command and tracking rewards.
EE_BODY_NAME = "panda_hand"

# Conservative default for an 8 GB laptop GPU. Scale up after checking VRAM.
# Override at the CLI with ``--num_envs`` once you know your headroom.
DEFAULT_NUM_ENVS = 256


@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    """Franka Panda reach task (joint-position control), tuned for 8 GB GPUs."""

    def __post_init__(self):
        # Build the generic reach MDP first.
        super().__post_init__()

        # --- scene: drop the Franka into each env, size for an 8 GB GPU ---
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = DEFAULT_NUM_ENVS

        # --- rewards: track the Franka end-effector body ---
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [EE_BODY_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [
            EE_BODY_NAME
        ]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [
            EE_BODY_NAME
        ]

        # --- action: scaled joint-position targets around the default pose ---
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # --- command: poses are tracked at the end-effector; it points along +z ---
        self.commands.ee_pose.body_name = EE_BODY_NAME
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    """Lightweight variant for visualizing / evaluating a trained policy."""

    def __post_init__(self):
        super().__post_init__()
        # Few envs, no observation noise — for clean rollouts / videos.
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
