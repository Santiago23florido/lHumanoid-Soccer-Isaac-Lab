"""Configuration for the NAO standing-balance task.

The task: keep the NAO upright, under random pushes whose magnitude grows as
training proceeds. It is the smallest problem that exercises the whole learning
stack and still has something for a policy to learn that a fixed feedback law
cannot do.

Every number here that could have been guessed is instead derived from the
robot, in :mod:`humanoid_transfer.nao.assets.nao_kinematics`. The ones that matter
most:

* :data:`~humanoid_transfer.nao.assets.nao.NAO_STAND_LIPM_OMEGA` = 6.04 rad/s, the
  inverted-pendulum frequency. An uncorrected balance error doubles in 115 ms,
  which is what forces the control rate.
* The zero-step capturable velocity, 0.554 m/s forward. Pushes beyond it cannot
  be recovered without stepping, so the curriculum stops there: there is nothing
  to learn from a perturbation no controller could survive.

See ``docs/task_nao_stand.md`` for the task description and ``docs/stand_nao.md`` for
the non-learning baseline this is measured against.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from humanoid_transfer.nao.assets import nao_kinematics as nk
from humanoid_transfer.nao.assets.nao import (
    NAO_STAND_CFG,
    NAO_STAND_COM_HEIGHT,
    NAO_STAND_LIPM_OMEGA,
    NAO_STAND_SPAWN_HEIGHT,
)

##
# Derived quantities the configuration is sized from
##

_CAPTURABLE = nk.capturable_com_velocity()
"""Zero-step capturable velocities at the nominal posture, from the URDF."""

_SUPPORT_X, _SUPPORT_Y = nk.support_polygon_double_stance()
"""Double-stance support polygon, in metres, about the midpoint of the soles."""

TORSO_BODY = "base_link"
"""Articulation body carrying the torso mass and geometry.

The URDF's ``base_link`` has no inertial of its own and reaches the torso
through a fixed joint, so the URDF-to-USD conversion merges the two. The
articulation therefore exposes 43 bodies with **no** link named ``torso``,
and asking for one raises at scene creation. Any name used here has to be an
articulation body name, which is not always the URDF link name.
"""


@configclass
class EventCfg:
    """Domain randomisation.

    The policy is trained on a distribution of robots rather than one robot, so
    that it learns a feedback law instead of a trajectory that happens to fit
    this particular set of inertias. That matters here more than usual: the
    upstream URDF's inertials are approximate, the collision geometry is convex
    hulls, and no actuator was ever measured on hardware. Randomising over those
    uncertainties is the honest response to not knowing them.

    Pushes are deliberately *not* an event term. They are applied by the
    environment itself so their magnitude can follow a curriculum and be fed to
    the critic, neither of which an event term can do.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # A polished lab floor against a rubber sole spans roughly this.
            # Below about 0.4 the NAO cannot generate the shear it needs and
            # the task stops being about balance.
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=TORSO_BODY),
            # +/- 300 g on the 1.05 kg torso, which is the scale of a battery,
            # a cable loom or an unmodelled cover.
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # About 3 degrees of joint scatter, so the episode never starts from
            # exactly the same posture and the policy cannot memorise one.
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                # Up to about 3 degrees of initial lean, in both planes.
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class NaoStandEnvCfg(DirectRLEnvCfg):
    """Direct RL configuration for NAO standing balance."""

    # -- timing ------------------------------------------------------------
    decimation = 2
    """Physics steps per policy step, giving a 100 Hz control loop.

    Two constraints meet here. The balance error doubles every 115 ms, so the
    loop has to be far faster than that; and the real NAO's DCM runs its joint
    controllers at exactly 100 Hz, so going faster would give the policy a
    bandwidth the hardware could not reproduce. 100 Hz satisfies both, leaving
    about eleven actions per doubling time.
    """

    episode_length_s = 20.0
    """Episode length in seconds, so 2000 control steps.

    Long enough for several pushes and the recovery after each, short enough
    that a failure is seen quickly in the return.
    """

    is_finite_horizon = False
    """Running out of time is truncation, not failure.

    With this False the wrapper emits ``time_outs``, and the algorithm
    bootstraps the value function at the final state instead of treating the
    end of the episode as though the robot had died there. Getting this wrong
    silently teaches the policy that surviving to 20 s is worth nothing.
    """

    # -- spaces ------------------------------------------------------------
    action_space = 19
    """Joint position offsets for the 11 leg and 8 arm degrees of freedom.

    ``RHipYawPitch`` is absent because one motor drives both hip yaw-pitch
    joints on the real robot. Head, wrists and hands are held by their own PD
    and are not worth the exploration noise.
    """

    observation_space = 65
    """What a real NAO could measure: IMU, joint encoders, foot contact.

    3 projected gravity + 3 angular velocity + 19 joint offsets + 19 joint
    velocities + 19 previous actions + 2 foot contacts. Deliberately nothing
    the hardware could not provide, so the policy stays deployable.
    """

    state_space = 17
    """Privileged state, for the critic only.

    3 base linear velocity + 2 CoM offset + 2 CoM velocity + 2 DCM offset +
    2 foot normal forces + 3 applied push + 1 base height + 1 CoM height +
    1 DCM margin. An asymmetric critic cuts value-estimation variance without
    biasing the policy, because none of this reaches the actor.
    """

    action_clip = 3.0
    """Bound applied to the raw policy output before scaling.

    A Gaussian policy is unbounded, so a tail sample can ask for a joint angle
    far outside the range. Three standard deviations at the initial noise level
    covers the useful output and discards only the tail that the joint limits
    would clamp away anyway.
    """

    action_scale = 0.25
    """Radians of joint offset per unit action.

    Small on purpose. The action is a displacement from the nominal posture, so
    zero is already the stabilising baseline controller; the policy only has to
    learn the correction. Balance corrections are a few degrees, and 0.25 rad
    of authority covers them with room to spare while keeping the policy away
    from the joint stops.
    """

    # -- simulation --------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        # The NAO is 0.58 m tall and never leaves its own tile in this task,
        # so the environments can sit closer together than a walking task needs.
        env_spacing=2.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = NAO_STAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )

    events: EventCfg = EventCfg()

    # -- termination -------------------------------------------------------
    termination_height = 0.20
    """Base height below which the episode ends, in metres.

    The nominal standing height is 0.321 m, so this is a 12 cm drop: past any
    balance excursion and reached well before the torso hits the ground.
    """

    termination_tilt = -0.7
    termination_foot_displacement = 0.05
    """Metres a foot may travel before the episode counts as failed.

    This makes zero-step balance a *constraint* rather than a price. Soft
    penalties on lifting and moving the feet were tried first and did not work:
    the policy kept stepping in 51 of 51 survivors, because per second of
    episode the penalty cost about -1.6 against the +4.5 of staying alive,
    while falling costs all of it. Paying was rational.

    A penalty prices a behaviour. A termination defines the task. What was
    wanted here was the second.

    50 mm against a 164 mm foot, and against the 4.2 mm the joint PD drifts
    under a push it survives, so there is an order of magnitude between drift
    and a step.
    """

    """``projected_gravity_b[2]`` above which the episode ends.

    Gravity in the base frame reads (0, 0, -1) when upright, so this is a tilt
    of about 45 degrees. Beyond it the feet cannot produce a restoring moment
    and the outcome is decided.
    """

    # -- perturbation curriculum ------------------------------------------
    push_interval_s = 2.0
    """Mean seconds between pushes.

    About ten pushes per episode, spaced well beyond the roughly one second a
    recovery takes, so the policy sees each disturbance settle.
    """

    push_fraction_initial = 0.15
    """Opening push, as a fraction of what is recoverable in that direction."""

    push_fraction_final = 0.90
    """Final push, as a fraction of the directional capturable bound.

    A fraction rather than a speed, because the bound is not the same in every
    direction: 0.548 m/s forward, 0.443 m/s backward, up to 0.827 m/s diagonally.
    The first curriculum ramped to a fixed 0.548 m/s in every direction, which
    is 124% of what is recoverable backward, so roughly half the pushes landed
    where no zero-step controller could have survived.

    The cost was visible in the learning curve. Episode length peaked at 1629
    steps around iteration 101, at 0.25 m/s, then fell monotonically to 602 as
    the curriculum ramped -- and kept falling for 450 iterations *after* the
    magnitude stopped changing. A policy whose experience is mostly unwinnable
    episodes has no signal separating a good action from a bad one.

    0.90 keeps every push inside the physically recoverable region while
    staying above what the joint PD reaches, which the threshold sweep put at
    0.73 of the bound forward and 0.90 backward.
    """

    nominal_com_offset: tuple[float, float] = (
        float(
            nk.center_of_mass(nk.NOMINAL_STAND_JOINT_POS)[0]
            - 0.5
            * (
                nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)["l_sole"][0, 3]
                + nk.forward_kinematics(nk.NOMINAL_STAND_JOINT_POS)["r_sole"][0, 3]
            )
        ),
        0.0,
    )
    """Nominal centre of mass relative to the sole midpoint, in metres.

    Needed to find the distance to the polygon edge along a push direction. The
    lateral component is zero by symmetry.
    """

    push_velocity_initial = 0.05
    """Push magnitude at the start of training, in m/s."""

    push_velocity_final = float(_CAPTURABLE["forward"])
    """Push magnitude at the end of the curriculum, in m/s.

    0.554 m/s: the zero-step capturable limit computed from the support polygon
    and the pendulum frequency. Beyond this the robot must step, which this task
    does not ask of it, so a larger push would only teach it to fall.
    """

    push_direction_rad: float | None = None
    """Fixed push heading relative to the robot's facing, or None for uniform.

    None while training: every direction should be equally likely, or the policy
    learns a lopsided recovery. A fixed angle when measuring a threshold, since
    the capturable bound depends on direction -- 0.548 m/s forward against
    0.443 m/s backward -- and a uniformly sampled push cannot tell you which
    bound a controller actually hit.

    0 is forward, pi is backward, pi/2 is to the robot's left.
    """

    push_exact_magnitude = False
    """Push at exactly the curriculum magnitude instead of uniformly below it.

    False while training, so the policy sees the whole range of severities.
    True when measuring, because a threshold is only meaningful if every
    environment received the same push.
    """

    push_curriculum_steps = 24_000_000
    """Environment steps over which the push grows to its final magnitude.

    Roughly the first 2000 iterations at 4096 environments. Ramping matters:
    starting at full magnitude means the policy falls constantly, sees almost no
    upright states, and never gets the gradient that would teach it to balance.
    """

    # -- reward scales -----------------------------------------------------
    alive_reward_scale = 2.0
    """Paid every step the episode has not ended.

    With early termination this is the main signal: staying upright accumulates
    it, falling cuts it off. The rest of the terms shape *how*.
    """

    dcm_reward_scale = 3.0
    """Keeping the divergent component near the centre of the support polygon.

    The theoretically motivated term. ``xi = x + xdot/omega_0`` is the unstable
    mode of the inverted pendulum; the robot is recoverable exactly while xi can
    be brought back inside the polygon. Rewarding a small xi rewards being
    *recoverable*, which a position-only term cannot express: a centre of mass
    sitting still with the wrong velocity is already falling.
    """

    dcm_reward_sigma = 0.03
    """Width of the DCM kernel, in metres.

    Comparable to the 8.7 mm the baseline holds while standing and well inside
    the 103 mm polygon, so the term still discriminates across the range the
    robot actually operates in.
    """

    upright_reward_scale = -2.0
    """Penalty on the horizontal part of gravity in the base frame.

    Zero when the torso is vertical. Cheap, and it stops the policy from
    discovering that a tilted-but-not-fallen posture also collects the alive
    bonus.
    """

    com_height_reward_scale = 1.0
    """Holding the centre of mass at its nominal height."""

    com_height_reward_sigma = 0.02
    """Width of the CoM height kernel, in metres.

    Without this the policy learns to crouch: a lower centre of mass is easier
    to balance, and nothing else in the reward objects.
    """

    posture_reward_scale = 0.5
    """Staying near the nominal joint configuration."""

    posture_reward_sigma = 0.5
    """Width of the posture kernel, in radians.

    Loose on purpose. It is a regulariser that keeps the robot in a recognisable
    standing pose, not a trajectory to track; too tight and it would forbid the
    hip and arm strategies the task exists to discover.
    """

    joint_torque_reward_scale = -1.0e-3
    """Penalty on squared applied torque, for energy and hardware wear."""

    joint_accel_reward_scale = -2.5e-7
    """Penalty on squared joint acceleration, which suppresses buzzing."""

    joint_vel_reward_scale = -1.0e-3
    """Penalty on squared joint velocity, rewarding stillness."""

    action_rate_reward_scale = -0.01
    """Penalty on the change in action between control steps.

    At 100 Hz a policy can chatter at frequencies no real actuator would follow.
    This is what keeps the commanded trajectory smooth enough to deploy.
    """

    foot_slip_reward_scale = -0.5
    """Penalty on horizontal foot velocity while that foot is loaded.

    Friction is finite, so a solution that slides the feet works in simulation
    and not on a real floor.
    """

    undesired_contact_reward_scale = -2.0
    foot_lift_reward_scale = -1.0
    """Penalty per foot that is not carrying load.

    Zero-step balance means the contact configuration does not change. Without
    this the task is not a standing task at all: the first trained policy solved
    it by stepping, every survivor of a backward push moving a foot by up to
    762 mm, because nothing made that cost anything. ``foot_slip`` penalises
    horizontal velocity only while a foot is *loaded*, so lifting it was free.

    At -1.0 one raised foot costs half the alive bonus per step, which makes a
    step expensive without making it impossible -- the policy can still pay for
    one if the alternative is falling.
    """

    foot_displacement_reward_scale = -4.0
    """Penalty on how far each foot has moved from where the episode started.

    Catches what the lift penalty alone would miss: a foot that slides or that
    lifts and lands somewhere new. Together they define "the feet stayed put",
    which is the contact assumption the capturability bound rests on and the
    thing that has to hold before that bound can be compared against.
    """

    """Penalty for any body other than the feet touching the ground."""

    joint_limit_reward_scale = -1.0
    """Penalty for driving joints past their soft limits.

    The soft limits sit at 90% of each range, so this bites before the simulator
    clamps and the policy stops receiving gradient.
    """

    # -- observation and reward constants ---------------------------------
    com_height_target = float(NAO_STAND_COM_HEIGHT)
    """Nominal centre-of-mass height above the soles, 0.2689 m."""

    lipm_omega = float(NAO_STAND_LIPM_OMEGA)
    """Inverted-pendulum frequency, 6.04 rad/s."""

    spawn_height = float(NAO_STAND_SPAWN_HEIGHT)
    """Root height at reset, in metres."""

    support_polygon_x: tuple[float, float] = _SUPPORT_X
    """Forward extent of the double-stance polygon, in metres."""

    support_polygon_y: tuple[float, float] = _SUPPORT_Y
    """Lateral extent of the double-stance polygon, in metres."""

    contact_force_threshold = 1.0
    """Newtons above which a body counts as touching something."""

    foot_body_names: tuple[str, ...] = nk.FOOT_BODIES
    """Bodies carrying the foot collision geometry, after the sole frames merge."""

    undesired_contact_body_names: tuple[str, ...] = (
        TORSO_BODY,
        "Head",
        "Neck",
        ".*Bicep",
        ".*ForeArm",
        ".*_wrist",
        ".*Tibia",
    )
    """Bodies that should never touch the ground while balancing.

    Names are the *articulation's*, not the URDF's. The shins are included
    because a robot that saves itself by kneeling has not balanced, and the
    height and tilt thresholds alone would not always catch it.
    """

    actuated_joint_names: tuple[str, ...] = nk.ACTUATED_JOINTS
    """The 19 joints the policy commands, in action order."""


@configclass
class NaoStandEnvPlayCfg(NaoStandEnvCfg):
    """Smaller, deterministic variant for inspecting a trained policy."""

    def __post_init__(self) -> None:
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        # Show the policy at the hardest perturbation it was trained for,
        # rather than at whatever point a curriculum would have reached.
        self.push_curriculum_steps = 1
        self.events.torso_mass = None
