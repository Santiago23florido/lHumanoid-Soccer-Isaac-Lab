"""Direct RL environment for NAO standing balance.

The policy commands joint position offsets on top of the nominal standing
posture, so a zero action is already the stabilising baseline controller from
``scripts/stand_nao.py`` and the policy only has to learn the correction. That
is a deliberately strong prior: exploration starts from a robot that stands
rather than one that falls, which is the difference between a learnable problem
and a needle in a haystack.

The reward is built around the **divergent component of motion**,
:math:`\\xi = x + \\dot{x}/\\omega_0`. The linear inverted pendulum splits into a
stable mode and an unstable one, and :math:`\\xi` is the unstable one:
:math:`\\dot{\\xi} = \\omega_0(\\xi - p)`, where :math:`p` is the centre of
pressure. The robot is recoverable exactly while :math:`\\xi` can be brought
back inside the support polygon, so rewarding a small :math:`\\xi` rewards being
*recoverable*. A reward on the centre of mass alone cannot say this: a centre of
mass sitting still with the wrong velocity is already falling.

Observations are split. The actor sees only what a real NAO could measure --
IMU, joint encoders, foot contact -- so the policy stays deployable. The critic
additionally sees the true centre of mass, the divergent component, the contact
forces and the push that was applied. Asymmetric actor-critic reduces the
variance of the value estimate without biasing the policy, because none of the
privileged state reaches the actor.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .nao_stand_env_cfg import NaoStandEnvCfg


class NaoStandEnv(DirectRLEnv):
    """Keep the NAO upright under growing random pushes."""

    cfg: NaoStandEnvCfg

    def __init__(self, cfg: NaoStandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Resolved by name, never by position: the articulation's joint order is
        # whatever the USD importer produced, and hard-coding it would silently
        # command the wrong joints if the conversion ever changed.
        self._actuated_ids, actuated_names = self._robot.find_joints(
            list(self.cfg.actuated_joint_names), preserve_order=True
        )
        assert actuated_names == list(self.cfg.actuated_joint_names), "joint order not preserved"
        assert len(self._actuated_ids) == self.cfg.action_space, (
            f"{len(self._actuated_ids)} actuated joints but action_space is {self.cfg.action_space}"
        )

        self._foot_ids, _ = self._robot.find_bodies(
            list(self.cfg.foot_body_names), preserve_order=True
        )
        self._contact_foot_ids, _ = self._contact_sensor.find_bodies(
            list(self.cfg.foot_body_names), preserve_order=True
        )
        self._undesired_contact_ids, _ = self._contact_sensor.find_bodies(
            list(self.cfg.undesired_contact_body_names)
        )

        # Masses are constant unless an event randomises them, and the torso
        # event runs once at startup, so the weights are cached after the first
        # read rather than gathered every step across 43 bodies.
        masses = self._robot.data.default_mass.to(self.device)
        self._mass_weights = (masses / masses.sum(dim=1, keepdim=True)).unsqueeze(-1)

        self._nominal_joint_pos = self._robot.data.default_joint_pos[:, self._actuated_ids].clone()

        # Steps until the next push, per environment. Staggered at reset so the
        # whole batch does not get shoved on the same frame.
        self._push_countdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_push = torch.zeros(self.num_envs, 3, device=self.device)

        self._push_interval_steps = max(1, int(self.cfg.push_interval_s / self.step_dt))

        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in (
                "alive",
                "dcm",
                "upright",
                "com_height",
                "posture",
                "joint_torque",
                "joint_accel",
                "joint_vel",
                "action_rate",
                "foot_slip",
                "undesired_contact",
                "joint_limit",
            )
        }

        self._compute_intermediate_values()

    ##
    # Scene
    ##

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    ##
    # Actions
    ##

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._nominal_joint_pos
        )
        self._apply_pushes()

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(
            self._processed_actions, joint_ids=self._actuated_ids
        )

    def _push_magnitude(self) -> float:
        """Current push magnitude in m/s, ramped over the curriculum.

        Starting at full magnitude would have the robot fall almost every
        episode, so the policy would see hardly any upright states and get no
        gradient telling it what balancing looks like. Growing the push keeps
        the task at the edge of what the current policy can already do.
        """
        consumed = self.common_step_counter * self.num_envs
        progress = min(1.0, consumed / max(1, self.cfg.push_curriculum_steps))
        span = self.cfg.push_velocity_final - self.cfg.push_velocity_initial
        return self.cfg.push_velocity_initial + span * progress

    def _apply_pushes(self) -> None:
        """Shove the environments whose timer has expired.

        The push sets root velocity directly, which is the standard stand-in for
        an impulse: an impulse J on a body of mass M is a velocity jump J/M, and
        the contact forces that would have delivered it act over far less than
        one control period anyway.
        """
        self._push_countdown -= 1
        due = self._push_countdown <= 0
        if not bool(due.any()):
            return

        env_ids = due.nonzero(as_tuple=False).squeeze(-1)
        magnitude = self._push_magnitude()

        # Uniform direction in the horizontal plane, so no direction is
        # systematically easier than another.
        angle = math_utils.sample_uniform(-torch.pi, torch.pi, (len(env_ids),), self.device)
        speed = math_utils.sample_uniform(0.0, magnitude, (len(env_ids),), self.device)
        push = torch.zeros(len(env_ids), 3, device=self.device)
        push[:, 0] = speed * torch.cos(angle)
        push[:, 1] = speed * torch.sin(angle)

        velocity = self._robot.data.root_com_vel_w[env_ids].clone()
        velocity[:, :3] += push
        self._robot.write_root_com_velocity_to_sim(velocity, env_ids=env_ids)

        self._last_push[env_ids] = push
        self._push_countdown[env_ids] = self._sample_push_interval(len(env_ids))

    def _sample_push_interval(self, count: int) -> torch.Tensor:
        """Randomised gap before the next push, so the timing is unpredictable."""
        return torch.randint(
            low=self._push_interval_steps // 2,
            high=max(self._push_interval_steps // 2 + 1, 2 * self._push_interval_steps),
            size=(count,),
            device=self.device,
        )

    ##
    # Derived state
    ##

    def _compute_intermediate_values(self) -> None:
        """Recompute the balance quantities the reward and observation share.

        Called from :meth:`_get_dones`, which the base class runs before both
        the reward and the observation, so every consumer sees the same state.
        """
        data = self._robot.data

        # Whole-body centre of mass: Isaac Lab exposes per-body states but no
        # aggregate, so it is assembled here.
        self._com_pos_w = (data.body_com_pos_w * self._mass_weights).sum(dim=1)
        self._com_vel_w = (data.body_com_lin_vel_w * self._mass_weights).sum(dim=1)

        feet_pos = data.body_pos_w[:, self._foot_ids, :]
        self._support_center_w = feet_pos.mean(dim=1)

        # Everything horizontal is expressed in the robot's heading frame, so
        # the policy cannot tell which way it happens to be facing. Yaw is
        # randomised at reset precisely to force that invariance.
        heading = math_utils.yaw_quat(data.root_quat_w)
        com_offset = self._com_pos_w - self._support_center_w
        self._com_offset_b = math_utils.quat_apply_inverse(heading, com_offset)[:, :2]
        self._com_vel_b = math_utils.quat_apply_inverse(heading, self._com_vel_w)[:, :2]

        # Divergent component of motion, the unstable mode of the pendulum.
        self._dcm_b = self._com_offset_b + self._com_vel_b / self.cfg.lipm_omega

        # Height above the ground, not above the ankle body origins. The ankle
        # frame sits 45 mm above the sole, so measuring from it would put the
        # centre of mass 45 mm low against a target defined at the sole -- more
        # than two kernel widths, which silently zeroes the height reward.
        self._com_height = self._com_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        self._dcm_margin = self._polygon_margin(self._dcm_b)

        forces = self._contact_sensor.data.net_forces_w_history
        foot_forces = forces[:, :, self._contact_foot_ids, :].norm(dim=-1).max(dim=1).values
        self._foot_contact = foot_forces > self.cfg.contact_force_threshold
        self._foot_normal_force = foot_forces

    def _polygon_margin(self, point: torch.Tensor) -> torch.Tensor:
        """Signed distance from ``point`` to the support polygon boundary.

        Positive inside, negative outside. The polygon is treated as the
        axis-aligned box the two flat feet span in the heading frame, which is
        the convex hull of the soles to within a millimetre or two.
        """
        x_min, x_max = self.cfg.support_polygon_x
        y_min, y_max = self.cfg.support_polygon_y
        return torch.minimum(
            torch.minimum(point[:, 0] - x_min, x_max - point[:, 0]),
            torch.minimum(point[:, 1] - y_min, y_max - point[:, 1]),
        )

    ##
    # Observations
    ##

    def _get_observations(self) -> dict:
        # Refresh before reading. The base class runs _get_dones, then the
        # reward, then _reset_idx, then this. The values computed in _get_dones
        # describe the state the reward was for, which is correct there and
        # stale here: for any environment that just reset, they still describe
        # the episode that ended. That would hand the policy a large, wrong
        # first observation on every single episode -- a systematic error, not
        # noise, and one that no test would flag.
        self._compute_intermediate_values()

        data = self._robot.data
        joint_offset = data.joint_pos[:, self._actuated_ids] - self._nominal_joint_pos
        joint_vel = data.joint_vel[:, self._actuated_ids]

        policy = torch.cat(
            (
                data.projected_gravity_b,
                data.root_ang_vel_b,
                joint_offset,
                joint_vel,
                self._previous_actions,
                self._foot_contact.float(),
            ),
            dim=-1,
        )

        # Privileged, critic only. Normalising the foot forces by body weight
        # keeps the critic's inputs on a comparable scale to everything else.
        weight = (self._robot.data.default_mass.to(self.device).sum(dim=1) * 9.81).unsqueeze(-1)
        critic = torch.cat(
            (
                data.root_lin_vel_b,
                self._com_offset_b,
                self._com_vel_b,
                self._dcm_b,
                self._foot_normal_force / weight,
                self._last_push,
                (data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]).unsqueeze(-1),
                self._com_height.unsqueeze(-1),
                self._dcm_margin.unsqueeze(-1),
            ),
            dim=-1,
        )

        self._previous_actions = self._actions.clone()
        return {"policy": policy, "critic": critic}

    ##
    # Reward
    ##

    def _get_rewards(self) -> torch.Tensor:
        data = self._robot.data

        # Capturability: how far the unstable mode sits from the middle of the
        # support polygon. This is the term the task is really about.
        dcm_error = torch.sum(torch.square(self._dcm_b), dim=1)
        dcm = torch.exp(-dcm_error / self.cfg.dcm_reward_sigma**2)

        # Torso verticality, from gravity read in the base frame.
        upright = torch.sum(torch.square(data.projected_gravity_b[:, :2]), dim=1)

        height_error = torch.square(self._com_height - self.cfg.com_height_target)
        com_height = torch.exp(-height_error / self.cfg.com_height_reward_sigma**2)

        joint_offset = data.joint_pos[:, self._actuated_ids] - self._nominal_joint_pos
        posture_error = torch.sum(torch.square(joint_offset), dim=1)
        posture = torch.exp(-posture_error / self.cfg.posture_reward_sigma**2)

        joint_torque = torch.sum(torch.square(data.applied_torque[:, self._actuated_ids]), dim=1)
        joint_accel = torch.sum(torch.square(data.joint_acc[:, self._actuated_ids]), dim=1)
        joint_vel = torch.sum(torch.square(data.joint_vel[:, self._actuated_ids]), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # Sliding feet solve the task in simulation and not on a real floor.
        foot_vel = data.body_com_lin_vel_w[:, self._foot_ids, :2].norm(dim=-1)
        foot_slip = torch.sum(foot_vel * self._foot_contact.float(), dim=1)

        forces = self._contact_sensor.data.net_forces_w_history
        undesired = forces[:, :, self._undesired_contact_ids, :].norm(dim=-1).max(dim=1).values
        undesired_contact = torch.sum(
            (undesired > self.cfg.contact_force_threshold).float(), dim=1
        )

        # Soft limits sit at 90% of each range, so this bites before the
        # simulator clamps and the gradient disappears.
        limits = data.soft_joint_pos_limits[:, self._actuated_ids, :]
        position = data.joint_pos[:, self._actuated_ids]
        over = (position - limits[..., 1]).clamp(min=0.0) + (limits[..., 0] - position).clamp(
            min=0.0
        )
        joint_limit = torch.sum(over, dim=1)

        alive = (~self.reset_terminated).float()

        rewards = {
            "alive": alive * self.cfg.alive_reward_scale * self.step_dt,
            "dcm": dcm * self.cfg.dcm_reward_scale * self.step_dt,
            "upright": upright * self.cfg.upright_reward_scale * self.step_dt,
            "com_height": com_height * self.cfg.com_height_reward_scale * self.step_dt,
            "posture": posture * self.cfg.posture_reward_scale * self.step_dt,
            "joint_torque": joint_torque * self.cfg.joint_torque_reward_scale * self.step_dt,
            "joint_accel": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "joint_vel": joint_vel * self.cfg.joint_vel_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "foot_slip": foot_slip * self.cfg.foot_slip_reward_scale * self.step_dt,
            "undesired_contact": (
                undesired_contact * self.cfg.undesired_contact_reward_scale * self.step_dt
            ),
            "joint_limit": joint_limit * self.cfg.joint_limit_reward_scale * self.step_dt,
        }

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    ##
    # Termination
    ##

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Runs before the reward and the observation, so this is where the
        # shared quantities are refreshed.
        self._compute_intermediate_values()

        height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        fallen = height < self.cfg.termination_height
        toppled = self._robot.data.projected_gravity_b[:, 2] > self.cfg.termination_tilt

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fallen | toppled, time_out

    ##
    # Reset
    ##

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)

        # Place the robot at its nominal state *before* the reset events run.
        #
        # Order matters and used to be wrong here. The events own the randomised
        # reset -- mdp.reset_root_state_uniform and mdp.reset_joints_by_offset
        # both build on the defaults and then perturb them -- so writing the
        # defaults afterwards silently threw their work away. It was invisible
        # from the outside: the environment ran, the policy trained, and the
        # domain randomisation simply did not exist. What exposed it was that
        # enabling and disabling randomisation produced identical fall-free
        # rates, which should have been impossible.
        #
        # Writing the nominal state first keeps a defined starting pose when no
        # events are configured, and lets the events perturb it when they are.
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        # default_root_state is expressed in the local environment frame.
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread the resets out so the whole batch does not terminate
            # together and spike the gradient.
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._last_push[env_ids] = 0.0
        self._push_countdown[env_ids] = self._sample_push_interval(len(env_ids))

        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        extras["Curriculum/push_velocity"] = self._push_magnitude()
        extras["Episode_Termination/fall"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"] = dict(extras)
