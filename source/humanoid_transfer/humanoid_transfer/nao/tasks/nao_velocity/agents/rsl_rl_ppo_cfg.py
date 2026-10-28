"""PPO for the oracle: Isaac Lab's G1 velocity runners, with the NAO's exploration noise.

``G1RoughPPORunnerCfg`` and ``G1FlatPPORunnerCfg`` (networks, iterations, every
hyperparameter), in rsl-rl 5's model configs. One value is the NAO's, as for its
trackers (``nao/tasks/nao_track/agents/rsl_rl_ppo_cfg.py``): ``init_std = 0.5``, since a
unit of exploration noise on 0.25 rad actions knocks the NAO over before it learns.

And one bound the G1's runners do not set, :data:`ACTION_BOUND`.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

ACTION_BOUND = 100.0
"""Actions are clipped here, where they no longer move the NAO: past ``|a|`` of 48 (the
elbow yaw, whose gain is the lowest) every joint's target is out of its range by more
than its torque limit over its gain, from any position, so the torque is saturated.
Unclipped, the actions are unbounded and the last one is observed: rough seed 43's ran
away to ~1e11 from iteration ~1450, and the action-rate penalty they earned drove the
value loss to infinity (``logs/contract/train_oracle_rough_s43.log``). The trained
oracles' L4 actions stay below 22 (``nao/rollouts/oracle_*.npz``)."""


def _model(hidden: list[int], std: float | None) -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=hidden,
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None
        if std is None
        else RslRlMLPModelCfg.GaussianDistributionCfg(init_std=std),
    )


@configclass
class NaoVelocityRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "nao_velocity_rough"
    clip_actions = ACTION_BOUND

    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _model([512, 256, 128], 0.5)
    critic = _model([512, 256, 128], None)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class NaoVelocityFlatPPORunnerCfg(NaoVelocityRoughPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "nao_velocity_flat"
        self.actor = _model([256, 128, 128], 0.5)
        self.critic = _model([256, 128, 128], None)
