"""PPO for the oracle: Isaac Lab's G1 velocity runners, with the NAO's exploration noise.

``G1RoughPPORunnerCfg`` and ``G1FlatPPORunnerCfg`` (networks, iterations, every
hyperparameter), in rsl-rl 5's model configs. One value is the NAO's, as for its
trackers (``nao/tasks/nao_track/agents/rsl_rl_ppo_cfg.py``): ``init_std = 0.5``, since a
unit of exploration noise on 0.25 rad actions knocks the NAO over before it learns.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


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
