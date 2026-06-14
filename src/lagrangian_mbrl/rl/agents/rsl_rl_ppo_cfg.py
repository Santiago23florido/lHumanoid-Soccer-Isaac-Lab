"""PPO agent configuration (RSL-RL) for the Franka reach task.

This defines the *recommended starting network* and PPO hyper-parameters for the
model-free baseline. PPO is the natural first choice for this continuous-control
task: it is on-policy, stable, and the default trainer Isaac Lab ships for the
manipulation suite.

Network (recommended default)
-----------------------------
A small multi-layer perceptron actor-critic with **ELU** activations:

- actor : ``obs -> [256, 128, 64] -> action_mean`` (+ learned state-independent
  log-std, ``init_noise_std=1.0``)
- critic: ``obs -> [256, 128, 64] -> value``

This is a touch wider than Isaac Lab's stock ``[64, 64]`` reach policy — more
capacity to start from, while still tiny and fast on an 8 GB GPU. Reach is an
easy task, so feel free to shrink back to ``[64, 64]`` for faster iteration.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FrankaReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy PPO runner for ``LMBRL-Franka-Reach-v0``."""

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "lmbrl_franka_reach"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
