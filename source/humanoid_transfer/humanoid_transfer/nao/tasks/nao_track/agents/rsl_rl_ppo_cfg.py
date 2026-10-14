"""PPO for the NAO tracker: the G1 tracker's, with the NAO's own exploration noise.

The algorithm and networks are the G1 tracker's, so the two bodies learn the
contract the same way. One value comes from the NAO's standing study instead:
``init_std = 0.5``. Actions are offsets of 0.25 rad per unit, and a unit of
exploration noise knocks the NAO over before it learns anything (see
``nao/tasks/nao_stand/agents/rsl_rl_ppo_cfg.py``).

With ``HUMANOID_SYMMETRY=1`` each minibatch is doubled with its mirror image, as
the G1's (``g1/tasks/g1_track/agents/rsl_rl_ppo_cfg.py``).
"""

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mirror import augment

SYMMETRY = os.environ.get("HUMANOID_SYMMETRY", "0") == "1"


@configclass
class NaoTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "nao_track"

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
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
        symmetry_cfg=(
            RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=augment)
            if SYMMETRY
            else None
        ),
    )
