"""PPO for the G1 tracker.

The algorithm is the teacher's (Isaac Lab's ``G1FlatPPORunnerCfg``) unchanged:
the body side is meant to be learned the way the body learned to walk, not
retuned for the contract. Two things differ, both about the input:

* **Width.** The plan adds 243 inputs to the teacher's 120, so the networks
  are the rough-terrain G1's ``[512, 256, 128]`` instead of the flat one's.
* **Normalisation and an asymmetric critic**, written in the rsl-rl 5 form the
  NAO's task uses: the plan is in normalised units and joint velocities in
  rad/s, so a running normaliser keeps either from dominating; the critic also
  reads the true ``z*`` against the plan.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1TrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "g1_track"

    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
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
    )
