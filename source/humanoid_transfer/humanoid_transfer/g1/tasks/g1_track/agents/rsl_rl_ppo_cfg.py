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

With ``HUMANOID_SYMMETRY`` the tracker is trained toward its mirror image
(:mod:`~humanoid_transfer.transfer.tasks.tracking.symmetry`): the family and the
robot are mirror-symmetric, so the optimal tracker is equivariant, and round 2's
was not (its left stance 0.041 of a stride short of its right across the
competence map).
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

SYMMETRY = os.environ.get("HUMANOID_SYMMETRY", "")
"""``augment``: each minibatch doubled with its mirror image; ``loss``: the mirror loss
``‖π(M o) − P π(o)‖²`` on the action means (Mittal et al., 2024). For a policy that is
already trained and asymmetric, augmentation does not correct the means: a reflected
sample's importance ratio ``π(P a | M o) / π(a | o)`` is ``exp(−Σ d² / 2σ²)``, nearly
zero, so PPO's clipped objective can only widen ``σ`` to reach it (the G1's round 3
did, ``σ`` 1.02 → 1.18 in 130 iterations). The loss acts on the means directly."""


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
        symmetry_cfg=(
            RslRlSymmetryCfg(
                use_data_augmentation=SYMMETRY == "augment",
                use_mirror_loss=SYMMETRY == "loss",
                mirror_loss_coeff=1.0,
                data_augmentation_func=augment,
            )
            if SYMMETRY in ("augment", "loss")
            else None
        ),
    )
