"""PPO for the NAO tracker: the G1 tracker's, with the NAO's own exploration noise.

The algorithm and networks are the G1 tracker's, so the two bodies learn the
contract the same way. One value comes from the NAO's standing study instead:
``init_std = 0.5``. Actions are offsets of 0.25 rad per unit, and a unit of
exploration noise knocks the NAO over before it learns anything (see
``nao/tasks/nao_stand/agents/rsl_rl_ppo_cfg.py``).

With ``HUMANOID_SYMMETRY`` the tracker is trained toward its mirror image, as
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

from ....transfer.tasks.tracking.physics_cfg import with_physics_heads
from ..mirror import augment

SYMMETRY = os.environ.get("HUMANOID_SYMMETRY", "")
"""``augment``: each minibatch doubled with its mirror image; ``loss``: the mirror loss
``‖π(M o) − P π(o)‖²`` on the action means (Mittal et al., 2024). For a policy that is
already trained and asymmetric, augmentation does not correct the means: a reflected
sample's importance ratio ``π(P a | M o) / π(a | o)`` is ``exp(−Σ d² / 2σ²)``, nearly
zero, so PPO's clipped objective can only widen ``σ`` to reach it (the G1's round 3
did, ``σ`` 1.02 → 1.18 in 130 iterations). The loss acts on the means directly."""


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


@configclass
class NaoTrackPhysicsPPORunnerCfg(NaoTrackPPORunnerCfg):
    """The same algorithm for the tracker whose action is a residual on its body's physics."""

    experiment_name = "nao_track_physics"

    def __post_init__(self) -> None:
        with_physics_heads(self)
