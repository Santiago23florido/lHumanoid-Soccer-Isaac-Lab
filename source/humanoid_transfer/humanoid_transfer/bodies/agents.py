"""PPO for a registered body's contract tracker: the G1 tracker's algorithm and networks.

The body's measured mirror (``transfer/bodies/<body>/mirror.json``, from
``transfer/scripts/measure_mirror.py``) is bound in a module-level function per body,
because rsl-rl resolves the augmentation function as a callable. ``HUMANOID_SYMMETRY``
selects ``augment`` or ``loss`` as for the G1 and the NAO.
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..transfer.tasks.tracking.symmetry import mirror_augmentation

REPO_ROOT = Path(__file__).resolve().parents[4]
SYMMETRY = os.environ.get("HUMANOID_SYMMETRY", "")


ENTROPY = float(os.environ.get("HUMANOID_ENTROPY", "0.008"))
"""PPO's entropy coefficient. The bonus ``entropy_coef * H`` pushes every action's log std up
by a constant that does not shrink as the policy converges, while the policy gradient's push
down does (the advantages stop correlating with the noise), so near convergence the std
climbs: every tracker's did (the H1's 0.82 -> 1.00 over its last 4500 iterations; the NAO's,
initialised at 0.5, is 1.22). ``0`` lets PPO's own gradient set the std, for a fine-tune
judged on contact timing (``nao_stand``'s PPO config documents the same runaway)."""


def _mirror(body: str) -> Path:
    return REPO_ROOT / "transfer" / "bodies" / body / "mirror.json"


def h1_augment(env, obs=None, actions=None):
    """Each H1 sample and its mirror image."""
    return mirror_augmentation(env, obs, actions, mirror=_mirror("h1"))


def cassie_augment(env, obs=None, actions=None):
    """Each Cassie sample and its mirror image."""
    return mirror_augmentation(env, obs, actions, mirror=_mirror("cassie"))


def digit_augment(env, obs=None, actions=None):
    """Each Digit sample and its mirror image."""
    return mirror_augmentation(env, obs, actions, mirror=_mirror("digit"))


def _symmetry(augment):
    if SYMMETRY not in ("augment", "loss"):
        return None
    return RslRlSymmetryCfg(
        use_data_augmentation=SYMMETRY == "augment",
        use_mirror_loss=SYMMETRY == "loss",
        mirror_loss_coeff=1.0,
        data_augmentation_func=augment,
    )


def _algorithm(augment) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=ENTROPY,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=_symmetry(augment),
    )


@configclass
class BodyTrackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "body_track"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)


@configclass
class H1TrackPPORunnerCfg(BodyTrackPPORunnerCfg):
    experiment_name = "h1_track"
    algorithm = _algorithm(h1_augment)


@configclass
class CassieTrackPPORunnerCfg(BodyTrackPPORunnerCfg):
    experiment_name = "cassie_track"
    algorithm = _algorithm(cassie_augment)


@configclass
class DigitTrackPPORunnerCfg(BodyTrackPPORunnerCfg):
    experiment_name = "digit_track"
    algorithm = _algorithm(digit_augment)
