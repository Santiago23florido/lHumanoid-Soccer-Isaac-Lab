"""PPO configuration for the NAO standing task.

Written against the rsl-rl 5.x API. That version removed ``ActorCritic`` and
replaced it with separate ``actor`` and ``critic`` model configs routed by
``obs_groups``; the older ``policy=RslRlPpoActorCriticCfg(...)`` form still runs
only because Isaac Lab rewrites it at load time, with a deprecation warning per
run. Most shipped examples are still on the old form, so this one follows the
``anymal_d`` config, which is the modern template in the tree.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class NaoStandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for balance: short rollouts, long effective horizon, asymmetric critic."""

    num_steps_per_env = 24
    """Control steps per environment per iteration.

    At 100 Hz this is 0.24 s of experience per environment, about two
    inverted-pendulum time constants, so a rollout spans a meaningful slice of
    the dynamics rather than a snapshot.
    """

    max_iterations = 3000
    save_interval = 100
    experiment_name = "nao_stand"

    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    """Asymmetric actor-critic.

    The actor reads only the deployable 65-dimensional observation: IMU, joint
    encoders and foot contacts. The critic reads that plus the 17 privileged
    values -- true centre of mass, divergent component, contact forces, the push
    that was applied -- for 82 inputs.

    This is free variance reduction. Value estimation is a supervised problem
    that only exists during training, so giving it the true state costs nothing
    at deployment and does not bias the policy gradient, because none of the
    privileged state reaches the actor.
    """

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    """Actor network.

    ``obs_normalization`` matters more here than in most locomotion tasks: the
    observation mixes projected gravity in [-1, 1] with joint velocities that
    reach 24 rad/s, and without a running normaliser the large-scale inputs
    would dominate the first layer for most of training.

    ``init_std=0.5`` rather than the usual 1.0. The action is a joint offset
    scaled by 0.25 rad, so 1.0 would start exploration at a quarter radian of
    noise per joint -- roughly 14 degrees -- which knocks the robot over before
    it has learned anything. Half that starts exploration close to the
    stabilising baseline and lets the entropy bonus open it up.
    """

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    """Critic network. No distribution: it outputs a scalar value."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    """PPO hyperparameters.

    ``gamma=0.995`` is the one value that departs from the locomotion default of
    0.99, and it is a consequence of the control rate. The effective horizon is
    ``1/(1-gamma)`` steps, so at 100 Hz, 0.99 would look only 1 s ahead -- about
    nine inverted-pendulum time constants, but barely longer than a single push
    recovery. 0.995 gives 2 s, long enough for the policy to see a recovery
    through to the end and be credited for it.

    ``schedule="adaptive"`` retunes the learning rate each update to hold the
    policy KL divergence near ``desired_kl``. With a curriculum the difficulty
    of the task changes underneath the optimiser, and a fixed rate that suited
    gentle pushes is wrong once they are four times larger.

    ``entropy_coef=0.001``, lowered from 0.005, and the reason is worth keeping
    because it invalidated four training runs before anyone noticed.

    The entropy bonus is ``entropy_coef * H``, a term whose size does not depend
    on the reward. That is fine while the return is large, and it stops being
    fine when the return shrinks. Adding the stepping constraint did exactly
    that: the foot penalties and the early termination cut the mean return from
    116 to 25. The bonus did not shrink with it, so buying entropy became a
    better trade than standing up, and PPO took it -- the action standard
    deviation climbed monotonically from 0.50 to 1.25 and never levelled off.
    On a task where success is measured in millimetres of foot travel, that is
    injected noise, and the policy degrades while the loss curve looks ordinary.

    Four runs make the case, because the one variable that tracks the runaway
    is the return, not the episode length::

        run        mean reward   action std (start -> iter 600)
        seed1          116          0.50 -> 0.64   stable
        nostep          66          0.50 -> 1.14   runaway
        zerostep        27          0.50 -> 1.22   runaway
        dircurr         25          0.50 -> 1.17   runaway

    ``nostep`` is the decisive row: its episodes run to the 2000-step timeout,
    so nothing was cut short, and its std still ran away once the foot penalties
    pulled the return down. Episode length is not the variable. Return is.

    0.001 restores roughly the ratio ``seed1`` had. It is a scale-dependent
    constant, so it has to be revisited whenever the reward terms change; the
    check is one line, ``Mean action std`` must flatten rather than climb.
    """


@configclass
class NaoStandPPOPlayRunnerCfg(NaoStandPPORunnerCfg):
    """Loading a checkpoint for inspection; the runner still wants a full config."""

    def __post_init__(self) -> None:
        self.max_iterations = 1
