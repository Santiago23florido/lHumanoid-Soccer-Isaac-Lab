"""G-N2: add a body to the shared PACT policy by PPO on a few of its parameters.

A body the shared policy was not trained on has no tracker to imitate, so its adaptation
uses the task's own reward: the same Track environment, reward and PPO algorithm its
from-scratch tracker trained with (``pact_train_trackers.sh``), with PACT as the actor
(:class:`~humanoid_transfer.transfer.latent.pact.actor.PactActor`) and only the arm's
parameters tuned (``etune``: the body's ``E_b`` and descriptor residuals; ``bitfit``:
also every bias and LayerNorm parameter; ``full``: everything). The critic is a fresh
MLP on the task's critic observation, warmed up with the actor frozen for
``--critic-warmup`` iterations.

The environment gains an observation group ``pact`` with the encoder's inputs (joint
features and IMU with the observation manager's history, and the anchor inputs), computed
as :class:`~humanoid_transfer.transfer.latent.pact.driver.PactDriver` computes them in
G-D1. Differences from the from-scratch tracker, declared: no mirror loss (its mirror
function does not know the ``pact`` group); the actor's inputs are the simulation's
noiseless readings, as in the captures PACT was trained on.

Usage::

    python transfer/scripts/pact_adapt.py --headless --task H1Track-v0 --body h1 \\
        --pact-model logs/pact/models/<model without h1>.pt --arm etune \\
        --max-iterations 1000 --run-name gn2_h1_etune

Writes rsl-rl's ``model_<it>.pt`` and, beside each, ``pact_<it>.pt``: the adapted policy
in PACT's checkpoint format (``capture_contract.py --pact-model``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
    pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="G-N2: add a body to PACT by PPO.")
parser.add_argument("--task", required=True, help="The body's Track task, e.g. H1Track-v0.")
parser.add_argument("--body", required=True, help="The body's registry name, e.g. h1.")
parser.add_argument("--pact-model", type=Path, required=True)
parser.add_argument("--arm", default="etune", choices=("zero", "etune", "bitfit", "full"))
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--max-iterations", type=int, default=1000)
parser.add_argument("--critic-warmup", type=int, default=20)
parser.add_argument("--init-std", type=float, default=0.2, help="Action noise, action units.")
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--mini-batches", type=int, default=16)
parser.add_argument("--save-interval", type=int, default=50)
parser.add_argument("--seed", type=int, default=41)
parser.add_argument("--run-name", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import humanoid_transfer.tasks  # noqa: E402,F401  (registers the tasks)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from humanoid_transfer.transfer.latent.pact.actor import (  # noqa: E402
    imu_features,
    joint_features,
    load_body,
)
from humanoid_transfer.transfer.latent.pact.anchors import (  # noqa: E402
    DriveLoads,
    KinematicAnchors,
)
from humanoid_transfer.transfer.latent.pact.descriptors import GRAVITY  # noqa: E402
from humanoid_transfer.transfer.latent.pact.driver import anchor_inputs  # noqa: E402
from humanoid_transfer.transfer.latent.pact.drives import (  # noqa: E402
    DRIVE_KEYS,
    drive_parameters,
)
from isaaclab.managers import ManagerTermBase  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from pact_check_anchors import sole_override  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

ACTOR = "humanoid_transfer.transfer.latent.pact.actor:PactActor"
COMMAND = "base_velocity"


class _BodyTerm(ManagerTermBase):
    """A PACT input term: the body's card, units and default pose, read once."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        descriptor, _, self.card, self.unit = load_body(REPO, cfg.params["body"])
        if tuple(self.robot.joint_names) != tuple(descriptor.joint_names):
            raise ValueError(f"{cfg.params['body']}: the card's joints are not the simulation's")
        self.time_unit = float(np.sqrt(descriptor.leg_length / GRAVITY))
        self.q0 = self.robot.data.default_joint_pos[0].clone()

    def reset(self, env_ids=None) -> None:
        pass


class PactJoints(_BodyTerm):
    def __call__(self, env, body: str) -> torch.Tensor:
        d = self.robot.data
        return joint_features(d.joint_pos, d.joint_vel, d.joint_pos_target, self.q0, self.time_unit)


class PactImu(_BodyTerm):
    def __call__(self, env, body: str) -> torch.Tensor:
        d = self.robot.data
        return imu_features(d.projected_gravity_b, d.root_ang_vel_b, self.time_unit)


class PactAnchor(_BodyTerm):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        body = cfg.params["body"]
        self.anchors = KinematicAnchors(self.card, self.unit, sole_override(body))
        self.drive = DriveLoads(self.card, self.anchors)
        drives = drive_parameters(self.robot)
        if drives is None:
            raise ValueError("PACT's drive loads need the drives' gains")
        self.drive.set_drives(*(drives[k] for k in DRIVE_KEYS))
        self.anchors, self.drive = self.anchors.to(env.device), self.drive.to(env.device)

    def __call__(self, env, body: str) -> torch.Tensor:
        d = self.robot.data
        return anchor_inputs(
            self.anchors, self.drive, d.joint_pos, d.joint_vel, d.projected_gravity_b,
            d.root_ang_vel_b, d.joint_pos_target,
        )  # fmt: skip


def pact_group(body: str, history: int):
    """The ``pact`` observation group: the layout ``PactActor`` splits."""

    @configclass
    class PactInputsCfg(ObsGroup):
        joints = ObsTerm(
            func=PactJoints, params={"body": body}, history_length=history,
            flatten_history_dim=True,
        )  # fmt: skip
        imu = ObsTerm(
            func=PactImu, params={"body": body}, history_length=history, flatten_history_dim=True
        )
        anchor = ObsTerm(func=PactAnchor, params={"body": body})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    return PactInputsCfg()


@configclass
class PactActorCfg:
    """rsl-rl builds the actor as ``class_name(obs, obs_groups, "actor", n, **rest)``."""

    class_name: str = ACTOR
    checkpoint: str = ""
    repo: str = ""
    body: str = ""
    view: int = 0
    skill_mean: list = []
    skill_std: list = []
    control_dt: float = 0.0
    action_joints: list = []
    action_scale: float = 1.0
    arm: str = "etune"
    init_std: float = 0.2


def main() -> int:
    checkpoint = torch.load(args.pact_model, map_location="cpu", weights_only=False)
    history = int(checkpoint["dims"]["history"])
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.pact = pact_group(args.body, history)
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    term = inner.command_manager.get_term(COMMAND)
    actions = inner.action_manager
    if len(actions.active_terms) != 1:
        raise ValueError(f"PACT drives one joint-position action, not {actions.active_terms}")
    action = actions.get_term(actions.active_terms[0])
    joints = action._joint_ids
    if isinstance(joints, slice):
        joints = list(range(inner.scene["robot"].num_joints))[joints]
    joints = [int(j) for j in joints]
    scale = action._scale
    scale = float(scale.flatten()[0]) if isinstance(scale, torch.Tensor) else float(scale)
    if isinstance(action._scale, torch.Tensor) and not torch.allclose(
        action._scale, torch.full_like(action._scale, scale)
    ):
        raise ValueError("PACT's actor takes one action scale")
    default = inner.scene["robot"].data.default_joint_pos[0]
    offset = action._offset
    if isinstance(offset, torch.Tensor) and not torch.allclose(offset[0], default[joints]):
        raise ValueError("PACT's targets are relative to the default pose; the action's are not")

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval
    agent_cfg.experiment_name = f"pact_adapt_{args.body}"
    agent_cfg.run_name = args.run_name
    agent_cfg.obs_groups = {"actor": ["policy", "pact"], "critic": ["critic"]}
    agent_cfg.actor = PactActorCfg(
        checkpoint=str(args.pact_model.resolve()),
        repo=str(REPO),
        body=args.body,
        view=int(term.view),
        skill_mean=term.generator.skill_norm.mean.cpu().tolist(),
        skill_std=term.generator.skill_norm.std.cpu().tolist(),
        control_dt=float(inner.step_dt),
        action_joints=joints,
        action_scale=scale,
        arm=args.arm,
        init_std=args.init_std,
    )
    agent_cfg.algorithm.symmetry_cfg = None
    agent_cfg.algorithm.learning_rate = args.learning_rate
    agent_cfg.algorithm.num_mini_batches = args.mini_batches

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = REPO / "logs/rsl_rl" / agent_cfg.experiment_name / f"{stamp}_{args.run_name}"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device="cuda:0")
    actor = runner.alg.actor
    print(
        f"[pact_adapt] {args.body}: arm {args.arm}, {actor.tuned_count()} tuned parameters; "
        f"view {term.view}, {len(joints)} actions, scale {scale}; log {log_dir}"
    )

    start = time.time()
    tuned = [p for p in actor.parameters() if p.requires_grad]
    warmup = min(args.critic_warmup, args.max_iterations)
    if warmup:
        for p in tuned:
            p.requires_grad_(False)
        runner.learn(num_learning_iterations=warmup, init_at_random_ep_len=True)
        for p in tuned:
            p.requires_grad_(True)
        # The adaptive schedule raised the rate while the actor could not move (KL 0).
        runner.alg.learning_rate = args.learning_rate
        for group in runner.alg.optimizer.param_groups:
            group["lr"] = args.learning_rate
    rest = args.max_iterations - warmup
    if rest > 0:
        runner.learn(num_learning_iterations=rest, init_at_random_ep_len=warmup == 0)
    print(f"[pact_adapt] trained in {time.time() - start:.0f} s")

    for saved in sorted(log_dir.glob("model_*.pt")):
        state = torch.load(saved, map_location="cpu", weights_only=False)
        actor.load_state_dict(state["actor_state_dict"])
        out = saved.with_name(saved.name.replace("model_", "pact_"))
        torch.save(actor.adapted_checkpoint(checkpoint), out)
    print(f"[pact_adapt] PACT checkpoints written beside rsl-rl's in {log_dir}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
