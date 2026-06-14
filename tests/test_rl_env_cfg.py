"""Structural tests for the model-free RL task configs.

These require Isaac Lab to be importable and are therefore **skipped** anywhere
it is not installed (e.g. plain CI or the Phase-0 ``.venv`` on Python 3.12).
They check config wiring only — no simulator app is launched.
"""

import pytest

# Skip the whole module unless Isaac Lab is available.
pytest.importorskip("isaaclab", reason="Isaac Lab not installed in this environment.")

try:
    from lagrangian_mbrl.rl.agents.rsl_rl_ppo_cfg import FrankaReachPPORunnerCfg
    from lagrangian_mbrl.rl.franka_reach_env_cfg import (
        DEFAULT_NUM_ENVS,
        EE_BODY_NAME,
        FrankaReachEnvCfg,
        FrankaReachEnvCfg_PLAY,
    )
except Exception as exc:  # pragma: no cover - needs a launched sim app on some setups
    pytest.skip(
        f"Isaac Lab present but cfg import needs a live app: {exc}", allow_module_level=True
    )


def test_env_cfg_targets_franka_end_effector():
    cfg = FrankaReachEnvCfg()
    # robot articulation is wired into the scene
    assert cfg.scene.robot is not None
    # default env count is tuned for an 8 GB GPU
    assert cfg.scene.num_envs == DEFAULT_NUM_ENVS
    # rewards track the Franka end-effector body
    body_names = cfg.rewards.end_effector_position_tracking.params["asset_cfg"].body_names
    assert EE_BODY_NAME in body_names
    # the command is generated for the same end-effector body
    assert cfg.commands.ee_pose.body_name == EE_BODY_NAME


def test_env_cfg_uses_joint_position_action():
    cfg = FrankaReachEnvCfg()
    assert cfg.actions.arm_action is not None
    assert cfg.actions.arm_action.joint_names == ["panda_joint.*"]


def test_play_cfg_is_small_and_clean():
    cfg = FrankaReachEnvCfg_PLAY()
    assert cfg.scene.num_envs == 16
    assert cfg.observations.policy.enable_corruption is False


def test_ppo_cfg_network_and_iterations():
    cfg = FrankaReachPPORunnerCfg()
    assert cfg.actor.hidden_dims == [256, 128, 64]
    assert cfg.critic.hidden_dims == [256, 128, 64]
    assert cfg.actor.activation == "elu"
    assert cfg.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert cfg.max_iterations > 0
