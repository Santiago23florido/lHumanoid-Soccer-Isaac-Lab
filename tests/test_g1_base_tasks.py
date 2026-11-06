"""The G1's base tasks (``g1/tasks/g1_base``): Isaac Lab's flat recipe, only the commands changed.

The configuration imports Isaac Lab, so it is read from source, as ``test_g1_teacher.py``
reads the walk teacher's. What matters most is the claim the module makes: a command range
defines each task, and nothing else of the recipe is touched.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_base"
CFG = PACKAGE / "g1_base_env_cfg.py"


def _module() -> ast.Module:
    return ast.parse(CFG.read_text(encoding="utf-8"))


def _constant(name: str):
    for node in _module().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def test_every_task_is_isaac_labs_flat_recipe():
    classes = {n.name: n for n in _module().body if isinstance(n, ast.ClassDef)}
    for name in ("Backward", "Sidestep", "Jog"):
        base = classes[f"G1Base{name}EnvCfg"]
        assert [b.id for b in base.bases] == ["G1FlatEnvCfg"]
        play = classes[f"G1Base{name}EnvCfg_PLAY"]
        assert [b.id for b in play.bases] == [f"G1Base{name}EnvCfg"]


def test_only_commands_and_play_settings_are_assigned():
    """No reward, terrain, action or observation term of the recipe is changed."""
    allowed = {
        "ranges.lin_vel_x",
        "ranges.lin_vel_y",
        "ranges.ang_vel_z",
        "ranges.heading",
        "cfg.scene.num_envs",
        "cfg.scene.env_spacing",
        "cfg.observations.policy.enable_corruption",
        "cfg.events.base_external_force_torque",
        "cfg.events.push_robot",
    }
    assigned = set()
    for node in ast.walk(_module()):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    assigned.add(ast.unparse(target))
    assert assigned <= allowed, assigned - allowed


def test_the_ranges_are_the_declared_ones():
    assert _constant("BACKWARD_SPEED") == (-0.5, 0.0)
    assert _constant("SIDESTEP_SPEED") == (-0.5, 0.5)
    assert _constant("JOG_SPEED") == (1.0, 1.5)


@pytest.mark.parametrize(
    "task, axis, speed",
    [("backward", 0, "BACKWARD_SPEED"), ("sidestep", 1, "SIDESTEP_SPEED"), ("jog", 0, "JOG_SPEED")],
)
def test_every_play_command_lies_in_its_training_range(task, axis, speed):
    low, high = _constant(speed)
    for vx, vy, wz in _constant("PLAY_COMMANDS")[task].values():
        command = (vx, vy)
        assert low <= command[axis] <= high
        assert command[1 - axis] == 0.0 and wz == 0.0


def test_registration_uses_isaac_labs_flat_ppo_and_both_variants():
    from humanoid_transfer.g1.tasks import g1_base

    assert g1_base.TASKS == ("Backward", "Sidestep", "Jog")
    assert g1_base.task_id("Jog") == "G1Base-Jog-v0"
    assert g1_base.task_id("Jog", play=True) == "G1Base-Jog-Play-v0"
    source = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "G1FlatPPORunnerCfg" in source and "G1RoughPPORunnerCfg" not in source
    gym = pytest.importorskip("gymnasium")
    if g1_base.REGISTERED:
        for name in g1_base.TASKS:
            assert g1_base.task_id(name) in gym.envs.registry
            assert g1_base.task_id(name, play=True) in gym.envs.registry
