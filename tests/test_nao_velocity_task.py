"""Checks on the oracle's configuration (``nao/tasks/nao_velocity``).

The environment module imports Isaac Lab, which needs the Kit application, so these
read the configuration out of the source, as ``test_nao_stand_task.py`` does; the
registration names its classes by string and is checked by importing.

What they protect is the oracle's one promise: Isaac Lab's G1 recipe in the NAO's
units. A reward weight written as a bare number would be the G1's value on a body
whose torques are 18 times smaller, and a joint pattern copied from the G1 would
match nothing on the NAO and silently drop its term.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.nao.assets import nao_kinematics as nk  # noqa: E402

TASK = ROOT / "source" / "humanoid_transfer" / "humanoid_transfer" / "nao" / "tasks"
SOURCE = (TASK / "nao_velocity" / "nao_velocity_env_cfg.py").read_text(encoding="utf-8")


def test_the_oracle_tasks_are_registered() -> None:
    gym = __import__("pytest").importorskip("gymnasium")
    from humanoid_transfer.nao.tasks import nao_velocity

    assert nao_velocity.REGISTERED
    for task_id in (
        nao_velocity.FLAT_TASK_ID,
        nao_velocity.FLAT_PLAY_TASK_ID,
        nao_velocity.ROUGH_TASK_ID,
        nao_velocity.ROUGH_PLAY_TASK_ID,
    ):
        assert task_id in gym.envs.registry


def test_every_reward_weight_is_converted_but_the_fall_penalty() -> None:
    weights = re.findall(r"(?<![.\w])weight=(weight\(|[-\d.e]+)", SOURCE)
    raw = [w for w in weights if w != "weight("]
    # The fall is paid once, on one control step of the same length on both bodies.
    assert raw == ["-200.0"], raw
    assignments = re.findall(r"\.weight = (.+)", SOURCE)
    assert assignments, "no weight is set after construction"
    for value in assignments:
        assert value.startswith("weight(") or value == "0.0", value


def test_every_joint_pattern_matches_a_nao_joint() -> None:
    joints = list(nk.ACTUATED_JOINTS)
    patterns = re.findall(r'"(\.\*[A-Za-z]+|L[A-Za-z]+|R[A-Za-z]+)"', SOURCE)
    assert patterns
    for pattern in patterns:
        assert any(re.fullmatch(pattern, j) for j in joints), pattern


def test_the_feet_pattern_names_the_nao_feet() -> None:
    feet = re.search(r'^FEET = "([^"]+)"', SOURCE, re.M).group(1)
    assert [b for b in nk.FOOT_BODIES if re.fullmatch(feet, b)] == list(nk.FOOT_BODIES)


def test_a_fall_is_measured_from_the_ground_under_the_base() -> None:
    from types import SimpleNamespace

    import torch
    from humanoid_transfer.nao.tasks.nao_velocity.terminations import base_below

    def data(**fields) -> SimpleNamespace:
        return SimpleNamespace(data=SimpleNamespace(**fields))

    # Standing in a sunken sub-terrain, fallen on a raised one, standing on a plane.
    base = torch.tensor([0.0, 0.25, 0.30])
    ground = torch.tensor([-0.33, 0.10, 0.0])[:, None].repeat(1, 5)
    env = SimpleNamespace(
        scene={
            "robot": data(root_pos_w=torch.stack([0 * base, 0 * base, base], dim=-1)),
            "height_scanner": data(ray_hits_w=torch.stack([0 * ground, 0 * ground, ground], -1)),
        }
    )
    assert base_below(env, 0.18, scanner="height_scanner").tolist() == [False, True, False]
    # Isaac Lab's measure, from the world's zero, gets both terrains wrong.
    assert base_below(env, 0.18).tolist() == [True, False, False]


def test_the_rough_task_reads_the_ground_and_the_flat_one_has_no_scanner_to_read() -> None:
    rough = r'func=base_below, params=\{"minimum_height": 0\.18, "scanner": "height_scanner"\}'
    assert re.search(rough, SOURCE)
    flat = SOURCE[SOURCE.index("class NaoVelocityFlatEnvCfg") :]
    assert "self.scene.height_scanner = None" in flat
    assert 'self.terminations.base_low.params["scanner"] = None' in flat
    assert "root_height_below_minimum" not in SOURCE


def test_the_spawn_spread_is_a_converted_length() -> None:
    assert re.search(r"spread = similar_range\(\(-0\.5, 0\.5\), LENGTH\)", SOURCE)
    assert re.search(r'"pose_range": \{"x": spread, "y": spread,', SOURCE)


def test_a_diverged_simulation_is_reset_and_left_unrewarded() -> None:
    import importlib
    import inspect
    from types import SimpleNamespace

    import torch
    from humanoid_transfer.nao.tasks.nao_velocity import terminations as t

    # Three environments: a valid state, a NaN joint, a base spinning at 1e10 rad/s.
    joint = torch.zeros(3, 4)
    joint[1, 2] = float("nan")
    spin = torch.zeros(3, 3)
    spin[2, 0] = 1.0e10
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_w=torch.zeros(3, 3), root_ang_vel_w=spin, joint_pos=joint,
            joint_vel=torch.full((3, 4), 5.0),
        )
    )  # fmt: skip
    env = SimpleNamespace(scene={"robot": robot})
    assert t.diverged(env).tolist() == [False, True, True]

    def speed(env, scale: float) -> torch.Tensor:
        return torch.tensor([1.0, float("nan"), 1.0e18]) * scale

    guarded = t.finite(speed)
    assert guarded(env, scale=2.0).tolist() == [2.0, 0.0, 0.0]
    # Isaac Lab checks the term's parameters against this signature ...
    assert inspect.signature(guarded) == inspect.signature(speed)
    # ... and Hydra carries the term as "module:name" and back.
    module, name = guarded.__module__, guarded.__name__
    assert getattr(importlib.import_module(module), name) is guarded


def test_the_velocity_tasks_guard_every_reward_and_reset_a_diverged_simulation() -> None:
    rough = SOURCE[SOURCE.index("class NaoVelocityRoughEnvCfg") :]
    assert "self.terminations.diverged = DoneTerm(func=diverged, time_out=True)" in rough
    assert re.search(
        r"for term in vars\(self\.rewards\)\.values\(\):\s+if isinstance\(term, RewTerm\):\s+"
        r"term\.func = finite\(term\.func\)",
        rough,
    )
