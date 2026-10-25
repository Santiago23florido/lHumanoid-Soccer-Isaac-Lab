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
