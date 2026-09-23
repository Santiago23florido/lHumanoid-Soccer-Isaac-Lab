"""Checks on the standing task's configuration.

The environment module imports Isaac Lab, which needs the Isaac Sim Kit
application, so these read the configuration out of the source the way
``test_nao_actuators.py`` does. Task registration itself pulls in no Isaac Lab
code -- the environment classes are named by string -- so that part is checked
by importing.

What they protect is the arithmetic. The observation and action widths are
declared as bare integers, and the environment builds the tensors somewhere
else entirely; if the two ever disagree the failure surfaces as a shape error
deep inside a training run, hours after the mistake.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "humanoid_soccer_lab"))

from humanoid_soccer_lab.nao.assets import nao_kinematics as nk  # noqa: E402

TASK = ROOT / "source" / "humanoid_soccer_lab" / "humanoid_soccer_lab" / "nao" / "tasks"
CFG_SOURCE = (TASK / "nao_stand" / "nao_stand_env_cfg.py").read_text(encoding="utf-8")
ENV_SOURCE = (TASK / "nao_stand" / "nao_stand_env.py").read_text(encoding="utf-8")


def _int_field(name: str) -> int:
    match = re.search(rf"^    {name} = (-?\d+)", CFG_SOURCE, re.M)
    assert match is not None, f"{name} not found in the config"
    return int(match.group(1))


def _float_field(name: str) -> float:
    match = re.search(rf"^    {name} = (-?[\d.]+)", CFG_SOURCE, re.M)
    assert match is not None, f"{name} not found in the config"
    return float(match.group(1))


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_both_task_variants_register() -> None:
    import gymnasium as gym
    from humanoid_soccer_lab.nao.tasks.nao_stand import PLAY_TASK_ID, REGISTERED, TASK_ID

    assert REGISTERED
    for task_id in (TASK_ID, PLAY_TASK_ID):
        assert task_id in gym.envs.registry


def test_registration_points_at_an_agent_config() -> None:
    """Without this entry point the training script cannot resolve the agent."""
    import gymnasium as gym
    from humanoid_soccer_lab.nao.tasks.nao_stand import TASK_ID

    kwargs = gym.envs.registry[TASK_ID].kwargs
    assert "env_cfg_entry_point" in kwargs
    assert "rsl_rl_cfg_entry_point" in kwargs


# --------------------------------------------------------------------------
# The declared widths must match what the environment builds
# --------------------------------------------------------------------------


def test_action_width_equals_the_number_of_commanded_joints() -> None:
    assert _int_field("action_space") == len(nk.ACTUATED_JOINTS) == 19


def test_actor_observation_width_matches_its_documented_breakdown() -> None:
    """3 gravity + 3 angular velocity + 19 + 19 + 19 + 2 contacts."""
    joints = len(nk.ACTUATED_JOINTS)
    expected = 3 + 3 + joints + joints + joints + len(nk.FOOT_BODIES)
    assert _int_field("observation_space") == expected == 65


def test_privileged_state_width_matches_its_documented_breakdown() -> None:
    """3 base velocity + 2 CoM + 2 CoM velocity + 2 DCM + 2 forces
    + 3 push + 1 base height + 1 CoM height + 1 margin."""
    expected = 3 + 2 + 2 + 2 + len(nk.FOOT_BODIES) + 3 + 1 + 1 + 1
    assert _int_field("state_space") == expected == 17


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def test_the_control_loop_is_fast_enough_for_the_pendulum() -> None:
    """The balance error doubles every 115 ms; the loop must beat that badly."""
    import math

    decimation = _int_field("decimation")
    physics_dt = 1.0 / 200.0
    control_period = decimation * physics_dt

    omega = nk.lipm_omega(nk.com_height_above_soles(nk.NOMINAL_STAND_JOINT_POS))
    doubling_time = math.log(2.0) / omega

    assert control_period == 0.01, "the real NAO's DCM runs at 100 Hz"
    assert doubling_time / control_period > 8.0


def test_the_discount_looks_far_enough_ahead_to_see_a_recovery() -> None:
    """gamma=0.99 at 100 Hz is a 1 s horizon: barely one push recovery."""
    agents = TASK / "nao_stand" / "agents" / "rsl_rl_ppo_cfg.py"
    source = agents.read_text(encoding="utf-8")
    match = re.search(r"^        gamma=([\d.]+),", source, re.M)
    assert match is not None, "gamma not found in the agent config"
    gamma = float(match.group(1))
    horizon_seconds = 1.0 / (1.0 - gamma) / 100.0
    # 1e-6 of slack: 1/(1-0.995) lands on 199.999... in binary floating point.
    assert horizon_seconds >= 2.0 - 1e-6, (
        f"effective horizon is {horizon_seconds:.2f} s, under the ~1 s a single "
        "push recovery already takes"
    )


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------


def test_the_curriculum_stops_at_the_capturable_limit() -> None:
    """Past it an ankle strategy cannot recover, so the ramp stops there.

    The value is taken from the kinematics rather than typed in, so this
    checks the wiring rather than a number: a literal here would go stale the
    moment the nominal posture moved the centre of mass.
    """
    assert 'push_velocity_final = float(_CAPTURABLE["forward"])' in CFG_SOURCE
    assert nk.capturable_com_velocity()["forward"] > 0.4


def test_the_curriculum_starts_far_below_where_it_ends() -> None:
    """Starting at full magnitude means falling constantly and learning nothing."""
    start = _float_field("push_velocity_initial")
    final = nk.capturable_com_velocity()["forward"]
    assert 0.0 < start < final / 5.0


# --------------------------------------------------------------------------
# Correctness details that are easy to regress
# --------------------------------------------------------------------------


def test_timeouts_are_truncation_rather_than_failure() -> None:
    """is_finite_horizon False makes the algorithm bootstrap on a timeout.

    Without it the policy is taught that surviving the full episode is worth
    nothing, which is the opposite of the task.
    """
    assert re.search(r"^    is_finite_horizon = False", CFG_SOURCE, re.M)


def test_the_reset_events_run_after_the_nominal_state_is_written() -> None:
    """Writing the defaults after the events would discard the randomisation."""
    write = ENV_SOURCE.index("write_joint_state_to_sim")
    events = ENV_SOURCE.index("super()._reset_idx(env_ids)")
    assert write < events, "the nominal state must be written before the reset events"


def test_the_observation_refreshes_the_derived_values() -> None:
    """Otherwise a freshly reset environment reports the episode that ended."""
    body = ENV_SOURCE[ENV_SOURCE.index("def _get_observations") :]
    body = body[: body.index("def _get_rewards")]
    assert "_compute_intermediate_values()" in body


def test_actions_and_targets_are_bounded() -> None:
    """A Gaussian policy is unbounded; the joint limits are not."""
    assert "action_clip" in CFG_SOURCE
    assert "clamp" in ENV_SOURCE


def test_the_body_names_are_articulation_names_not_urdf_names() -> None:
    """The URDF's torso merges into base_link, and asking for it raises."""
    assert 'TORSO_BODY = "base_link"' in CFG_SOURCE
    assert '"torso"' not in CFG_SOURCE.replace("TORSO_BODY", "")
