"""The pinned teacher: is the artifact in the repository the one it claims to be?

``g1/policy/`` tracks a trained policy, which is the only derived binary this
repository keeps. The reason is in ``g1/policy/README.md``: the teacher is not
reproducible, and every transfer comparison is a comparison *against it*, so a
teacher that drifts between experiments turns the study into a measurement of
the teacher.

Drift is silent by nature -- a different ``.pt`` loads, runs and produces
plausible actions -- so these checks exist to make it loud. They are cheap:
hashing 357 KB and parsing one JSON file, with the TorchScript load skipped when
torch is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from humanoid_transfer.common.scaling import froude_matched_speed

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "g1" / "policy"
MANIFEST_PATH = POLICY_DIR / "teacher_policy.json"
TASK_CFG = ROOT / "source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_walk/g1_walk_env_cfg.py"

MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# --- the artifact is present and is what the manifest says -------------------


def test_the_teacher_is_tracked_rather_than_left_in_the_log_tree() -> None:
    """``logs/`` is gitignored, so a teacher that lives only there is not in the
    branch at all: a fresh clone has no teacher and no error saying so."""
    policy = POLICY_DIR / MANIFEST["policy"]["file"]
    assert policy.is_file()


def test_the_policy_bytes_match_the_manifest_hash() -> None:
    """The whole point of pinning. A swapped teacher fails here and nowhere else."""
    policy = POLICY_DIR / MANIFEST["policy"]["file"]
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert digest == MANIFEST["policy"]["sha256"]
    assert policy.stat().st_size == MANIFEST["policy"]["bytes"]


def test_the_manifest_records_what_the_run_cannot_be_asked_again() -> None:
    """Seed, iterations and environment count decide whether a rerun is the same
    experiment. They are not recoverable from the weights."""
    provenance = MANIFEST["provenance"]
    for key in ("seed", "iterations", "num_envs", "steps_per_env_per_update"):
        assert isinstance(provenance[key], int)
    assert provenance["task_id"] == "G1Walk-Teacher-v0"
    assert provenance["control_hz"] == pytest.approx(
        1.0 / (provenance["sim_dt"] * provenance["decimation"])
    )
    assert (
        provenance["total_env_steps"]
        == provenance["iterations"]
        * provenance["num_envs"]
        * provenance["steps_per_env_per_update"]
    )


def test_the_configurations_the_run_dumped_are_kept_beside_the_weights() -> None:
    """Weights without their environment are not a reproducible result."""
    for name in MANIFEST["provenance"]["params_dumped_by_the_run"]:
        assert (POLICY_DIR / name).is_file()


# --- the pinned teacher still matches the code in this branch ----------------


def test_the_pinned_command_range_is_the_one_this_branch_would_train() -> None:
    """The failure this catches: someone changes ``STUDENT_TARGET_SPEED`` and the
    pinned teacher silently becomes a policy for a different experiment."""
    expected = froude_matched_speed(0.15, 0.2689, 0.74)
    pinned = MANIFEST["command_distribution"]["lin_vel_x"][1]
    assert pinned == pytest.approx(expected, rel=1e-12)

    source = TASK_CFG.read_text(encoding="utf-8")
    assert "STUDENT_TARGET_SPEED = 0.15" in source
    assert "STUDENT_LEG_LENGTH = 0.2689" in source
    assert "TEACHER_LEG_LENGTH = 0.74" in source


def test_the_run_itself_recorded_the_same_command_range() -> None:
    """Read out of the YAML the training run dumped, not out of the manifest.

    Isaac Lab dumps ``!!python/tuple`` tags that ``yaml.safe_load`` refuses, so
    this reads the one line that matters as text. Less elegant than parsing, and
    it checks the thing that matters: the weights and the recorded config agree.
    """
    dumped = (POLICY_DIR / "teacher_env.yaml").read_text(encoding="utf-8")
    pinned = MANIFEST["command_distribution"]["lin_vel_x"][1]
    assert repr(pinned) in dumped, f"{pinned!r} is not in the dumped config"


def test_the_pinned_teacher_is_untouched_by_the_unreachable_dimensions() -> None:
    """Lateral and turning commands were zeroed; a teacher trained with them is a
    different teacher even at the same forward speed."""
    commands = MANIFEST["command_distribution"]
    for key in ("lin_vel_y", "ang_vel_z", "heading"):
        assert commands[key] == [0.0, 0.0]


# --- the verification is recorded, because reward does not show a walk -------


def test_the_manifest_records_a_measured_gait_not_a_reward_curve() -> None:
    """A policy that stands still, slides or steps in place can saturate episode
    length and collect the same reward. Only the measurement separates those."""
    measured = MANIFEST["verification"]
    assert measured["verdict"] == "walking"
    assert measured["achieved_forward_velocity_body_frame_mps"] > 0.0
    assert measured["fraction_of_command"] == pytest.approx(
        measured["achieved_forward_velocity_body_frame_mps"]
        / measured["commanded_forward_velocity_mps"],
        abs=0.01,
    )
    # Swing phase rather than contact jitter: centimetres, not millimetres.
    for foot in ("foot_clearance_left_mm", "foot_clearance_right_mm"):
        assert measured[foot]["mean"] > 20.0
        assert measured[foot]["max"] >= measured[foot]["mean"]


def test_the_manifest_states_what_the_teacher_does_not_establish() -> None:
    """One seed, one speed, no hardware, and nothing about transfer."""
    assert len(MANIFEST["what_this_does_not_establish"]) >= 4


def test_the_achieved_speed_scales_back_to_under_the_student_target() -> None:
    """83 % of command is not a failure, it is the teacher's actual speed, and it
    is the one the rollouts have to be scaled from rather than the commanded
    0.249 m/s -- which would ask the student for a gait the teacher never walked.
    """
    achieved = MANIFEST["verification"]["achieved_forward_velocity_body_frame_mps"]
    student_equivalent = froude_matched_speed(achieved, 0.74, 0.2689)
    assert student_equivalent == pytest.approx(0.124, abs=1e-3)
    assert student_equivalent < 0.15


# --- the weights load, and are the shape the manifest promises ---------------


def _load_policy():
    torch = pytest.importorskip("torch")
    policy = torch.jit.load(str(POLICY_DIR / MANIFEST["policy"]["file"]), map_location="cpu")
    policy.eval()
    return torch, policy


def test_the_export_loads_with_torch_alone_and_has_the_promised_width() -> None:
    """TorchScript, not a checkpoint: usable without Isaac Lab, Isaac Sim or
    Nucleus access. That is what makes pinning it worth the 357 KB."""
    torch, policy = _load_policy()

    spec = MANIFEST["policy"]
    assert sum(p.numel() for p in policy.parameters()) == spec["parameter_count"]

    with torch.no_grad():
        action = policy(torch.zeros(4, spec["observation_dim"]))
    assert tuple(action.shape) == (4, spec["action_dim"])
    assert torch.isfinite(action).all()


def test_the_action_width_is_every_joint_of_the_articulation() -> None:
    """37, not 23. Isaac Lab's shipped action term commands the hand joints too,
    and assuming 23 is the same class of mistake as the 29-DOF name the asset
    does not live up to (``g1/TRAINING.md`` section 5)."""
    assert MANIFEST["policy"]["action_dim"] == 37
    assert MANIFEST["policy"]["action_dim"] == 23 + 14


def test_a_wrong_observation_width_is_rejected_rather_than_broadcast() -> None:
    """Silent shape coercion is how a teacher ends up being fed a vector
    assembled differently and still returning plausible numbers."""
    torch, policy = _load_policy()

    wrong = MANIFEST["policy"]["observation_dim"] - 1
    with pytest.raises(RuntimeError), torch.no_grad():
        policy(torch.zeros(1, wrong))


def test_the_parameter_count_is_consistent_with_the_file_size() -> None:
    """A float32 TorchScript module cannot be much smaller than its weights."""
    spec = MANIFEST["policy"]
    assert spec["bytes"] > 4 * spec["parameter_count"]
    assert spec["bytes"] < 4 * spec["parameter_count"] * 1.5


def test_the_documented_wall_clock_is_the_reason_this_is_pinned() -> None:
    """Not a vanity number: it is why regenerating is not an acceptable
    alternative to tracking the bytes."""
    assert MANIFEST["provenance"]["wall_clock_seconds"] > 3600
    assert math.isclose(MANIFEST["provenance"]["wall_clock_seconds"] / 3600.0, 1.96, abs_tol=0.05)
