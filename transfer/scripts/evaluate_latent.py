"""Does the G1 still walk when its policy runs through the shared latent space?

Offline imitation error says how close the factored policy's actions are to
the teacher's on the teacher's own states. It says nothing about what happens
when the factored policy chooses the states -- small errors compound, and a
policy that imitates at R^2 0.96 can still fall. So each policy is run
closed-loop with ``g1/scripts/capture_rollouts.py --driver-policy``, the
teacher labelling every step, and this script compares the captures:

* **falls**: episodes that terminated rather than timed out;
* **speed**: forward CoM speed in the heading frame, after a settling time;
* **gait signature**: the dimensionless numbers of :mod:`common.gait`;
* **on-policy disagreement**: how far the driver's actions are from what the
  teacher would have done in the same state.

Pass criteria, fixed before the run in the plan for this phase: speed at least
90 % of the teacher's, fall rate no more than 2 points above it, stride
length, period and duty factor within 10 %. The bypass policy -- the decoder
with the latent action frozen -- is expected to *fail*: if it walks, the
skill was never in the shared space.

Usage::

    python transfer/scripts/evaluate_latent.py \
        --run teacher=g1/rollouts/closed_loop_teacher.npz \
        --run anchored=g1/rollouts/closed_loop_anchored.npz ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.gait import (  # noqa: E402
    dominant_rhythm,
    episode_start,
    gait_signature,
)
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402

SETTLE_STEPS = 100
"""Two seconds after a reset are excluded from the speed: the robot is starting up."""

CRITERIA = {"speed_ratio_min": 0.90, "fall_rate_excess_max": 0.02, "signature_tolerance": 0.10}


def _settled(done: np.ndarray, already_running: int) -> np.ndarray:
    """``(T, N)`` True once ``SETTLE_STEPS`` have passed since the episode began."""
    age = np.zeros(done.shape, dtype=int)
    current = np.full(done.shape[1], already_running)
    starts = episode_start(done)
    starts[0] = False
    for t in range(done.shape[0]):
        current = np.where(starts[t], 0, current + (t > 0))
        age[t] = current
    return age >= SETTLE_STEPS


def summarise(path: Path) -> dict:
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    done, terminated = data["done"], data["terminated"]
    z = data["skill_state"].astype(float)
    running = int(data["settle_steps"]) if "settle_steps" in data.files else 0
    settled = _settled(done, already_running=running)

    forward = SKILL_LAYOUT.unpack(z)["com_velocity"][..., 0] * units.velocity
    episodes = int(done.shape[1] + done.sum())
    executed, label = data["action_executed"], data["action_teacher"]
    dt_star = float(data["control_dt"]) / units.time
    signature = gait_signature(z, done, dt_star).as_dict()
    rhythm = dominant_rhythm(z, dt_star)
    return {
        "file": path.resolve().relative_to(REPO_ROOT).as_posix(),
        "driver": str(data["driver"]),
        "steps_by_envs": list(done.shape),
        "episodes": episodes,
        "falls": int(terminated.sum()),
        "fall_rate": float(terminated.sum() / episodes),
        # None when no episode lasted long enough to settle: it never walked.
        "forward_speed": float(forward[settled].mean()) if settled.any() else None,
        "commanded_speed": float(data["command"][..., 0].mean()),
        "on_policy_disagreement_mse": float(((executed - label) ** 2).mean()),
        "teacher_action_variance": float(label.reshape(-1, label.shape[-1]).var(axis=0).mean()),
        "signature": signature,
        # Supplementary, added after the touchdown-based period proved sensitive to
        # the contact threshold on this gait (swings last about two control steps).
        "stride_period_by_threshold": {
            f"{t:.2f}": gait_signature(z, done, dt_star, threshold=t).stride_period
            for t in (0.01, 0.05, 0.10, 0.20)
        },
        "weight_shift_rhythm": {"period": rhythm.period, "power_share": rhythm.power_share},
    }


def verdict(run: dict, teacher: dict) -> dict:
    ratio = (
        run["forward_speed"] / teacher["forward_speed"] if run["forward_speed"] is not None else 0.0
    )
    excess = run["fall_rate"] - teacher["fall_rate"]
    deviations = {}
    for key in ("stride_length", "stride_period"):
        reference = teacher["signature"][key]
        deviations[key] = abs(run["signature"][key] - reference) / reference
    for side, (value, reference) in enumerate(
        zip(run["signature"]["duty_factor"], teacher["signature"]["duty_factor"], strict=True)
    ):
        deviations[f"duty_factor_{side}"] = abs(value - reference) / reference
    passes = (
        ratio >= CRITERIA["speed_ratio_min"]
        and excess <= CRITERIA["fall_rate_excess_max"]
        and all(
            np.isfinite(v) and v <= CRITERIA["signature_tolerance"] for v in deviations.values()
        )
    )
    reference = teacher["weight_shift_rhythm"]["period"]
    rhythm = abs(run["weight_shift_rhythm"]["period"] - reference) / reference
    return {
        "speed_ratio": ratio,
        "fall_rate_excess": excess,
        "signature_deviation": deviations,
        "passes": bool(passes),
        "supplementary_rhythm_deviation": rhythm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="name=path/to/capture.npz")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "transfer" / "results" / "latent_g1_closed_loop.json",
    )
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        runs[name] = summarise(REPO_ROOT / path)
    if "teacher" not in runs:
        print("a run named 'teacher' is needed as the reference")
        return 2

    for name, run in runs.items():
        if name != "teacher":
            run["verdict"] = verdict(run, runs["teacher"])

    report = {"criteria": CRITERIA, "settle_steps": SETTLE_STEPS, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    header = f"{'run':<16}{'falls':>8}{'speed':>9}{'ratio':>8}{'stride':>9}{'period':>9}"
    print(header + f"{'duty L/R':>13}{'rhythm':>8}{'disagree':>10}  verdict")
    for name, run in runs.items():
        sig = run["signature"]
        ratio = run.get("verdict", {}).get("speed_ratio", 1.0)
        status = run.get("verdict", {}).get("passes")
        speed = run["forward_speed"] if run["forward_speed"] is not None else float("nan")
        print(
            f"{name:<16}{run['falls']:>4}/{run['episodes']:<3}{speed:>9.3f}"
            f"{ratio:>8.2f}{sig['stride_length']:>9.3f}{sig['stride_period']:>9.2f}"
            f"{sig['duty_factor'][0]:>7.2f}/{sig['duty_factor'][1]:.2f}"
            f"{run['weight_shift_rhythm']['period']:>8.2f}"
            f"{run['on_policy_disagreement_mse']:>10.4f}  "
            f"{'reference' if status is None else ('PASS' if status else 'fail')}"
        )
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
