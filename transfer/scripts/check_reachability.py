"""Check a captured teacher gait against what the student's joints can reach.

Run this on a teacher rollout before using it to supervise anything. It answers
the one feasibility question that is about the student's body rather than about
its strategy: mapped through the correspondence, does the teacher's
configuration land inside the NAO's joint limits?

A nonzero exit means some part of the teacher's demonstration is unreachable.
That is not automatically a defect in the data -- the robots genuinely differ --
but it is always something to look at before training on it, because the two
causes need opposite responses:

* a **convention mismatch** is a bug in ``correspondence.JOINT_SIGN`` and must
  be fixed;
* a **genuine range difference** is a fact about the NAO and should be absorbed
  by ``feasibility.reachability_mask`` instead.

The report prints the worst overshoot per joint so the two can be told apart. A
mismatch is usually large and affects every sample; a range difference is
usually partial.

Usage::

    python transfer/scripts/check_reachability.py
    python transfer/scripts/check_reachability.py --rollouts path/to/teacher.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.transfer import correspondence  # noqa: E402
from humanoid_transfer.transfer.feasibility import (  # noqa: E402
    joint_reachability,
    reachability_mask,
    reachability_report,
    standing_margin,
)

DEFAULT_ROLLOUTS = REPO_ROOT / "g1" / "rollouts" / "teacher.npz"

# Froude-matched speeds, from ``common.scaling``. The student's centre-of-mass
# velocity implied by a teacher state is the teacher's scaled by this ratio.
STUDENT_SPEED = 0.15
TEACHER_SPEED = 0.2488


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=DEFAULT_ROLLOUTS,
        help="teacher rollout archive (default: g1/rollouts/teacher.npz)",
    )
    args = parser.parse_args()

    if not args.rollouts.is_file():
        print(f"no rollouts at {args.rollouts}")
        print("capture them first:  python g1/scripts/capture_rollouts.py")
        return 2

    data = np.load(args.rollouts, allow_pickle=True)
    names = [str(s) for s in data["joint_names"]]
    joint_pos = data["joint_pos"].reshape(-1, len(names))
    com_vel = data["com_vel"].reshape(-1, 3)

    print(f"rollouts : {args.rollouts}")
    print(f"samples  : {joint_pos.shape[0]}")
    print(f"teacher  : {float(data['commanded_speed']):.4f} m/s commanded")
    print()

    print("JOINT-RANGE REACHABILITY  (the criterion that constrains the body)")
    print(reachability_report(joint_pos, names))
    print()

    weight = reachability_mask(joint_pos, names)
    print(
        f"reachability weight: mean {weight.mean():.3f}, "
        f"{100 * np.mean(weight == 0.0):.2f}% of samples at zero"
    )
    print()

    # Instrumentation. Reported because it describes the gait, never imposed.
    scale = STUDENT_SPEED / TEACHER_SPEED
    margin = standing_margin(com_vel[:, :2] * scale)
    print("ZERO-STEP MARGIN  (instrumentation only -- not a constraint)")
    print(
        f"  fraction of the standing envelope used, median {np.median(margin):.3f}"
        f", max {margin.max():.3f}"
    )
    print("  Reported because a gait that keeps margin without being asked to is")
    print("  a result. Requiring it would make the same number an assumption.")
    print("  This is the standing task's criterion and must not weight a")
    print("  walking objective: a state needing a step is a walking state.")
    print()

    reach = joint_reachability(joint_pos, names)
    failures = {k: v for k, v in reach.fraction_outside.items() if v > 0.0}
    if not failures:
        print("PASS  every mapped joint of the teacher gait is reachable")
        if reach.sign_blind:
            print(
                f"      ({len(reach.sign_blind)} joints have symmetric ranges, "
                "where a sign error would not show)"
            )
        return 0

    print("FAIL  the teacher asks for configurations the student cannot reach")
    for name, fraction in sorted(failures.items(), key=lambda kv: -kv[1]):
        index = reach.joint_names.index(name)
        worst = float(np.abs(reach.excess[:, index]).max())
        verdict = (
            "every sample, large overshoot -> suspect a sign convention"
            if fraction > 0.99 and worst > 0.2
            else "partial -> likely a genuine range difference"
        )
        print(
            f"      {name:<16}{100 * fraction:>7.2f}% of samples, worst {worst:.3f} rad   {verdict}"
        )
    print()
    print("      Signs live in correspondence.JOINT_SIGN:")
    for role, signs in sorted(correspondence.JOINT_SIGN.items()):
        if any(s < 0 for s in signs):
            print(f"        {role}: {signs}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
