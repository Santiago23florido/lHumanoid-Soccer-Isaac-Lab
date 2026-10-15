"""Close the loop on a contract: does a body that follows the plans walk like the teacher?

Reads closed-loop captures (``g1/scripts/capture_rollouts.py``) and applies two
gates fixed before the tracker was trained:

**Use (gate B).** The state a body reaches is scored against the plan it was
holding and against the plan another environment was holding at the same
instant (``contract_error`` and ``contract_error_other``, recorded by the
capture). A body that follows its plan is much closer to its own: the median
own error must be below half the median other error. A body that ignored the
plan and walked anyway would score alike against both.

**Behaviour (gate C).** The same criteria every closed-loop run of the study
was held to, from ``evaluate_latent.py``: speed at least 0.90 of the
teacher's, fall rate at most 2 points above it, and stride length, stride
period and both duty factors within 10 %.

**Across bodies** (pre-registered for the blind test, ``skill_space.md`` §8.4). A
body samples a gait at its own control rate, and a coarser rate lengthens every
stance it can represent: the NAO's step is 1.62 times the G1's in body time. When
a run's rate differs from the teacher's, ``verdict_at_body_rate`` also holds it
to the teacher's gait resampled, in body time, to the run's rate. Both verdicts
are reported; within one body they coincide.

**Where a miss comes from** (captures that record the plan). The plans' own
due states, stitched in time, are the motion the body was asked for: its gait
against the teacher's is the generator's share of a miss, the body's gait
against it the tracker's. The error is also given per field of ``z*``.

Usage::

    python transfer/scripts/evaluate_contract.py --run teacher=g1/rollouts/closed_loop_teacher.npz \
        --run tracker=g1/rollouts/closed_loop_track.npz --output transfer/results/contract_g1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_latent import CRITERIA, SETTLE_STEPS, summarise, verdict  # noqa: E402
from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402

USE_RATIO_MAX = 0.5


def contract_use(path: Path) -> dict | None:
    data = np.load(path)
    if "contract_error" not in data.files:
        return None
    # Own and other are both scored on the state alone: the other environment's
    # plan was drawn from another place, so its trajectory term would not compare.
    own_key = "contract_error_state" if "contract_error_state" in data.files else "contract_error"
    own, other = data[own_key], data["contract_error_other"]
    done = data["done"]
    # The step after a reset carries no plan and scores zero; leave it out.
    first = np.zeros_like(done)
    first[1:] = done[:-1]
    valid = ~first & (own > 0)
    own, other = own[valid], other[valid]
    ratio = float(np.median(own) / np.median(other))
    return {
        "median_error_own_plan": float(np.median(own)),
        "median_error_other_plan": float(np.median(other)),
        "mean_error_own_plan": float(own.mean()),
        "ratio": ratio,
        "rule": f"median own / median other < {USE_RATIO_MAX}",
        "passes": ratio < USE_RATIO_MAX,
    }


def asked(path: Path) -> dict | None:
    """The motion the plans asked for, and the error per field of ``z*``."""
    data = np.load(path)
    if "contract_due" not in data.files:
        return None
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    done = data["done"]
    # The due state of step t is the target of the state recorded at t + 1.
    due = data["contract_due"][:-1].astype(float)
    after = done[:-1]
    # A reset's first step holds no plan: cut the stream there, as an episode end.
    cut = after.copy()
    cut[:-1] |= done[1:-1]
    dt_star = float(data["control_dt"]) / units.time
    signature = gait_signature(due, cut, dt_star).as_dict()
    forward = SKILL_LAYOUT.unpack(due)["com_velocity"][..., 0] * units.velocity
    first = np.zeros_like(done)
    first[1:] = done[:-1]
    fields = data["contract_fields"]
    valid = ~first & (fields.sum(-1) > 0)
    names = [f.name for f in SKILL_LAYOUT.fields] + ["progress"]
    return {
        "forward_speed": float(forward[SETTLE_STEPS:].mean()),
        "signature": signature,
        "median_error_by_field": {
            n: float(np.median(fields[..., i][valid])) for i, n in enumerate(names)
        },
    }


def dt_star(path: Path) -> float:
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    return float(data["control_dt"]) / units.time


def signature_at_rate(path: Path, rate: float) -> dict:
    """The gait of a capture as a body controlled every ``rate`` body times would sample it."""
    data = np.load(path)
    z = data["skill_state"].astype(float)
    done = data["done"]
    source = np.arange(z.shape[0]) * dt_star(path)
    target = np.arange(0.0, source[-1], rate)
    resampled = np.empty((len(target), z.shape[1], z.shape[2]))
    for env in range(z.shape[1]):
        for k in range(z.shape[2]):
            resampled[:, env, k] = np.interp(target, source, z[:, env, k])
    # A new sample ends an episode if any original step since the previous one did.
    ended = np.cumsum(done, axis=0)
    index = np.minimum(np.searchsorted(source, target, side="right") - 1, len(source) - 1)
    counts = ended[index]
    resampled_done = np.zeros(counts.shape, dtype=bool)
    resampled_done[1:] = counts[1:] > counts[:-1]
    return gait_signature(resampled, resampled_done, rate).as_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="name=path/to/capture.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        name, _, path = item.partition("=")
        runs[name] = summarise(REPO_ROOT / path)
        capture = np.load(REPO_ROOT / path)
        units = UnitSystem(float(capture["leg_length"]), float(capture["total_mass"]))
        runs[name]["velocity_unit"] = units.velocity
        use = contract_use(REPO_ROOT / path)
        if use is not None:
            runs[name]["contract_use"] = use
        plan = asked(REPO_ROOT / path)
        if plan is not None:
            runs[name]["asked"] = plan
    if "teacher" not in runs:
        print("a run named 'teacher' is needed as the reference")
        return 2

    # Speeds compare in V = sqrt(g L), the unit every body shares; on one body the
    # ratio is the same as in m/s.
    def in_v(run: dict) -> dict:
        speed = run["forward_speed"]
        return {**run, "forward_speed": None if speed is None else speed / run["velocity_unit"]}

    paths = {}
    for item in args.run:
        name, _, path = item.partition("=")
        paths[name] = REPO_ROOT / path
    teacher_rate = dt_star(paths["teacher"])
    for name, run in runs.items():
        if name != "teacher":
            run["verdict"] = verdict(in_v(run), in_v(runs["teacher"]))
            rate = dt_star(paths[name])
            if abs(rate / teacher_rate - 1.0) > 0.01:
                at_rate = {**runs["teacher"]}
                at_rate["signature"] = signature_at_rate(paths["teacher"], rate)
                run["verdict_at_body_rate"] = {
                    **verdict(in_v(run), in_v(at_rate)),
                    "teacher_signature": at_rate["signature"],
                    "body_dt_star": rate,
                    "teacher_dt_star": teacher_rate,
                }
        if "asked" in run:
            plan = run["asked"]
            asked_run = {**run, "forward_speed": plan["forward_speed"]}
            asked_run["signature"] = plan["signature"]
            plan["generator_share"] = verdict(in_v(asked_run), in_v(runs["teacher"]))[
                "signature_deviation"
            ]

    report = {
        "criteria": CRITERIA,
        "use_ratio_max": USE_RATIO_MAX,
        "settle_steps": SETTLE_STEPS,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, run in runs.items():
        sig = run["signature"]
        line = (
            f"{name:<22} falls {run['falls']}/{run['episodes']}  speed {run['forward_speed']}  "
            f"stride {sig['stride_length']:.3f}  period {sig['stride_period']:.3f}  "
            f"duty {np.round(sig['duty_factor'], 3)}"
        )
        if "verdict" in run:
            v = run["verdict"]
            line += (
                f"  ratio {v['speed_ratio']:.2f}  dev "
                f"{ {k: round(x, 3) for k, x in v['signature_deviation'].items()} }  "
                f"{'PASS' if v['passes'] else 'fail'}"
            )
        if "contract_use" in run:
            u = run["contract_use"]
            line += f"  use {u['ratio']:.3f} ({'PASS' if u['passes'] else 'fail'})"
        print(line)
        if "verdict_at_body_rate" in run:
            v = run["verdict_at_body_rate"]
            print(
                f"{'  at the body rate':<22} dev "
                f"{ {k: round(x, 3) for k, x in v['signature_deviation'].items()} }  "
                f"{'PASS' if v['passes'] else 'fail'}"
            )
        if "asked" in run:
            a = run["asked"]
            sig = a["signature"]
            print(
                f"{'  asked by the plans':<22} speed {a['forward_speed']:.3f}  "
                f"stride {sig['stride_length']:.3f}  period {sig['stride_period']:.3f}  "
                f"duty {np.round(sig['duty_factor'], 3)}"
            )
            fields = {k: round(v, 2) for k, v in a["median_error_by_field"].items()}
            print(f"{'  error by field':<22} {fields}")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
