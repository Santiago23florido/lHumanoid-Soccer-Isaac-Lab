"""A tracker's competence over the plan family: gait by gait, against what was asked.

``capture_contract.py --gait-probes`` makes every environment follow one fixed
gait of the procedural family (``make_gait_probes.py``). For each probe the
body's gait -- the same signature every capture is judged with -- is compared
with the gait asked for, the plan's due states stitched in time. The criteria
are the ones pre-registered for the twins (``transfer/docs/skill_space.md``
§8.3):

* falls at most 5 % of the probe's episodes;
* mean velocity within 10 % of the asked (a floor of 0.05 ``V`` for slow probes);
* stride period and both duties within 10 % of the asked;
* the plan used: median error against its own plan below half that against
  another environment's.

The share of probes passed, per group, is the tracker's competence map; a
probe the body's control step cannot represent (a swing shorter than 1.5
control steps) is reported apart. Without probes -- a capture of random
schedules -- the level L1 numbers: falls, use and the biped's own falls.

Usage::

    python transfer/scripts/evaluate_competence.py --capture g1/rollouts/competence_r12.npz \
        --output transfer/results/competence_g1_r12.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.gait import gait_signature  # noqa: E402
from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from humanoid_transfer.transfer.latent.procedural import (  # noqa: E402
    SWING_STEPS,
    GaitRanges,
    feasible,
)

FALLS_MAX = 0.05
TOLERANCE = 0.10
SPEED_FLOOR = 0.05
USE_MAX = 0.5
VELOCITY = SKILL_LAYOUT.slice_of("com_velocity")


def use_ratio(own: np.ndarray, other: np.ndarray, done: np.ndarray) -> float:
    first = np.zeros_like(done)
    first[1:] = done[:-1]
    valid = ~first & (own > 0)
    if not valid.any():
        return float("nan")
    return float(np.median(own[valid]) / np.median(other[valid]))


def judge(data: dict, envs: np.ndarray, dt_star: float) -> dict:
    """The body against the asked gait, over ``envs``."""
    z = data["skill_state"][:, envs].astype(float)
    done = data["done"][:, envs]
    due = data["contract_due"][:-1, envs].astype(float)
    cut = done[:-1].copy()
    cut[:-1] |= done[1:-1]
    body = gait_signature(z, done, dt_star, threshold=0.01)
    asked = gait_signature(due, cut, dt_star, threshold=0.01)
    v_body = z[..., VELOCITY][..., :2].reshape(-1, 2).mean(0)
    v_asked = due[..., VELOCITY][..., :2].reshape(-1, 2).mean(0)
    speed_gap = float(np.linalg.norm(v_body - v_asked))
    speed_scale = max(float(np.linalg.norm(v_asked)), SPEED_FLOOR)
    falls = int(data["terminated"][:, envs].sum())
    episodes = len(envs) + int(done.sum())
    use = use_ratio(
        data["contract_error_state"][:, envs], data["contract_error_other"][:, envs], done
    )

    def relative(a: float, b: float) -> float:
        return abs(a - b) / abs(b) if b and math.isfinite(a) and math.isfinite(b) else float("inf")

    deviation = {
        "speed": speed_gap / speed_scale,
        "stride_period": relative(body.stride_period, asked.stride_period),
        "duty_left": relative(body.duty_factor[0], asked.duty_factor[0]),
        "duty_right": relative(body.duty_factor[1], asked.duty_factor[1]),
    }
    checks = {
        "falls": falls / episodes <= FALLS_MAX,
        "speed": deviation["speed"] <= TOLERANCE,
        "stride_period": deviation["stride_period"] <= TOLERANCE,
        "duty": max(deviation["duty_left"], deviation["duty_right"]) <= TOLERANCE,
        "use": use < USE_MAX,
    }
    return {
        "envs": len(envs),
        "falls": falls,
        "episodes": episodes,
        "use": use,
        "body": {
            "velocity": v_body.tolist(),
            "period": body.stride_period,
            "duty": list(body.duty_factor),
        },
        "asked": {
            "velocity": v_asked.tolist(),
            "period": asked.stride_period,
            "duty": list(asked.duty_factor),
        },
        "deviation": deviation,
        "checks": checks,
        "passes": all(checks.values()),
        "biped_fallen": float(data["biped_fallen"][:, envs].any(axis=0).mean()),
        "biped_recovering": float(data["biped_recovering"][:, envs].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Judge every env as one group against the plans it followed (any source of plans).",
    )
    args = parser.parse_args()
    data = dict(np.load(REPO_ROOT / args.capture))
    leg = float(data["leg_length"])
    dt_star = float(data["control_dt"]) / math.sqrt(leg / 9.81)
    report: dict = {"capture": str(args.capture), "dt_star": dt_star, "criteria": {
        "falls_max": FALLS_MAX, "tolerance": TOLERANCE, "speed_floor": SPEED_FLOOR,
        "use_max": USE_MAX}}  # fmt: skip

    if args.all:
        envs = np.arange(data["done"].shape[1])
        report["all"] = judge(data, envs, dt_star)
        print(json.dumps(report["all"], indent=1))
    elif "probe_index" not in data:
        envs = np.flatnonzero(data["procedural"].any(axis=0))
        if len(envs) == 0:
            envs = np.arange(data["done"].shape[1])
        done = data["done"][:, envs]
        falls = int(data["terminated"][:, envs].sum())
        episodes = len(envs) + int(done.sum())
        use = use_ratio(
            data["contract_error_state"][:, envs], data["contract_error_other"][:, envs], done
        )
        report["random"] = {
            "envs": len(envs),
            "falls": falls,
            "episodes": episodes,
            "fall_rate": falls / episodes,
            "use": use,
            "biped_fallen": float(data["biped_fallen"][:, envs].any(axis=0).mean()),
            "biped_recovering": float(data["biped_recovering"][:, envs].mean()),
            "passes_l1": bool(use < USE_MAX and falls / episodes <= 0.10),
        }
        print(json.dumps(report["random"], indent=1))
    else:
        probes = json.loads(str(data["probes"]))
        index = data["probe_index"]
        ranges = GaitRanges()
        min_swing = max(ranges.min_swing, SWING_STEPS * dt_star)
        results = []
        for i, probe in enumerate(probes):
            envs = np.flatnonzero(index == i)
            if len(envs) == 0:
                continue
            t = torch.tensor
            representable = bool(
                feasible(
                    t([probe["period"]]), t([probe["duty"]]), t([probe["forward"]]),
                    t([probe["lateral"]]), t([probe["width"]]), ranges, min_swing,
                )[0]
            )  # fmt: skip
            results.append({**probe, "representable": representable, **judge(data, envs, dt_star)})
        summary = {}
        for group in sorted({r["group"] for r in results}):
            rows = [r for r in results if r["group"] == group and r["representable"]]
            summary[group] = {
                "probes": len(rows),
                "passed": sum(r["passes"] for r in rows),
                "share": sum(r["passes"] for r in rows) / max(len(rows), 1),
                "not_representable": sum(
                    1 for r in results if r["group"] == group and not r["representable"]
                ),
            }
        report["summary"] = summary
        report["probes"] = results
        print(json.dumps(summary, indent=1))
        for r in results:
            if r["group"] != "grid" or not r["passes"]:
                failed = [k for k, v in r["checks"].items() if not v]
                print(
                    f"{r['group']:9s} {r['name']:22s} {'PASS' if r['passes'] else 'fail'} "
                    f"falls {r['falls']}/{r['episodes']} use {r['use']:.2f} "
                    f"period {r['body']['period']:.2f}/{r['asked']['period']:.2f} "
                    f"duty {np.round(r['body']['duty'], 2)}/{np.round(r['asked']['duty'], 2)} "
                    f"v {np.round(r['body']['velocity'], 3)}/{np.round(r['asked']['velocity'], 3)} "
                    f"{failed}"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
