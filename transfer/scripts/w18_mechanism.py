"""What W18's held-out generator asked on sidestep, and what the bodies did with it
(``skill_space.md`` §10.19, read after the result; instrumentation, not a criterion).

From W18's L4 captures (after the 100 recorded settle steps, living environments), per body,
command and arm, for the plans (``contract_due``) and the bodies (``skill_state``), averaged
over the three trackers; and for sidestep's teacher:

- lateral foot separation (left ``y`` minus right ``y``, in ``L``): median, 5th percentile, and
  the share under 0.02 (feet together or crossed);
- CoM height, the shares of double support and of flight (loads above 0.1 of the weight);
- two checks of the plans' physics:
  - ``dcm``: the share of single support on the leading foot with the lateral DCM beyond it;
  - ``cop``: the share of supported steps with the CoP implied by the CoM's acceleration
    (``p - c = -h a``, dimensionless) outside the loaded feet by more than 0.05 ``L`` sideways
    and 0.10 ``L`` lengthwise.

    python transfer/scripts/w18_mechanism.py --output transfer/results/w18_mechanism.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.skill_state import dcm_offset  # noqa: E402

TRACKERS = {
    "g1": ("mt_g1_s41_sym", "mt_g1_s42_sym", "mt_g1_s43_sym"),
    "nao": ("mt_nao_s31_sym", "mt_nao_s32_sym", "mt_nao_s33_sym"),
}
ARMS = ("mt17_c", "mt18_no_sidestep", "mt18_no_sidestep_m", "mt17_m")
DIRECTION = {"left": 1, "left_fast": 1, "right": -1}
SETTLE = 100
LOADED = 0.1
TOGETHER = 0.02
MARGIN = 0.05


def dt_star(data) -> float:
    return float(data["control_dt"]) / float(np.sqrt(float(data["leg_length"]) / 9.81))


def measures(z: np.ndarray, alive: np.ndarray, sign: int, dt: float) -> dict:
    """``z`` (T, E, 27) dimensionless, ``alive`` (T, E)."""
    flat = z[alive]
    sep = flat[:, 11] - flat[:, 14]
    load = flat[:, 22:24] > LOADED
    lead = 0 if sign > 0 else 1
    single = load[:, lead] & ~load[:, 1 - lead]
    beyond = sign * (dcm_offset(flat)[:, 1] - flat[:, 11 + 3 * lead])[single]
    # The implied CoP, from the CoM velocity's central difference.
    v = z[:, :, 1:3]
    a = (v[2:] - v[:-2]) / (2 * dt)
    mid = z[1:-1]
    ok = alive[2:] & alive[:-2] & alive[1:-1]
    cop = -mid[..., :1] * a
    loaded = mid[..., 22:24] > LOADED
    big = 1e9
    fy = mid[..., [11, 14]]
    fx = mid[..., [10, 13]]
    ey = np.maximum(np.where(loaded, fy, big).min(-1) - MARGIN - cop[..., 1], 0) + np.maximum(
        cop[..., 1] - np.where(loaded, fy, -big).max(-1) - MARGIN, 0
    )
    ex = np.maximum(np.where(loaded, fx, big).min(-1) - 2 * MARGIN - cop[..., 0], 0) + np.maximum(
        cop[..., 0] - np.where(loaded, fx, -big).max(-1) - 2 * MARGIN, 0
    )
    supported = loaded.any(-1) & ok
    outside = np.hypot(ex, ey)[supported] > 0
    return {
        "separation_median": float(np.median(sep)),
        "separation_p05": float(np.percentile(sep, 5)),
        "together_share": float((sep < TOGETHER).mean()),
        "height": float(flat[:, 0].mean()),
        "double_support": float((load.sum(-1) == 2).mean()),
        "flight": float((load.sum(-1) == 0).mean()),
        "dcm_beyond_leading_foot": float((beyond > 0).mean()) if beyond.size else None,
        "cop_outside_support": float(outside.mean()),
    }


def mean(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict = {"teacher": {}, "bodies": {}}
    for command, sign in DIRECTION.items():
        data = np.load(REPO_ROOT / f"g1/rollouts/base_sidestep_play_{command}.npz")
        alive = ~data["done"][SETTLE:]
        report["teacher"][command] = measures(
            data["skill_state"][SETTLE:], alive, sign, dt_star(data)
        )
    for body, tags in TRACKERS.items():
        report["bodies"][body] = {}
        for command, sign in DIRECTION.items():
            entry = {}
            for arm in ARMS:
                asked, executed, falls = [], [], 0
                for tag in tags:
                    path = (
                        REPO_ROOT
                        / body
                        / "rollouts"
                        / f"zs_{tag}_{arm}_own_auto_kin_{command}_encoder.npz"
                    )
                    data = np.load(path)
                    alive = ~data["done"][SETTLE:]
                    dt = dt_star(data)
                    asked.append(measures(data["contract_due"][SETTLE:], alive, sign, dt))
                    executed.append(measures(data["skill_state"][SETTLE:], alive, sign, dt))
                    falls += int(data["terminated"].sum())
                entry[arm] = {"falls": falls, "asked": mean(asked), "body": mean(executed)}
                a, b = entry[arm]["asked"], entry[arm]["body"]
                print(
                    f"{body} {command:9s} {arm:20s} falls {falls:4d}  separation p05 asked "
                    f"{a['separation_p05']:.3f} body {b['separation_p05']:.3f}  double "
                    f"{a['double_support']:.3f}  flight {a['flight']:.3f}  height {a['height']:.3f}"
                )
            report["bodies"][body][command] = entry
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
