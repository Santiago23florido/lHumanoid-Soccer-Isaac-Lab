"""Does a tracker cut short the load transfers it is asked for, inside the family? (W3, step 1)

Reads a capture of ``transfer/probes/transfer_probes.json`` (``make_gait_probes.py --transfer``):
twelve walking schedules, each written with three stance-load warps (0.5: loaded at
touchdown, slow unloading; 1: the half-sine every tracker practised; 2: its mirror). Per
probe, the body is scored against its own plan with ``common/contact.py`` at the 5 %
threshold gate C uses, and the evaluation's own judge for reference. Each warp is then
paired with the half-sine schedule by schedule:

* the stance shortfall, ``duty_body − duty_asked`` (negative: the foot is up when it
  should be loaded);
* ``airborne_when_light``: the foot in the air where the plan asks for a light contact;
* lift-off and touchdown lags.

The prediction pre-registered in ``skill_space.md`` §10.5 holds if, at warp 0.5, the
shortfall is more negative and the airborne share larger than at the half-sine, each by
more than twice the standard error of the twelve paired differences.

    python transfer/scripts/transfer_probe_effect.py \
        g1/rollouts/zs_warm_r3_logload_sym_transfer.npz \
        --output transfer/results/transfer_probes_g1_warm_r3_logload_sym.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import contact_fidelity  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402

METRICS = ("shortfall", "airborne_when_light", "liftoff_lag_mean", "touchdown_lag_mean")


def probe_metrics(data, envs: np.ndarray, dt_star: float) -> dict:
    """``common.contact`` on the probe's environments, each plan against the state after it."""
    z = data["skill_state"][:, envs].astype(float)
    done = data["done"][:, envs].astype(bool)
    due = data["contract_due"][:-1, envs].astype(float)
    cut = done[:-1].copy()
    cut[:-1] |= done[1:-1]
    f = contact_fidelity(z[1:], cut, dt_star, asked=due).as_dict()
    return {
        "duty_body": f["duty_body"],
        "duty_asked": f["duty_asked"],
        "shortfall": [b - a for b, a in zip(f["duty_body"], f["duty_asked"], strict=True)],
        "light_share_asked": f["light_share_asked"],
        "light_share_body": f["light_share_body"],
        "airborne_when_light": f["airborne_when_light"],
        "liftoff_lag_mean": f["liftoff_lag_mean"],
        "touchdown_lag_mean": f["touchdown_lag_mean"],
        "falls": int(data["terminated"][:, envs].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.capture)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    dt_star = float(data["control_dt"]) / units.time
    probes = json.loads(str(data["probes"]))
    index = data["probe_index"]
    per_probe = {}
    for i, probe in enumerate(probes):
        envs = np.flatnonzero(index == i)
        if len(envs):
            per_probe[probe["name"]] = {
                "warp": probe["load_shape"],
                "schedule": probe["name"].rsplit("_k", 1)[0],
                **probe_metrics(data, envs, dt_star),
            }

    by_schedule: dict = defaultdict(dict)
    for r in per_probe.values():
        by_schedule[r["schedule"]][r["warp"]] = r
    summary = {}
    for warp in (0.5, 1.0, 2.0):
        rows = [s[warp] for s in by_schedule.values() if warp in s]
        summary[f"warp{warp}"] = {
            m: np.nanmean([r[m] for r in rows], axis=0).tolist() for m in METRICS
        } | {
            "light_share_asked": np.mean([r["light_share_asked"] for r in rows], axis=0).tolist(),
            "falls": int(sum(r["falls"] for r in rows)),
            "probes": len(rows),
        }
    paired = {}
    for warp in (0.5, 2.0):
        paired[f"warp{warp}_minus_half_sine"] = {}
        for m in METRICS:
            diffs = np.asarray(
                [s[warp][m] for s in by_schedule.values() if warp in s and 1.0 in s], dtype=float
            ) - np.asarray(
                [s[1.0][m] for s in by_schedule.values() if warp in s and 1.0 in s], dtype=float
            )
            diffs = diffs[np.isfinite(diffs).all(axis=1)]
            mean = diffs.mean(axis=0)
            se = diffs.std(axis=0, ddof=1) / np.sqrt(len(diffs))
            paired[f"warp{warp}_minus_half_sine"][m] = {
                "mean": mean.tolist(),
                "se": se.tolist(),
                "pairs": int(len(diffs)),
            }
    p = paired["warp0.5_minus_half_sine"]
    verdict = {
        foot: bool(
            p["shortfall"]["mean"][k] < -2 * p["shortfall"]["se"][k]
            and p["airborne_when_light"]["mean"][k] > 2 * p["airborne_when_light"]["se"][k]
        )
        for k, foot in enumerate(("left", "right"))
    }
    report = {
        "capture": args.capture.as_posix(),
        "dt_star": dt_star,
        "summary": summary,
        "paired": paired,
        "prediction_holds": verdict,
        "probes": per_probe,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def pair(v) -> str:
        return "/".join(f"{x:+.3f}" for x in v)

    print(args.capture.name)
    for warp, s in summary.items():
        print(
            f"  {warp:<8} shortfall {pair(s['shortfall'])}  airborne|light "
            f"{pair(s['airborne_when_light'])}  light asked {pair(s['light_share_asked'])}  "
            f"falls {s['falls']}"
        )
    for name, d in paired.items():
        line = "  ".join(
            f"{m} {pair(d[m]['mean'])} ± {'/'.join(f'{x:.3f}' for x in d[m]['se'])}"
            for m in METRICS
        )
        print(f"  {name}: {line}")
    print(f"  prediction holds: {verdict}")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
