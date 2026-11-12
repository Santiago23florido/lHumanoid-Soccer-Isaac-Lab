"""W6's reading: the NAO's sole point at its contact.

The arm (``warm_nao_sole_s<seed>``: r1_3498 continued as S's NAO seeds, with
``HUMANOID_SOLE=contact``) against S's NAO seeds 31-33, on the L4 captures of the deployed
path (autonomous reference, kinematic estimator), both skills x three commands:

* **target** -- the clearance ratio, reached over asked, each above its own stance level
  (``common.contact``), averaged over feet, skills and commands;
* **screen** -- with only seed 31, whether its ratio is above every control seed's;
* **with three seeds** -- the arm's seed mean above the control's by more than twice the
  standard error of the difference of seed means, ``sqrt(s_a^2/n_a + s_c^2/n_c)``; no
  regression: falls (not more than the control's + 2), speed ratio at least 0.9 at every
  command, the largest deviation and the duty deviation not worse by 2 SE, S0's walk.

    python transfer/scripts/w6_compare.py --output transfer/results/w6_sole.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import capture_fidelity  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402

RESULTS = REPO_ROOT / "transfer" / "results"
ROLLOUTS = REPO_ROOT / "nao" / "rollouts"
CONTROL = {31: "warm_nao_r1_s31_sym", 32: "warm_nao_r1_s32_sym", 33: "warm_nao_r1_s33_sym"}
ARM = {seed: f"warm_nao_sole_s{seed}_sym" for seed in (31, 32, 33)}
SKILLS = ("flat_dagger2", "rough_b4")
COMMANDS = ("forward", "fast", "turning")


def clearance_ratio(tag: str, generator: str, command: str) -> float | None:
    path = ROLLOUTS / f"zs_{tag}_{generator}_auto_kin_{command}_encoder.npz"
    if not path.exists():
        return None
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    fidelity = capture_fidelity(data, float(data["control_dt"]) / units.time)
    return float(np.mean(fidelity.clearance_body) / np.mean(fidelity.clearance_asked))


def features(tag: str) -> dict | None:
    rows = []
    for generator in SKILLS:
        for command in COMMANDS:
            path = RESULTS / f"zero_shot_nao_{tag}_{generator}_auto_kin_l4_{command}.json"
            ratio = clearance_ratio(tag, generator, command)
            if not path.exists() or ratio is None:
                return None
            run = json.loads(path.read_text(encoding="utf-8"))["runs"]["encoder"]
            dev = run["verdict"]["signature_deviation"]
            rows.append(
                {
                    "clearance_ratio": ratio,
                    "duty": (abs(dev["duty_factor_0"]) + abs(dev["duty_factor_1"])) / 2,
                    "largest": max(abs(v) for v in dev.values()),
                    "speed_ratio": run["verdict"]["speed_ratio"],
                    "falls": run["falls"],
                }
            )
    walk_path = RESULTS / f"zero_shot_nao_{tag}_walk_auto_kin.json"
    walk = None
    if walk_path.exists():
        w = json.loads(walk_path.read_text(encoding="utf-8"))["runs"]["encoder"]["verdict"]
        walk = {"speed_ratio": w["speed_ratio"], "fall_rate_excess": w["fall_rate_excess"]}
    return {
        "clearance_ratio": float(np.mean([r["clearance_ratio"] for r in rows])),
        "duty": float(np.mean([r["duty"] for r in rows])),
        "largest": float(np.mean([r["largest"] for r in rows])),
        "speed_ratio_min": float(min(r["speed_ratio"] for r in rows)),
        "falls": int(sum(r["falls"] for r in rows)),
        "walk": walk,
        "per_command": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = {s: f for s, t in CONTROL.items() if (f := features(t)) is not None}
    arm = {s: f for s, t in ARM.items() if (f := features(t)) is not None}
    report: dict = {"control": control, "arm": arm}
    if control and 31 in arm:
        best = max(f["clearance_ratio"] for f in control.values())
        report["screen"] = {
            "arm_s31": arm[31]["clearance_ratio"],
            "control_max": best,
            "passes": bool(arm[31]["clearance_ratio"] > best),
        }
    if len(arm) >= 3 and len(control) >= 3:
        stats = {}
        for key in ("clearance_ratio", "duty", "largest", "falls"):
            a = np.array([f[key] for f in arm.values()], dtype=float)
            c = np.array([f[key] for f in control.values()], dtype=float)
            se = float(np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)))
            stats[key] = {"arm": float(a.mean()), "control": float(c.mean()), "se": se}
        diff = {k: v["arm"] - v["control"] for k, v in stats.items()}
        regressions = {
            "falls": sum(f["falls"] for f in arm.values())
            <= sum(f["falls"] for f in control.values()) + 2,
            "speed": all(f["speed_ratio_min"] >= 0.9 for f in arm.values()),
            "largest": diff["largest"] <= 2 * stats["largest"]["se"],
            "duty": diff["duty"] <= 2 * stats["duty"]["se"],
            "walk": all(
                f["walk"] is not None
                and f["walk"]["speed_ratio"] >= 0.9
                and f["walk"]["fall_rate_excess"] <= 0.02
                for f in arm.values()
            ),
        }
        report["comparison"] = stats
        report["target_met"] = bool(diff["clearance_ratio"] > 2 * stats["clearance_ratio"]["se"])
        report["no_regression"] = regressions
        report["passes"] = bool(report["target_met"] and all(regressions.values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, group in (("control", control), ("arm", arm)):
        for seed, f in sorted(group.items()):
            print(
                f"{name:7s} s{seed}: clearance {f['clearance_ratio']:.3f}  "
                f"duty dev {f['duty']:.3f}  largest {f['largest']:.3f}  "
                f"speed min {f['speed_ratio_min']:.2f}  "
                f"falls {f['falls']}"
            )
    for key in ("screen", "target_met", "no_regression", "passes"):
        if key in report:
            print(f"{key}: {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
