"""How high a tracker lifts its feet against what the family's probes ask.

For each probe capture (the competence map's grid and corners), the swing clearance the body
reaches against the plan's own (``common.contact``: the sole's highest point in each swing
above its stance level, median over swings), grouped by the probe's clearance and period.

    python transfer/scripts/clearance_probes.py \
        --capture nao/rollouts/zs_r1_3498_sym_f5_corners.npz \
        --capture g1/rollouts/zs_warm_r3_logload_sym_corners.npz \
        --output transfer/results/clearance_probes.json
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for path in args.capture:
        data = np.load(path)
        units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
        dt_star = float(data["control_dt"]) / units.time
        probes = json.loads(str(data["probes"]))
        index = data["probe_index"]
        groups: dict = defaultdict(lambda: {"asked": [], "body": []})
        for i, probe in enumerate(probes):
            envs = np.flatnonzero(index == i)
            if not len(envs) or probe.get("group") not in ("grid", "corner"):
                continue
            z = data["skill_state"][:, envs].astype(float)
            done = data["done"][:, envs].astype(bool)
            due = data["contract_due"][:-1, envs].astype(float)
            cut = done[:-1].copy()
            cut[:-1] |= done[1:-1]
            f = contact_fidelity(z[1:], cut, dt_star, asked=due)
            key = f"clearance {probe['clearance']:.2f}, period {probe['period']:.1f}"
            groups[key]["asked"].append(float(np.mean(f.clearance_asked)))
            groups[key]["body"].append(float(np.mean(f.clearance_body)))
        rows = {}
        for key, g in sorted(groups.items()):
            asked, body = np.nanmean(g["asked"]), np.nanmean(g["body"])
            rows[key] = {
                "asked": asked,
                "body": body,
                "ratio": body / asked,
                "probes": len(g["body"]),
            }
        report[path.as_posix()] = {"dt_star": dt_star, "groups": rows}
        print(path.name)
        for key, r in rows.items():
            print(
                f"  {key}: asked {r['asked']:.3f} L, body {r['body']:.3f} L, ratio {r['ratio']:.2f}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
