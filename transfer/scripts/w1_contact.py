"""Where W1's asked duty went: the plan's contact events against the body's, both interfaces.

W1 (skill_space.md §10.3) lowered the duty the NAO is asked for; this reads, from the
contact-fidelity scores of its captures, whether the body's contact events moved with
the plan's. Each linear capture is paired with the same tracker, skill and command under
``HUMANOID_RESAMPLE=generator``; per skill, the mean over trackers, commands and feet of
each metric under both interfaces, and of the paired difference with its standard
deviation over the nine pairs. Deployed path (``encoder`` run).

    python transfer/scripts/contact_fidelity.py \
        --glob "zero_shot_nao_r1_3498_sym_v3_linrep_*_l4_*.json" \
        --glob "zero_shot_nao_r1_3498_sym_v3_gen_*_l4_*.json" \
        --glob "zero_shot_nao_warm_nao_r1_s3[23]_sym_*_l4_*.json" \
        --output transfer/results/contact_fidelity_w1.json
    python transfer/scripts/w1_contact.py --scores transfer/results/contact_fidelity_w1.json \
        --output transfer/results/w1_contact.json

The incumbent's linear side is its control re-run (``_linrep``), captured under the same
code as the generator side.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

METRICS = (
    "duty_asked",
    "duty_body",
    "liftoff_lag_mean",
    "touchdown_lag_mean",
    "graze_in_swing",
    "airborne_when_light",
    "clearance_asked",
    "clearance_body",
)
KEY = re.compile(
    r"zero_shot_nao_(?P<tracker>.+?)(?P<gen>_gen)?_(?P<skill>flat_dagger2|rough_b4)"
    r"_auto_kin_l4_(?P<command>\w+)\.json$"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = json.loads(args.scores.read_text(encoding="utf-8"))["reports"]
    sides: dict[tuple, dict[bool, dict]] = {}
    for r in reports:
        m = KEY.search(r["result"].replace("\\", "/"))
        if m is None:
            continue
        tracker = m["tracker"].removesuffix("_linrep")
        run = dict(r["runs"]["encoder"], duty_teacher=r["teacher"]["duty_body"])
        sides.setdefault((tracker, m["skill"], m["command"]), {})[bool(m["gen"])] = run
    out = {}
    for skill in ("flat_dagger2", "rough_b4"):
        pairs = [v for k, v in sides.items() if k[1] == skill and len(v) == 2]
        teacher = np.array([np.mean(p[False]["duty_teacher"]) for p in pairs])
        row = {"pairs": len(pairs), "duty_teacher_own_grid": float(teacher.mean())}
        for interface, side in (("linear", False), ("generator", True)):
            asked = np.array([np.mean(p[side]["duty_asked"]) for p in pairs])
            body = np.array([np.mean(p[side]["duty_body"]) for p in pairs])
            row[f"above_teacher_{interface}"] = {
                "asked": float((asked - teacher).mean()),
                "body": float((body - teacher).mean()),
            }
        print(f"{skill} ({len(pairs)} pairs): duty above the teacher's own grid, asked / body")
        for interface in ("linear", "generator"):
            v = row[f"above_teacher_{interface}"]
            print(f"  {interface:9} {v['asked']:+.4f} / {v['body']:+.4f}")
        for metric in METRICS:
            linear = np.array([np.mean(p[False][metric]) for p in pairs])
            generator = np.array([np.mean(p[True][metric]) for p in pairs])
            diff = generator - linear
            row[metric] = {
                "linear": float(linear.mean()),
                "generator": float(generator.mean()),
                "difference": float(diff.mean()),
                "difference_sd": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
            }
        out[skill] = row
        print(f"{skill}: paired, linear against generator")
        for metric in METRICS:
            v = row[metric]
            print(
                f"  {metric:20} linear {v['linear']:.4f}  generator {v['generator']:.4f}  "
                f"difference {v['difference']:+.4f} (sd {v['difference_sd']:.4f})"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
