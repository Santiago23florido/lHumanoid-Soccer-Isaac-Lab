"""W17's pre-registered reading (``skill_space.md`` §10.18): the command a future achieved.

Arms, each a generator (``logs/contract/<arm>``) deployed on W13's six trackers (G1 s41-43, NAO
s31-33), rough's L4 at its three commands on the deployed path, in each tracker's own units, as
W16 (``w16_compare.py``), whose ``mt16_c`` and ``rough_b4`` captures are reused:

- ``mt17_c``: the G1 experts' five base tasks, every command relabelled with the one its future
  achieved (``latent/hindsight.py``);
- ``mt17_m``: ``mt17_c`` then the bodies' histories with the teacher's phase-matched futures,
  each with its target's achieved command;
- ``mt17_l``: as ``mt17_m``, its history summary held to carry no body by an MMD constraint;
  evaluated only if its leakage probe does not beat its control by more than 3 SE.

Readings, per body, paired by tracker (mean ± SE over three), on rough's largest gate C
deviation (the mean over the three commands):

1. **relabel** -- ``mt17_c - mt16_c``: "help" below -2 SE, "hurt" above +2 SE, else "none";
2. **cost** -- ``mt17_c - rough_b4``, and the share of W16's cost closed,
   ``1 - (mt17_c - rough_b4) / (mt16_c - rough_b4)`` on the means; reported;
3. **bodies** -- ``mt17_m - mt17_c``, as 1;
4. **latent** -- ``mt17_l - mt17_m``, as 1;
5. **no regression** -- per arm against ``mt17_c``: falls at most two more over the three
   trackers, and every command's speed ratio at least 0.9.

    python transfer/scripts/w17_compare.py --precheck transfer/results/held_out_chains_w17_pre.json
    python transfer/scripts/w17_compare.py --leakage-gate
    python transfer/scripts/w17_compare.py --output transfer/results/w17_relabel.json

``--precheck`` is the gate before any Isaac run: ``mt17_c`` chained on its own plans from rough's
histories walks a speed ratio within 0.9-1.1 at every command, and its mean largest deviation is
below ``mt16_c``'s (``held_out_chains_w16.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from w8_compare import RESULTS  # noqa: E402
from w16_compare import (  # noqa: E402
    COMMANDS,
    FALLS_MORE,
    LEAK_SE,
    SPEED_MIN,
    TRACKERS,
    difference,
    generator_report,
    per_command,
    reading,
    sign,
)

ARMS = ("mt17_c", "mt17_m", "mt17_l", "mt16_c", "rough_b4")
NEW = ("mt17_c", "mt17_m", "mt17_l")
SPEED_BAND = (0.9, 1.1)
BASELINE_CHAINS = RESULTS / "held_out_chains_w16.json"


def precheck(path: Path) -> tuple[bool, str]:
    chains = json.loads(path.read_text(encoding="utf-8"))["generators"]["mt17_c"]
    baseline = json.loads(BASELINE_CHAINS.read_text(encoding="utf-8"))["generators"]["mt16_c"]
    speeds = [chains[c]["speed_ratio"] for c in COMMANDS]
    lo, hi = SPEED_BAND
    in_band = all(lo <= s <= hi for s in speeds)
    better = chains["largest_mean"] < baseline["largest_mean"]
    passes = in_band and better
    return passes, (
        f"{'passes' if passes else 'fails'}: chain speed ratios "
        f"{', '.join(f'{s:.3f}' for s in speeds)} (band {lo}-{hi}); mean largest "
        f"{chains['largest_mean']:.3f} against mt16_c's {baseline['largest_mean']:.3f}"
    )


def leakage_gate(arm: str = "mt17_l") -> tuple[bool, str]:
    entry = generator_report(arm)
    if entry is None or entry["leakage"] is None:
        return False, f"{arm} has no leakage measurement"
    leak = entry["leakage"]
    margin = leak["balanced_accuracy"] - leak["control"]
    passes = margin <= LEAK_SE * leak["se"]
    return passes, (
        f"{'passes' if passes else 'fails'}: balanced accuracy {leak['balanced_accuracy']:.3f}, "
        f"control {leak['control']:.3f}, 3 SE {LEAK_SE * leak['se']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precheck", type=Path)
    parser.add_argument("--leakage-gate", action="store_true")
    args = parser.parse_args()
    if args.precheck is not None:
        passes, line = precheck(args.precheck)
        print(line)
        return 0 if passes else 1
    if args.leakage_gate:
        passes, line = leakage_gate()
        print(line)
        return 0 if passes else 1
    report: dict = {
        "generators": {arm: generator_report(arm) for arm in NEW},
        "leakage_gate": leakage_gate()[1],
        "per_command": per_command(ARMS),
        "bodies": {},
    }
    summary = []
    for body, tags in TRACKERS.items():
        rows = {t: {arm: reading(body, t, arm) for arm in ARMS} for t in tags}
        relabel = difference(rows, "mt17_c", "mt16_c")
        cost = difference(rows, "mt17_c", "rough_b4")
        cost_before = difference(rows, "mt16_c", "rough_b4")
        closed = None
        if cost is not None and cost_before is not None and cost_before["mean"] != 0:
            closed = 1.0 - cost["mean"] / cost_before["mean"]
        bodies = difference(rows, "mt17_m", "mt17_c")
        latent = difference(rows, "mt17_l", "mt17_m")
        regression = {}
        for arm in NEW:
            if any(rows[t][arm] is None or rows[t]["mt17_c"] is None for t in tags):
                continue
            falls = sum(rows[t][arm]["falls"] for t in tags)
            control = sum(rows[t]["mt17_c"]["falls"] for t in tags)
            speed = min(rows[t][arm]["speed_ratio_min"] for t in tags)
            regression[arm] = {
                "falls": falls,
                "falls_control": control,
                "speed_ratio_min": speed,
                "passes": bool(falls <= control + FALLS_MORE and speed >= SPEED_MIN),
            }
        entry = {
            "rows": rows,
            "relabel": None if relabel is None else {**relabel, "verdict": sign(relabel)},
            "cost": cost,
            "cost_w16": cost_before,
            "share_closed": closed,
            "bodies": None if bodies is None else {**bodies, "verdict": sign(bodies)},
            "latent": None if latent is None else {**latent, "verdict": sign(latent)},
            "no_regression": regression,
        }
        report["bodies"][body] = entry
        verdict = entry["relabel"]["verdict"] if entry["relabel"] else "incomplete"
        part = [f"{body}: relabel {verdict}"]
        if closed is not None:
            part.append(f"closed {closed:.2f}")
        if entry["bodies"]:
            part.append(f"bodies {entry['bodies']['verdict']}")
        if entry["latent"]:
            part.append(f"latent {entry['latent']['verdict']}")
        summary.append(", ".join(part))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print("; ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
