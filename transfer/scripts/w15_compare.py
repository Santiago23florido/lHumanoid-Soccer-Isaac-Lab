"""W15's pre-registered reading (``skill_space.md`` §10.16): the family's share alone, on the NAO.

One NAO tracker, S's recipe with the family's share at 0.5 and no task plans
(``fam05_nao_s31_sym``), against the two seed-31 trackers it lies between: S's
(``warm_nao_r1_s31_sym``, family 0.7, no task plans) and W13's (``mt_nao_s31_sym``, family 0.5
with the G1's task plans). All three are read on the NAO's pinned path (W14).

With ``d`` the arm's flat largest deviation (mean over the three commands), ``s`` and ``w``
S's and W13's, and ``g = w - s``:

- ``task plans`` if ``d <= s + g/3``: the family's share does not regress the NAO;
- ``family share`` if ``d >= w - g/3``;
- ``both`` otherwise.

Reported beside it, not read: the falls, S0's walk, the body's duty on flat on both paths,
and jog's largest deviation at its middle command.

    python transfer/scripts/w15_compare.py --output transfer/results/w15_family.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from w8_compare import COMMANDS, RESULTS, condition  # noqa: E402
from w13_compare import walk  # noqa: E402

ARM = "fam05_nao_s31_sym"
S = "warm_nao_r1_s31_sym"
W13 = "mt_nao_s31_sym"
FLAT = "flat_dagger2_own"
JOG = "base_jog_b4_own"


def duty(tag: str, path: str) -> float | None:
    """The body's duty on flat on ``path`` (encoder or truth), over commands and feet."""
    values = []
    for command in COMMANDS:
        file = RESULTS / f"zero_shot_nao_{tag}_{FLAT}_auto_kin_l4_{command}.json"
        if not file.exists():
            return None
        runs = json.loads(file.read_text(encoding="utf-8"))["runs"]
        if path not in runs:
            return None
        values.append(float(np.mean(runs[path]["signature"]["duty_factor"])))
    return float(np.mean(values))


def tracker(tag: str) -> dict | None:
    rows = [condition("nao", tag, FLAT, c) for c in COMMANDS]
    if any(r is None for r in rows):
        return None
    jog = condition("nao", tag, JOG, "mid")
    return {
        "flat_largest": float(np.mean([r["largest"] for r in rows])),
        "flat_falls": int(sum(r["falls"] for r in rows)),
        "flat_speed_ratio_min": float(min(r["speed_ratio"] for r in rows)),
        "duty_deployed": duty(tag, "encoder"),
        "duty_true_state": duty(tag, "truth"),
        "jog_largest": None if jog is None else jog["largest"],
        "walk": walk("nao", tag),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {tag: tracker(tag) for tag in (S, ARM, W13)}
    report: dict = {"rows": rows, "verdict": "incomplete"}
    if all(r is not None for r in rows.values()):
        d, s, w = (rows[t]["flat_largest"] for t in (ARM, S, W13))
        g = w - s
        if d <= s + g / 3:
            verdict = "task plans"
        elif d >= w - g / 3:
            verdict = "family share"
        else:
            verdict = "both"
        report |= {"d": d, "s": s, "w": w, "gap": g, "verdict": verdict}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    if report["verdict"] == "incomplete":
        print("incomplete")
    else:
        print(
            f"{report['verdict']}: d {report['d']:.3f} between S {report['s']:.3f} "
            f"and W13 {report['w']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
