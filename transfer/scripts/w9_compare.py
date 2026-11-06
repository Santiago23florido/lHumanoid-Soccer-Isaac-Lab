"""W9's pre-registered reading (skill_space.md §10.11): the plans' loaded feet grounded and still.

For each body's three frozen trackers, the captures with ``HUMANOID_PLAN_CONTACT=stick``
(``w9_stick.sh``) against a baseline capture of the same tracker, deployed path, L4 of both
skills at three commands, and S0's walk. The features are ``w8_compare.features``.

* **G1** -- against W8's ``ground`` captures, the G1's recorded candidate. Target: rough's
  largest gate C deviation falls by more than twice the standard error of the three paired
  seed differences, and flat's rises by no more than that.
* **NAO** -- against the plans as drawn, the NAO's deployment (W8's grounding hurt it):
  W8's NAO criterion, the duty deviation over both skills falling by more than 2 SE.
  ``ground`` is reported beside it.
* **No regression** -- falls (summed per body, at most two more), the speed ratio (at least
  0.9 at every command), S0's walk (speed ratio at least 0.9, fall excess at most 0.02).
* **Beside them** -- the bodies' loaded-foot horizontal speed (``plan_slip.loaded_motion``),
  per tracker and skill: whether a body drags its stance foot less.

    python transfer/scripts/w9_compare.py --output transfer/results/w9_stick.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_contact import aligned  # noqa: E402
from plan_slip import loaded_motion  # noqa: E402
from w8_compare import COMMANDS, CONTACT, SKILLS, TRACKERS, features, paired  # noqa: E402

BASELINE = {"g1": "ground", "nao": "as_drawn"}
BESIDE = {"g1": "as_drawn", "nao": "ground"}


def tagged(tag: str, contact: str) -> str:
    return tag if contact == "as_drawn" else f"{tag}_{contact}"


def body_drag(body: str, tag: str) -> dict | None:
    """Median horizontal speed of the body's loaded feet, per skill."""
    out = {}
    for skill in SKILLS:
        bodies = []
        for command in COMMANDS:
            path = (
                REPO_ROOT / body / "rollouts" / f"zs_{tag}_{skill}_auto_kin_{command}_encoder.npz"
            )
            if not path.exists():
                return None
            executed, _, valid, _ = aligned(path)
            bodies.append(executed[valid])
        motion = loaded_motion(np.concatenate(bodies))
        out[skill] = motion["loaded"]["horizontal_speed"]["median"]
    return out


def compare(body: str, baseline: str) -> dict:
    pairs = []
    for tag in TRACKERS[body]:
        a, b = features(body, tagged(tag, baseline)), features(body, tagged(tag, "stick"))
        if a is None or b is None:
            print(f"{body} {tag}: results missing ({baseline} or stick)")
            continue
        pairs.append({"tracker": tag, "before": a, "after": b})
    entry: dict = {"baseline": baseline, "pairs": pairs}
    if len(pairs) < 2:
        return entry
    keys = ("largest", "duty", *CONTACT)
    diff = {skill: {k: paired(pairs, skill, k) for k in keys} for skill in SKILLS}
    for p in pairs:
        for when in ("before", "after"):
            p[when]["both"] = {
                "duty": (p[when]["flat_dagger2"]["duty"] + p[when]["rough_b4"]["duty"]) / 2
            }
    both = paired(pairs, "both", "duty")
    if body == "g1":
        rough, flat = diff["rough_b4"]["largest"], diff["flat_dagger2"]["largest"]
        target = rough["mean"] < -2 * rough["se"] and flat["mean"] <= 2 * flat["se"]
    else:
        target = both["mean"] < -2 * both["se"]
    regressions = {
        "falls": sum(p["after"][s]["falls"] for p in pairs for s in SKILLS)
        <= sum(p["before"][s]["falls"] for p in pairs for s in SKILLS) + 2,
        "speed": all(p["after"][s]["speed_ratio_min"] >= 0.9 for p in pairs for s in SKILLS),
        "walk": all(
            p["after"]["walk"] is not None
            and p["after"]["walk"]["speed_ratio"] >= 0.9
            and p["after"]["walk"]["fall_rate_excess"] <= 0.02
            for p in pairs
        ),
    }
    entry.update(
        difference=diff,
        duty_both_skills=both,
        target_met=bool(target),
        no_regression=regressions,
        passes=bool(target and all(regressions.values())),
    )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bodies", nargs="+", choices=tuple(TRACKERS), default=list(TRACKERS))
    args = parser.parse_args()
    report: dict = {}
    for body in args.bodies:
        entry = compare(body, BASELINE[body])
        entry["beside"] = {
            k: v for k, v in compare(body, BESIDE[body]).items() if k not in ("pairs",)
        }
        entry["body_drag"] = {
            contact: {tag: body_drag(body, tagged(tag, contact)) for tag in TRACKERS[body]}
            for contact in ("as_drawn", "ground", "stick")
        }
        report[body] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        b: {k: v for k, v in e.items() if k not in ("pairs", "body_drag")}
        for b, e in report.items()
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
