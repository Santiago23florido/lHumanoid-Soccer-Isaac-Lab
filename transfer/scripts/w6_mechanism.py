"""Where W6's swing went: the feet's stance level and swing peak, arm against control.

W6 (skill_space.md §10.8) lowered the NAO's sole point to its resting contact. Its screen
reads the clearance ratio, which is measured above each foot's own stance level, so it
cannot say whether the switch moved the stance level, the swing peak or both. This reads
the L4 captures of the deployed path (both skills x three commands) of S's NAO seeds and
W6's arm. Per capture, plan ``contract_due[t]`` against the state at ``t + 1``
(``common.contact.capture_fidelity``'s alignment), each sole point's height
(``common.contact.foot_height``):

* **stance level** -- median over each foot's contact samples (load above the threshold);
* **peak** -- per swing (a run of samples without contact), the highest point, absolute;
  median over swings;
* **clearance** -- ``capture_fidelity``'s, peak above the foot's own stance level;
* **air time** -- the median swing's length, body over plan;
* the duty and the lift-off and touchdown lags (``capture_fidelity``).

Everything is the mean over the six captures, then per skill. Leg lengths throughout.
``arm_s31_ground`` is the same tracker on W8n's grounded plans (§10.10).

    python transfer/scripts/w6_mechanism.py --output transfer/results/w6_mechanism.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import _runs, capture_fidelity, foot_height  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)

ROLLOUTS = REPO_ROOT / "nao" / "rollouts"
TAGS = {
    "control_s31": "warm_nao_r1_s31_sym",
    "control_s32": "warm_nao_r1_s32_sym",
    "control_s33": "warm_nao_r1_s33_sym",
    "arm_s31": "warm_nao_sole_s31_sym",
    "arm_s31_ground": "warm_nao_sole_s31_sym_ground",
}
SKILLS = ("flat_dagger2", "rough_b4")
COMMANDS = ("forward", "fast", "turning")


def swings(data) -> dict:
    """Stance level, absolute peak and swing length, body and plan, medians."""
    z = data["skill_state"].astype(float)
    done = data["done"].astype(bool)
    body, asked = z[1:], data["contract_due"][:-1].astype(float)
    cut = done[:-1].copy()
    cut[:-1] |= done[1:-1]
    valid = np.ones_like(cut)
    valid[1:] = ~cut[:-1]
    valid[0] = False
    out: dict[str, list[float]] = {}
    for who, state in (("body", body), ("asked", asked)):
        height = foot_height(state)
        contact = SKILL_LAYOUT.unpack(state)["foot_load"] > DEFAULT_CONTACT_THRESHOLD
        level, peak, length = [], [], []
        for side in range(2):
            for env in range(state.shape[1]):
                h, c, v = height[:, env, side], contact[:, env, side], valid[:, env]
                if (c & v).any():
                    level.append(float(np.median(h[c & v])))
                for start, stop in _runs(~c & v):
                    peak.append(float(h[start:stop].max()))
                    length.append(stop - start)
        out[f"stance_level_{who}"] = float(np.median(level))
        out[f"peak_{who}"] = float(np.median(peak))
        out[f"swing_steps_{who}"] = float(np.median(length))
    return out


def condition(tag: str, skill: str, command: str) -> dict:
    data = np.load(ROLLOUTS / f"zs_{tag}_{skill}_auto_kin_{command}_encoder.npz")
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    fidelity = capture_fidelity(data, float(data["control_dt"]) / units.time)
    row = swings(data)
    row.update(
        clearance_body=float(np.mean(fidelity.clearance_body)),
        clearance_asked=float(np.mean(fidelity.clearance_asked)),
        duty_body=float(np.mean(fidelity.duty_body)),
        duty_asked=float(np.mean(fidelity.duty_asked)),
        liftoff_lag_mean=float(np.mean(fidelity.liftoff_lag_mean)),
        touchdown_lag_mean=float(np.mean(fidelity.touchdown_lag_mean)),
    )
    row["air_time_ratio"] = row["swing_steps_body"] / row["swing_steps_asked"]
    return row


def mean(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {}
    for name, tag in TAGS.items():
        per_skill = {s: [condition(tag, s, c) for c in COMMANDS] for s in SKILLS}
        report[name] = {
            "tag": tag,
            "both": mean([r for rows in per_skill.values() for r in rows]),
            **{s: mean(rows) for s, rows in per_skill.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    keys = ("stance_level_body", "peak_body", "clearance_body", "clearance_asked", "duty_body")
    for name, entry in report.items():
        print(f"{name:12s} " + "  ".join(f"{k} {entry['both'][k]:.4f}" for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
