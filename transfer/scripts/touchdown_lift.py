"""Does a swing that lifts less land earlier? Per swing, its lift against its touchdown lag.

``common/contact.py`` scores a capture's touchdowns against its plan on average. This
script pairs them swing by swing, on the deployed path (``encoder`` run) of each L4 result
given:

* every body swing that starts and ends in a valid stance is matched to the plan's
  touchdown nearest the body's, within half a pendulum time (``contact_fidelity``'s
  window);
* its **lag** is the body's touchdown minus the plan's, in ``T`` (negative: landed early);
* its **peak** is the sole's highest point above its own stance level, in ``L``
  (``contact_fidelity``'s clearance);
* its **early height** is the sole's height at the swing's second airborne sample, for
  swings of at least three samples: a lift the touchdown cannot have cut short.

Reported per capture and per group (body, skill, interface): Spearman's correlation of
each lift measure with the lag, and the mean lag in each quartile of it. It is a
correlation, not an intervention: it says whether early touchdowns come with low swings,
not which causes which.

The record (W1's three NAO trackers under both interfaces, S's three G1 seeds)::

    python transfer/scripts/touchdown_lift.py \
        --glob "zero_shot_nao_r1_3498_sym_v3_flat_dagger2_auto_kin_l4_*.json" \
        --glob "zero_shot_nao_r1_3498_sym_v3_rough_b4_auto_kin_l4_*.json" \
        --glob "zero_shot_nao_r1_3498_sym_v3_gen_*_l4_*.json" \
        --glob "zero_shot_nao_warm_nao_r1_s3[23]_sym_*_l4_*.json" \
        --glob "zero_shot_g1_warm_r3_logload_sym_*_l4_*.json" \
        --glob "zero_shot_g1_warm_r3_s4[23]_log_sym_*_l4_*.json" \
        --output transfer/results/touchdown_lift.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import _runs, foot_height  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)

WINDOW = 0.5
"""How far, in ``T``, a planned touchdown may be matched (``contact_fidelity``'s default)."""


def swing_touchdowns(
    body: np.ndarray, asked: np.ndarray, valid: np.ndarray, dt_star: float
) -> np.ndarray:
    """``(n, 4)`` per matched body swing: peak, early height (nan if short), lag ``T``, length.

    ``body`` and ``asked`` are ``(T, N, 27)``, aligned (``asked[t]`` is the plan for
    ``body[t]``); ``valid`` is ``(T, N)``.
    """
    contact = SKILL_LAYOUT.unpack(body)["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    planned = SKILL_LAYOUT.unpack(asked)["foot_load"] > DEFAULT_CONTACT_THRESHOLD
    height = foot_height(body, SKILL_LAYOUT)
    window = max(1, round(WINDOW / dt_star))
    rows = []
    for env in range(body.shape[1]):
        v = valid[:, env]
        for side in range(2):
            c, a, h = contact[:, env, side], planned[:, env, side], height[:, env, side]
            stance = c & v
            if not stance.any():
                continue
            base = float(np.median(h[stance]))
            both = v[1:] & v[:-1]
            touchdowns = np.flatnonzero(~a[:-1] & a[1:] & both) + 1
            for start, stop in _runs(~c & v):
                if start == 0 or stop >= len(c):
                    continue
                if not (v[start - 1] and v[stop] and c[start - 1] and c[stop]):
                    continue
                if not v[start - 1 : stop + 1].all():
                    continue
                near = touchdowns[np.abs(touchdowns - stop) <= window]
                if not near.size:
                    continue
                lag = (stop - near[np.argmin(np.abs(near - stop))]) * dt_star
                peak = float(h[start:stop].max() - base)
                early = float(h[start + 1] - base) if stop - start >= 3 else np.nan
                rows.append((peak, early, lag, (stop - start) * dt_star))
    return np.array(rows, dtype=float).reshape(-1, 4)


def capture_swings(path: Path) -> np.ndarray:
    """:func:`swing_touchdowns` of a capture, its plan aligned as ``capture_fidelity`` does."""
    data = np.load(path)
    units = UnitSystem(float(data["leg_length"]), float(data["total_mass"]))
    dt_star = float(data["control_dt"]) / units.time
    z = data["skill_state"].astype(float)
    done = data["done"].astype(bool)
    due = data["contract_due"][:-1].astype(float)
    body = z[1:]
    cut = done[:-1].copy()
    cut[:-1] |= done[1:-1]
    first = np.zeros_like(cut)
    first[0] = True
    first[1:] = cut[:-1]
    return swing_touchdowns(body, due, ~first, dt_star)


def summarize(rows: np.ndarray) -> dict:
    """Spearman's correlation of each lift measure with the lag, and the lag by quartile."""
    out = {"swings": int(len(rows)), "lag_mean": float(rows[:, 2].mean())}
    for name, column in (("peak", 0), ("early_height", 1)):
        keep = ~np.isnan(rows[:, column])
        x, lag = rows[keep, column], rows[keep, 2]
        rho = float(np.corrcoef(rankdata(x), rankdata(lag))[0, 1])
        bins = np.digitize(x, np.quantile(x, [0.25, 0.5, 0.75]))
        out[name] = {
            "swings": int(keep.sum()),
            "spearman_with_lag": rho,
            "lag_by_quartile": [
                float(lag[bins == k].mean()) if (bins == k).any() else float("nan")
                for k in range(4)
            ],
            "quartile_edges": [float(q) for q in np.quantile(x, [0.25, 0.5, 0.75])],
        }
    return out


def group_of(result: Path) -> str:
    """``<body> <skill> <interface>`` from an L4 result's name."""
    name = result.name
    body = "nao" if "_nao_" in name else "g1"
    skill = "rough" if "_rough_" in name else "flat"
    interface = "generator" if re.search(r"_gen_", name) else "linear"
    return f"{body} {skill} {interface}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--result", action="append", type=Path, default=[])
    parser.add_argument("--glob", action="append", default=[], help="pattern in transfer/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = list(args.result)
    for pattern in args.glob:
        results += sorted((REPO_ROOT / "transfer" / "results").glob(pattern))
    captures, groups = {}, defaultdict(list)
    for result in sorted(set(results)):
        run = json.loads(result.read_text(encoding="utf-8"))["runs"].get("encoder")
        if run is None:
            continue
        rows = capture_swings(REPO_ROOT / run["file"])
        rel = result.relative_to(REPO_ROOT).as_posix() if result.is_absolute() else str(result)
        captures[rel] = {"capture": run["file"], **summarize(rows)}
        groups[group_of(result)].append(rows)
        s = captures[rel]
        print(
            f"{result.name:72} n {s['swings']:5}  rho peak {s['peak']['spearman_with_lag']:+.3f}"
            f"  early {s['early_height']['spearman_with_lag']:+.3f}"
        )
    pooled = {name: summarize(np.concatenate(rows)) for name, rows in sorted(groups.items())}
    for name, s in pooled.items():
        print(
            f"{name:22} swings {s['swings']:6}  lag {s['lag_mean']:+.3f} T  "
            f"peak rho {s['peak']['spearman_with_lag']:+.3f}, lag by quartile "
            + " ".join(f"{x:+.3f}" for x in s["peak"]["lag_by_quartile"])
            + f"  early rho {s['early_height']['spearman_with_lag']:+.3f}, by quartile "
            + " ".join(f"{x:+.3f}" for x in s["early_height"]["lag_by_quartile"])
        )
    report = {"window_T": WINDOW, "groups": pooled, "captures": captures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
