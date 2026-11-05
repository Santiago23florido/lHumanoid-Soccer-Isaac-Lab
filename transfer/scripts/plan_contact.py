"""Do the plans' loaded feet rest on the ground, and does it matter where they do not?

``z*`` carries each foot's load and position as separate fields. Two readings, deployed
path:

* **where a loaded foot is** -- the sole's height (``common/contact.foot_height``) of every
  foot carrying more than the contact threshold, and of those in light contact (at most a
  quarter of the weight): in the teachers' captures, in the plans the trackers were given
  (``contract_due``), in the procedural biped's plans, and in the bodies;
* **the events** -- each planned lift-off and touchdown matched to the body's nearest event
  of the same kind on the same foot within half a pendulum time. Against the lag (body
  minus plan, ``T``) goes the plan's own sole height over its two loaded samples next to
  the event (the last two before a lift-off, the first two of a stance). Reported as
  Spearman's correlation, and as the mean lag in each quartile of that height.

    python transfer/scripts/plan_contact.py --output transfer/results/plan_contact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from humanoid_transfer.common.contact import LIGHT_LOAD, foot_height  # noqa: E402
from humanoid_transfer.common.similarity import UnitSystem  # noqa: E402
from humanoid_transfer.common.skill_state import (  # noqa: E402
    DEFAULT_CONTACT_THRESHOLD,
    SKILL_LAYOUT,
)

WINDOW = 0.5
TEACHERS = {
    "flat": [f"g1/rollouts/flat_play_{c}.npz" for c in ("forward", "fast", "turning")],
    "rough": [f"g1/rollouts/rough_play_{c}.npz" for c in ("forward", "fast", "turning")],
}
TRACKERS = {
    # S's three G1 seeds and S's NAO seeds, the deployed path, symmetrized.
    "g1": ("warm_r3_logload_sym", "warm_r3_s42_log_sym", "warm_r3_s43_log_sym"),
    "nao": ("r1_3498_sym_v3", "warm_nao_r1_s32_sym", "warm_nao_r1_s33_sym"),
}
SKILLS = {"flat": "flat_dagger2", "rough": "rough_b4"}
PROCEDURAL = "g1/rollouts/zs_warm_r3_logload_sym_procedural_starts.npz"
COMMANDS = ("forward", "fast", "turning")


def where_loaded(z: np.ndarray) -> dict:
    """Sole heights of loaded feet, and of feet in light contact, ``L``."""
    load = SKILL_LAYOUT.unpack(z)["foot_load"]
    height = foot_height(z)
    loaded = height[load > DEFAULT_CONTACT_THRESHOLD]
    light = height[(load > DEFAULT_CONTACT_THRESHOLD) & (load <= LIGHT_LOAD)]
    return {
        "loaded": {"median": float(np.median(loaded)), "p90": float(np.quantile(loaded, 0.9))},
        "light": {"median": float(np.median(light)), "p90": float(np.quantile(light, 0.9))},
    }


def aligned(path: Path):
    """Body, plan and validity of a capture, the plan aligned as ``capture_fidelity`` does."""
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
    return body, due, ~first, dt_star


def event_pairs(body, due, valid, dt_star) -> dict[str, np.ndarray]:
    """``{kind: (n, 2)}`` the plan's sole height next to each event, and the body's lag."""
    threshold = DEFAULT_CONTACT_THRESHOLD
    b = SKILL_LAYOUT.unpack(body)["foot_load"] > threshold
    a = SKILL_LAYOUT.unpack(due)["foot_load"] > threshold
    height = foot_height(due)
    window = max(1, round(WINDOW / dt_star))
    out: dict[str, list] = {"liftoff": [], "touchdown": []}
    for env in range(body.shape[1]):
        v = valid[:, env]
        both = v[1:] & v[:-1]
        for side in range(2):
            ae, be, h = a[:, env, side], b[:, env, side], height[:, env, side]
            events = {
                "liftoff": (
                    np.flatnonzero(ae[:-1] & ~ae[1:] & both) + 1,
                    np.flatnonzero(be[:-1] & ~be[1:] & both) + 1,
                ),
                "touchdown": (
                    np.flatnonzero(~ae[:-1] & ae[1:] & both) + 1,
                    np.flatnonzero(~be[:-1] & be[1:] & both) + 1,
                ),
            }
            for kind, (planned, bodily) in events.items():
                for t in planned:
                    if t < 3 or t + 3 >= len(ae) or not v[t - 3 : t + 3].all():
                        continue
                    near = bodily[np.abs(bodily - t) <= window]
                    if not near.size:
                        continue
                    lag = (near[np.argmin(np.abs(near - t))] - t) * dt_star
                    loaded = h[t - 2 : t] if kind == "liftoff" else h[t : t + 2]
                    out[kind].append((float(loaded.mean()), lag))
    return {k: np.array(v, dtype=float).reshape(-1, 2) for k, v in out.items()}


def summarize(pairs: np.ndarray) -> dict:
    height, lag = pairs[:, 0], pairs[:, 1]
    edges = np.quantile(height, [0.25, 0.5, 0.75])
    bins = np.digitize(height, edges)
    return {
        "events": int(len(pairs)),
        "height_median": float(np.median(height)),
        "height_quartile_edges": [float(e) for e in edges],
        "spearman_height_lag": float(spearmanr(height, lag).correlation),
        "lag_by_height_quartile": [float(lag[bins == k].mean()) for k in range(4)],
        "lag_mean": float(lag.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"teachers": {}, "trackers": {}}
    for skill, files in TEACHERS.items():
        z = np.concatenate(
            [np.load(REPO_ROOT / f)["skill_state"].astype(float) for f in files], axis=1
        )
        report["teachers"][skill] = where_loaded(z)
        t = report["teachers"][skill]
        print(
            f"teacher {skill:5}  loaded {t['loaded']['median']:+.4f} (p90 {t['loaded']['p90']:.4f})"
            f"  light {t['light']['median']:+.4f} (p90 {t['light']['p90']:.4f})"
        )
    # The procedural biped's plans, as the G1 incumbent was given them at L1.
    procedural = np.load(REPO_ROOT / PROCEDURAL)["contract_due"][:-1].astype(float)
    report["procedural"] = where_loaded(procedural)
    q = report["procedural"]
    print(f"procedural plans  loaded {q['loaded']['median']:+.4f} (p90 {q['loaded']['p90']:.4f})")
    for body_name, tags in TRACKERS.items():
        root = REPO_ROOT / body_name / "rollouts"
        for skill, generator in SKILLS.items():
            plans, bodies, pairs = [], [], {"liftoff": [], "touchdown": []}
            for tag in tags:
                for command in COMMANDS:
                    path = root / f"zs_{tag}_{generator}_auto_kin_{command}_encoder.npz"
                    body, due, valid, dt_star = aligned(path)
                    plans.append(due[valid])
                    bodies.append(body[valid])
                    for kind, v in event_pairs(body, due, valid, dt_star).items():
                        pairs[kind].append(v)
            entry = {
                "plans": where_loaded(np.concatenate(plans)[:, None]),
                "bodies": where_loaded(np.concatenate(bodies)[:, None]),
                **{kind: summarize(np.concatenate(v)) for kind, v in pairs.items()},
            }
            report["trackers"][f"{body_name} {skill}"] = entry
            p = entry["plans"]
            print(
                f"{body_name} {skill:5} plans: loaded {p['loaded']['median']:+.4f} "
                f"(p90 {p['loaded']['p90']:.4f}), light {p['light']['median']:+.4f} "
                f"(p90 {p['light']['p90']:.4f})"
            )
            for kind in ("liftoff", "touchdown"):
                s = entry[kind]
                print(
                    f"    {kind:9} n {s['events']:6}  plan height median {s['height_median']:.4f}"
                    f"  rho {s['spearman_height_lag']:+.3f}  lag by quartile "
                    + " ".join(f"{x:+.3f}" for x in s["lag_by_height_quartile"])
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
