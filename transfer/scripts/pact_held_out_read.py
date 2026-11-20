"""Read the held-out task with PACT driving (``pact_held_out.sh``) against the tracker it imitated.

Same deployment, state estimate and generators for both drivers; the readings are W20's
(``w8_compare.condition`` for the base tasks, ``w18_compare.sidestep_reading`` for sidestep):
per command the largest signature deviation from the teacher's gait and the falls, and on
sidestep the directed-speed ratio.

Registered criterion (prereg, "Held-out task with PACT driving"): summed over sidestep's
three commands and both generators, PACT's falls at most the tracker's + 2 SE (bootstrap
over environments, the two runs independent) and PACT's mean largest deviation within 10 %
of the tracker's. The base tasks are reported beside it (seen in training).

Usage::

    python transfer/scripts/pact_held_out_read.py --pact-tag pact_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "source" / "humanoid_transfer"))

from w8_compare import COMMANDS as FLAT_COMMANDS  # noqa: E402
from w8_compare import condition  # noqa: E402
from w18_compare import COMMANDS, sidestep_reading, teacher_directed  # noqa: E402
from w19_compare import HELD_OUT  # noqa: E402

SEED = {"g1": 41, "nao": 31}
UNITS = {"g1": "", "nao": "_own"}  # the base tasks' units, as W20


def tracker_tag(body: str, variant: str = "plain") -> str:
    return f"xs_{body}_s{SEED[body]}_{variant}"


def falls_per_env(body: str, tag: str, generator: str) -> np.ndarray | None:
    counts = []
    for command in COMMANDS:
        path = REPO / body / "rollouts" / f"zs_{tag}_{generator}_own_auto_kin_{command}_encoder.npz"
        if not path.exists():
            return None
        counts.append(np.load(path)["terminated"].sum(0))
    return np.concatenate(counts).astype(float)


def boot_se(values: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    return float(values[idx].sum(1).std(ddof=1))


def base(body: str, tag: str) -> dict | None:
    units = UNITS[body]
    flat = [condition(body, tag, f"flat_dagger2{units}", c) for c in FLAT_COMMANDS]
    jog = condition(body, tag, f"base_jog_b4{units}", "mid")
    if any(r is None for r in flat) or jog is None:
        return None
    return {
        "flat_largest": float(np.mean([r["largest"] for r in flat])),
        "flat_falls": int(sum(r["falls"] for r in flat)),
        "jog_largest": float(jog["largest"]),
        "jog_falls": int(jog["falls"]),
    }


def read(pact_tag: str, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    report: dict = {"pact_tag": pact_tag, "bootstrap": draws, "bodies": {}}
    for body in SEED:
        drivers = {
            "tracker": tracker_tag(body),
            "pact": f"{pact_tag}_{body}",
            "tracker_sym_w20": tracker_tag(body, "sym"),
        }
        entry: dict = {"tags": drivers, "base": {}, "sidestep": {}}
        for name, tag in drivers.items():
            entry["base"][name] = base(body, tag)
            entry["sidestep"][name] = {g: sidestep_reading(body, tag, g, teacher) for g in HELD_OUT}
        rows = entry["sidestep"]
        if all(rows[d][g] is not None for d in ("tracker", "pact") for g in HELD_OUT):
            falls = {
                d: np.concatenate([falls_per_env(body, drivers[d], g) for g in HELD_OUT])
                for d in ("tracker", "pact")
            }
            se = float(np.hypot(*(boot_se(falls[d], draws, rng) for d in ("tracker", "pact"))))
            largest = {
                d: float(np.mean([rows[d][g]["largest"] for g in HELD_OUT]))
                for d in ("tracker", "pact")
            }
            diff = float(falls["pact"].sum() - falls["tracker"].sum())
            entry["criterion"] = {
                "falls": {d: float(falls[d].sum()) for d in falls},
                "falls_diff": diff,
                "falls_se": se,
                "largest": largest,
                "largest_ratio": largest["pact"] / largest["tracker"],
                "passes": bool(diff <= 2 * se and largest["pact"] <= 1.1 * largest["tracker"]),
            }
        report["bodies"][body] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pact-tag", default="pact_final")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = read(args.pact_tag, args.bootstrap, args.seed)
    output = args.output or REPO / f"transfer/results/icml_held_out_{args.pact_tag}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for body, entry in report["bodies"].items():
        for name in ("tracker", "pact", "tracker_sym_w20"):
            b = entry["base"][name]
            seen = (
                f"flat {b['flat_largest']:.3f} ({b['flat_falls']} falls), "
                f"jog {b['jog_largest']:.3f} ({b['jog_falls']})"
                if b
                else "base: missing"
            )
            side = "  ".join(
                f"{g}: {r['largest']:.3f}, {r['falls']} falls, dir {r['directed_ratio_min']:.2f}"
                if r
                else f"{g}: missing"
                for g, r in entry["sidestep"][name].items()
            )
            print(f"{body:4s} {name:16s} {seen} | {side}")
        if "criterion" in entry:
            c = entry["criterion"]
            print(
                f"{body:4s} held out: falls PACT {c['falls']['pact']:.0f} vs tracker "
                f"{c['falls']['tracker']:.0f} ({c['falls_diff']:+.0f} ± {c['falls_se']:.1f}); "
                f"largest ratio {c['largest_ratio']:.2f}; passes {c['passes']}"
            )
    print(f"written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
