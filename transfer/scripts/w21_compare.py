"""W21's pre-registered reading (``skill_space.md`` §10.22): the multi-body generator with a
learned latent, from free to collapsed through a bottleneck, on the best decoders so far.

Generators, all trained without sidestep and rough, on W18's recipe and bodies' data:

- ``c``: ``mt18_no_sidestep``, the experts only (the design's control);
- ``m``: ``mt18_no_sidestep_m``, with the bodies' histories; its history summary is free (beta 0);
- ``b1``, ``b2``: as ``m``, with the summary through a Gaussian bottleneck, beta 0.001 and 0.01;
- ``z``: as ``m``, beta 1.0, where the summary collapses: ``z*`` and the command alone.

Decoders, per body (``--decoders``): W20's ``xs`` if all three of its seeds are read, it regresses
neither on rough nor on the base tasks, and it falls less on sidestep under ``mt18_no_sidestep``
than W19's ``xf`` on the same seeds; otherwise W19's ``xf``.

Readings, per body, paired over the three decoders:

0. **The correction** (W17's reading 4 under the corrected leakage gate): ``mt17_l - mt17_m`` on
   rough's largest deviation, on W13's trackers;
1. **leakage** beyond the ``z*`` history (``w21_leakage.py``): only arms that pass are deployed;
2. **the learned latent:** ``m - z`` (free summary against ``z*`` alone);
3. **the bottleneck:** ``b1 - m`` and ``b2 - m``;
4. **the bodies' data:** ``m - c``;

each on sidestep (falls under the arm, summed and paired; largest deviation, paired) and on
rough (largest deviation, paired; falls). "Help" below -2 SE, "hurt" above +2 SE. Beside them,
5. **open objective** on sidestep per arm (W18's reading 3: falls at most ``mt17_c``'s on W13's
   trackers plus 2, every directed ratio at least 0.9).

    python transfer/scripts/w21_compare.py --output transfer/results/w21_latent.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "source" / "humanoid_transfer"))

from w8_compare import RESULTS  # noqa: E402
from w16_compare import FALLS_MORE, SPEED_MIN, paired, sign  # noqa: E402
from w16_compare import reading as rough_reading  # noqa: E402
from w18_compare import COMMANDS, sidestep_reading, teacher_directed  # noqa: E402
from w19_compare import MAIN, w18_bar  # noqa: E402

SEEDS = {"g1": (41, 42, 43), "nao": (31, 32, 33)}
GENERATORS = {
    "c": "mt18_no_sidestep",
    "m": "mt18_no_sidestep_m",
    "b1": "mt21_b1",
    "b2": "mt21_b2",
    "z": "mt21_z",
}
CONTRASTS = {
    "learned_latent": ("m", "z"),
    "bottleneck_b1": ("b1", "m"),
    "bottleneck_b2": ("b2", "m"),
    "bodies_data": ("m", "c"),
}


def tag(arm: str, body: str, seed: int) -> str:
    return f"{arm}_{body}_s{seed}_sym"


def decoder_arm(body: str, teacher: dict) -> str:
    """W20's ``xs`` where it is complete, regresses nowhere and falls less than W19's ``xf``."""
    path = RESULTS / "w20_from_s.json"
    if not path.exists():
        return "xf"
    entry = json.loads(path.read_text(encoding="utf-8"))["bodies"][body]
    if len(entry.get("seeds", [])) < len(SEEDS[body]):
        return "xf"
    rough = entry.get("rough_regression") or {"regresses": True}
    base = entry.get("base_tasks")
    if rough["regresses"] or not base:
        return "xf"
    if base["flat"]["regresses"] or base["jog"]["regresses"] or not base["walk_passes"]:
        return "xf"
    falls = {}
    for arm in ("xs", "xf"):
        rows = [sidestep_reading(body, tag(arm, body, s), MAIN, teacher) for s in SEEDS[body]]
        if any(r is None for r in rows):
            return "xf"
        falls[arm] = sum(r["falls"] for r in rows)
    return "xs" if falls["xs"] < falls["xf"] else "xf"


def correction(body: str) -> dict | None:
    """W17's reading 4, ``mt17_l - mt17_m`` on rough, on W13's trackers."""
    rows = {
        s: {a: rough_reading(body, tag("mt", body, s), a) for a in ("mt17_l", "mt17_m")}
        for s in SEEDS[body]
    }
    if any(r is None for row in rows.values() for r in row.values()):
        return None
    values = [rows[s]["mt17_l"]["largest"] - rows[s]["mt17_m"]["largest"] for s in rows]
    stat = paired(values)
    return {
        "largest": {**stat, "verdict": sign(stat)},
        "falls": {a: sum(rows[s][a]["falls"] for s in rows) for a in ("mt17_l", "mt17_m")},
        "speed_ratio_min": {
            a: min(rows[s][a]["speed_ratio_min"] for s in rows) for a in ("mt17_l", "mt17_m")
        },
    }


def leakage_passes() -> dict[str, bool]:
    path = RESULTS / "w21_leakage.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))["generators"]
    return {arm: report[name]["passes"] for arm, name in GENERATORS.items() if name in report}


def contrast(rows: dict, a: str, b: str, key: str) -> dict | None:
    if any(rows[s][a] is None or rows[s][b] is None for s in rows):
        return None
    stat = paired([rows[s][a][key] - rows[s][b][key] for s in rows])
    return {**stat, "verdict": sign(stat)}


def body_reading(body: str, decoder: str, teacher: dict) -> dict:
    seeds = SEEDS[body]
    tags = {s: tag(decoder, body, s) for s in seeds}
    side = {s: {a: sidestep_reading(body, tags[s], g, teacher) for a, g in GENERATORS.items()}
            for s in seeds}  # fmt: skip
    rough = {s: {a: rough_reading(body, tags[s], g) for a, g in GENERATORS.items()} for s in seeds}
    bar = w18_bar(body, seeds)
    objective = {}
    for arm in GENERATORS:
        if any(side[s][arm] is None for s in seeds) or bar is None:
            continue
        falls = sum(side[s][arm]["falls"] for s in seeds)
        speed = min(side[s][arm]["directed_ratio_min"] for s in seeds)
        objective[arm] = {
            "falls": falls,
            "directed_ratio_min": speed,
            "passes": bool(falls <= bar + FALLS_MORE and speed >= SPEED_MIN),
        }
    contrasts = {
        name: {
            "sidestep_falls": contrast(side, a, b, "falls"),
            "sidestep_largest": contrast(side, a, b, "largest"),
            "rough_largest": contrast(rough, a, b, "largest"),
            "rough_falls": None
            if any(rough[s][a] is None or rough[s][b] is None for s in seeds)
            else {x: sum(rough[s][x]["falls"] for s in seeds) for x in (a, b)},
        }
        for name, (a, b) in CONTRASTS.items()
    }
    return {
        "decoder": decoder,
        "bar_mt17_c_falls": bar,
        "open_objective": objective,
        "contrasts": contrasts,
        "correction_mt17_l": correction(body),
        "sidestep": side,
        "rough": rough,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decoders", action="store_true", help="print each body's decoder arm")
    args = parser.parse_args()
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    if args.decoders:
        for body in SEEDS:
            print(body, decoder_arm(body, teacher))
        return 0
    report = {
        "generators": GENERATORS,
        "leakage_passes": leakage_passes(),
        "bodies": {b: body_reading(b, decoder_arm(b, teacher), teacher) for b in SEEDS},
    }
    summary = []
    for body, entry in report["bodies"].items():
        parts = [f"{body} ({entry['decoder']})"]
        for name, sides in entry["contrasts"].items():
            verdicts = [
                f"{k.split('_')[0]} {v['verdict']}"
                for k, v in sides.items()
                if isinstance(v, dict) and "verdict" in v and k.endswith("largest")
            ]
            falls = sides["sidestep_falls"]
            if falls:
                verdicts.append(f"falls {falls['verdict']}")
            if verdicts:
                parts.append(f"{name}: " + ", ".join(verdicts))
        summary.append("; ".join(parts))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2, default=str) + "\n").encode("utf-8"))
    print(" | ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
