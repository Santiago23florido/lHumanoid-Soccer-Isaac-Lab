"""Read the held-out task on every body (``pact_held_out.sh``): a model against the body's teacher.

Same deployment, state estimate and generators for both drivers (the teacher's tag is
``teacher_<body>``, the model's ``<tag>_<body>``), every task in the tracker's own units. The
readings are W20's: ``w8_compare.condition`` for the base tasks, ``w18_compare.sidestep_reading``
for sidestep. Per command they give the largest signature deviation from the G1 expert's gait,
the falls and the directed-speed ratio.

Registered criterion, per body (prereg, "Held-out task with PACT driving"): summed over
sidestep's three commands and both generators, the model's falls are at most the teacher's
+ 2 SE, and its mean largest deviation is within 10 % of the teacher's. SE is a bootstrap over
environments, with the two runs independent. The base tasks are reported beside it.

Each tag's reading is written to ``transfer/results/icml_held_out_<tag>.json``. It includes
the sidestep falls of every environment, so later contrasts need no capture. ``--contrast A B``
compares two tags' sidestep falls summed over the bodies both have; it is the reading behind
"better out of distribution" (prereg R3).

    python transfer/scripts/icml_held_out_read.py --tag icml_pact_d_s0
    python transfer/scripts/icml_held_out_read.py --contrast icml_pact_d_s0 icml_pact_na_s0
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

from body_conditions import TEACHERS, registry, rollouts_dir  # noqa: E402
from w8_compare import COMMANDS as FLAT_COMMANDS  # noqa: E402
from w8_compare import condition  # noqa: E402
from w18_compare import COMMANDS, sidestep_reading, teacher_directed  # noqa: E402
from w19_compare import HELD_OUT  # noqa: E402

RESULTS = REPO / "transfer/results"


def teacher_tag(body: str, stem: str = "teacher") -> str:
    return f"{stem}_{body}"


def falls_per_env(body: str, tag: str, generator: str) -> list[int] | None:
    counts = []
    for command in COMMANDS:
        path = rollouts_dir(body) / f"zs_{tag}_{generator}_own_auto_kin_{command}_encoder.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            counts.append(data["terminated"].sum(0))
    return [int(x) for x in np.concatenate(counts)]


def base(body: str, tag: str) -> dict | None:
    flat = [condition(body, tag, "flat_dagger2_own", c) for c in FLAT_COMMANDS]
    jog = condition(body, tag, "base_jog_b4_own", "mid")
    if any(r is None for r in flat) or jog is None:
        return None
    return {
        "flat_largest": float(np.mean([r["largest"] for r in flat])),
        "flat_falls": int(sum(r["falls"] for r in flat)),
        "jog_largest": float(jog["largest"]),
        "jog_falls": int(jog["falls"]),
    }


def driver(body: str, tag: str, teacher: dict) -> dict:
    side = {g: sidestep_reading(body, tag, g, teacher) for g in HELD_OUT}
    falls = {g: falls_per_env(body, tag, g) for g in HELD_OUT}
    complete = all(side[g] is not None and falls[g] is not None for g in HELD_OUT)
    return {
        "tag": tag,
        "base": base(body, tag),
        "sidestep": side,
        "sidestep_falls_per_env": falls if complete else None,
        "sidestep_largest": float(np.mean([side[g]["largest"] for g in HELD_OUT])) if complete else None,
    }


def boot_se(values: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    return float(values[idx].sum(1).std(ddof=1))


def all_falls(entry: dict) -> np.ndarray:
    return np.concatenate([entry["sidestep_falls_per_env"][g] for g in HELD_OUT]).astype(float)


def criterion(model: dict, teacher: dict, draws: int, rng: np.random.Generator) -> dict | None:
    if model["sidestep_falls_per_env"] is None or teacher["sidestep_falls_per_env"] is None:
        return None
    m, t = all_falls(model), all_falls(teacher)
    se = float(np.hypot(boot_se(m, draws, rng), boot_se(t, draws, rng)))
    diff = float(m.sum() - t.sum())
    ratio = model["sidestep_largest"] / teacher["sidestep_largest"]
    return {
        "falls": {"model": float(m.sum()), "teacher": float(t.sum())},
        "falls_diff": diff,
        "falls_se": se,
        "largest_ratio": ratio,
        "passes": bool(diff <= 2 * se and ratio <= 1.1),
    }


def read(tag: str, draws: int, seed: int, bodies: list[str], stem: str = "teacher") -> dict:
    rng = np.random.default_rng(seed)
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    report: dict = {"tag": tag, "bootstrap": draws, "bodies": {}}
    for body in bodies:
        t = driver(body, teacher_tag(body, stem), teacher)
        m = driver(body, f"{tag}_{body}", teacher)
        report["bodies"][body] = {"teacher": t, "model": m, "criterion": criterion(m, t, draws, rng)}
    rows = [e["criterion"] for e in report["bodies"].values()]
    report["passes_every_body"] = bool(rows) and all(r is not None and r["passes"] for r in rows)
    return report


def contrast(a: str, b: str, draws: int, seed: int) -> dict:
    """Tag ``a`` minus tag ``b`` in sidestep falls, summed over the bodies both have read."""
    rng = np.random.default_rng(seed)
    ra = json.loads((RESULTS / f"icml_held_out_{a}.json").read_text("utf-8"))["bodies"]
    rb = json.loads((RESULTS / f"icml_held_out_{b}.json").read_text("utf-8"))["bodies"]
    bodies = [
        x for x in ra
        if x in rb and ra[x]["model"]["sidestep_falls_per_env"] and rb[x]["model"]["sidestep_falls_per_env"]
    ]  # fmt: skip
    fa = [all_falls(ra[x]["model"]) for x in bodies]
    fb = [all_falls(rb[x]["model"]) for x in bodies]
    se = float(np.sqrt(sum(boot_se(v, draws, rng) ** 2 for v in fa + fb))) if bodies else float("nan")
    diff = float(sum(v.sum() for v in fa) - sum(v.sum() for v in fb))
    return {
        "a": a, "b": b, "bodies": bodies, "falls_a": float(sum(v.sum() for v in fa)),
        "falls_b": float(sum(v.sum() for v in fb)), "diff": diff, "se": se,
        "a_better_by_2se": bool(diff < -2 * se),
    }  # fmt: skip


H2_GENERATOR = "mt18_no_sidestep_s0"


def h2(tag: str, bodies: list[str], stem: str = "teacher") -> dict:
    """H2's closed loop (``pact_held_out.sh`` with ``PACT_HELD_OUT_H2=1``): sidestep under the
    s0 generator, plans as drawn (``<tag>``) and projected onto still loaded feet
    (``<tag>_stick``). Registered (prereg G-P, 2026-11-22 01:20 and 07:14): "without more falls"
    means falls projected − as drawn ≤ 2 Poisson SE (``sqrt`` of their sum), summed over the
    sidestep commands, per body and driver."""
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    report: dict = {"tag": tag, "generator": H2_GENERATOR, "bodies": {}}
    for body in bodies:
        entry = {}
        for name, base_tag in (("teacher", teacher_tag(body, stem)), ("model", f"{tag}_{body}")):
            row = {}
            for mode, t in (("drawn", base_tag), ("stick", f"{base_tag}_stick")):
                r = sidestep_reading(body, t, H2_GENERATOR, teacher)
                row[mode] = None if r is None else {"falls": r["falls"], "largest": r["largest"]}
            if row["drawn"] and row["stick"]:
                diff = row["stick"]["falls"] - row["drawn"]["falls"]
                se = float(np.sqrt(row["stick"]["falls"] + row["drawn"]["falls"]))
                row["falls_diff"], row["falls_se"] = diff, se
                row["no_more_falls"] = bool(diff <= 2 * se)
            entry[name] = row
        report["bodies"][body] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", default="")
    parser.add_argument("--contrast", nargs=2, metavar=("A", "B"), default=None)
    parser.add_argument("--h2", action="store_true", help="read H2's closed loop for --tag")
    parser.add_argument("--bodies", default=",".join(registry(TEACHERS)))
    parser.add_argument("--teacher-tag", default="teacher", help="the teacher's tag stem")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.contrast:
        c = contrast(*args.contrast, args.bootstrap, args.seed)
        output = args.output or RESULTS / f"icml_held_out_contrast_{c['a']}_vs_{c['b']}.json"
        output.write_text(json.dumps(c, indent=2) + "\n", encoding="utf-8")
        print(
            f"{c['a']} - {c['b']}: sidestep falls {c['falls_a']:.0f} vs {c['falls_b']:.0f} "
            f"({c['diff']:+.0f} ± {c['se']:.1f}) over {c['bodies']}; better by > 2 SE: {c['a_better_by_2se']}"
        )
        return 0
    if not args.tag:
        parser.error("--tag or --contrast")
    if args.h2:
        report = h2(args.tag, [b for b in args.bodies.split(",") if b], args.teacher_tag)
        output = args.output or RESULTS / f"icml_h2_{args.tag}.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for body, entry in report["bodies"].items():
            for name, row in entry.items():
                if "falls_diff" not in row:
                    print(f"{body:7s} {name:8s} incomplete")
                    continue
                print(
                    f"{body:7s} {name:8s} falls drawn {row['drawn']['falls']} stick {row['stick']['falls']} "
                    f"({row['falls_diff']:+d} ± {row['falls_se']:.1f}); largest "
                    f"{row['drawn']['largest']:.3f} -> {row['stick']['largest']:.3f}; "
                    f"no more falls {row['no_more_falls']}"
                )
        print(f"written {output}")
        return 0
    bodies = [b for b in args.bodies.split(",") if b]
    report = read(args.tag, args.bootstrap, args.seed, bodies, args.teacher_tag)
    output = args.output or RESULTS / f"icml_held_out_{args.tag}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for body, entry in report["bodies"].items():
        c = entry["criterion"]
        if c is None:
            print(f"{body:7s} incomplete")
            continue
        print(
            f"{body:7s} sidestep falls model {c['falls']['model']:.0f} vs teacher {c['falls']['teacher']:.0f} "
            f"({c['falls_diff']:+.0f} ± {c['falls_se']:.1f}); largest ratio {c['largest_ratio']:.2f}; "
            f"passes {c['passes']}"
        )
    print(f"every body passes: {report['passes_every_body']}; written {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
