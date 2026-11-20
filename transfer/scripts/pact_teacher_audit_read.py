"""Read the teacher audit (``pact_teacher_audit.sh``): every body's tracker, as trained and
mirror-symmetrised, on every task PACT learns from or is judged on.

Per body and variant (``plain``, ``sym``):

- probe map: probes passed (grid / manoeuvre / twin) and falls
  (``evaluate_competence.py``; plain = the co-executions' map stage);
- training task: terminations over the capture (falls and plans left) with a bootstrap SE
  over environments, forward speed (V) and median contract error (plain = DAgger round 0);
- G1 and NAO only (W20's ladder, autonomous, kinematic estimator): the base tasks (flat's
  three commands, jog's middle one) and the held-out sidestep under both generators:
  largest signature deviation and falls.

Usage::

    python transfer/scripts/pact_teacher_audit_read.py
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

from humanoid_transfer.common.skill_state import SKILL_LAYOUT  # noqa: E402
from pact_held_out_read import SEED, base  # noqa: E402
from w18_compare import COMMANDS, sidestep_reading, teacher_directed  # noqa: E402
from w19_compare import HELD_OUT  # noqa: E402

BODIES = ("g1", "nao", "h1", "cassie")
GROUPS = ("grid", "manoeuvre", "twin")


def probes(path: Path) -> dict | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text("utf-8"))
    rows = [p for p in report["probes"] if p.get("representable", True)]
    return {
        **{g: int(report["summary"][g]["passed"]) for g in GROUPS},
        "falls": int(sum(p["falls"] for p in rows)),
    }


def training_task(path: Path, draws: int, rng: np.random.Generator) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path)
    per_env = data["terminated"].sum(0).astype(float)
    boots = per_env[rng.integers(0, len(per_env), (draws, len(per_env)))].sum(1)
    speed = SKILL_LAYOUT.unpack(data["skill_state"])["com_velocity"][..., 0].mean()
    return {
        "terminations": float(per_env.sum()),
        "terminations_se": float(boots.std(ddof=1)),
        "forward_speed_v": float(speed),
        "median_contract_error": float(np.median(data["contract_error"])),
    }


def read(draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    teacher = {c: teacher_directed(c) for c in COMMANDS}
    report: dict = {"bootstrap": draws, "bodies": {}}
    for body in BODIES:
        entry = {}
        for v in ("plain", "sym"):
            row = {
                "probes": probes(
                    REPO / "transfer/results" / f"icml_competence_{body}.json"
                    if v == "plain"
                    else REPO / "transfer/results" / f"icml_teacher_{body}_sym_probes.json"
                ),
                "training_task": training_task(
                    REPO / "logs/pact/dagger/train/r0" / f"{body}.npz"
                    if v == "plain"
                    else REPO / "logs/pact/teachers" / f"{body}_sym_train.npz",
                    draws,
                    rng,
                ),
            }
            if body in SEED:
                tag = f"xs_{body}_s{SEED[body]}_{v}"
                row["base_tasks"] = base(body, tag)
                row["sidestep"] = {g: sidestep_reading(body, tag, g, teacher) for g in HELD_OUT}
            entry[v] = row
        report["bodies"][body] = entry
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=REPO / "transfer/results/icml_teacher_audit.json"
    )
    args = parser.parse_args()
    report = read(args.bootstrap, args.seed)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for body, entry in report["bodies"].items():
        for v, row in entry.items():
            p, t = row["probes"], row["training_task"]
            cells = [
                f"probes {p['grid']}/{p['manoeuvre']}/{p['twin']} falls {p['falls']}"
                if p
                else "probes -",
                f"train term {t['terminations']:.0f}±{t['terminations_se']:.0f} "
                f"err {t['median_contract_error']:.2f} v {t['forward_speed_v']:.3f}"
                if t
                else "train -",
            ]
            if "base_tasks" in row:
                b = row["base_tasks"]
                cells.append(
                    f"flat {b['flat_largest']:.3f}/{b['flat_falls']} jog {b['jog_largest']:.3f}/"
                    f"{b['jog_falls']}"
                    if b
                    else "base -"
                )
                cells.append(
                    "sidestep "
                    + " ".join(
                        f"{r['largest']:.3f}/{r['falls']}" if r else "-"
                        for r in row["sidestep"].values()
                    )
                )
            print(f"{body:6s} {v:5s} " + " | ".join(cells))
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
