"""W16's pre-registered reading (``skill_space.md`` §10.17): rough, held out of everything, on
both bodies' frozen W13 decoders, with the multi-task generators of step 5.

Arms, each a generator (``logs/contract/<arm>``) deployed on the same six trackers (G1 s41-43,
NAO s31-33), rough's L4 at its three commands on the deployed path, in each tracker's own units:

- ``mt16_c``: the G1 experts' five base tasks only (the control);
- ``mt16_m``: ``mt16_c`` then the bodies' histories with the teacher's phase-matched futures;
- ``mt16_d``: ``mt16_c`` then the bodies' histories with their own futures;
- ``mt16_l``: as ``mt16_m``, its history summary trained to carry no body; evaluated only if its
  leakage probe does not beat its control by more than 3 SE (``--leakage-gate``);
- ``rough_b4``: a generator that saw rough, the reference for what zero-shot costs.

Readings, per body, paired by tracker (mean ± SE over three), on rough's largest gate C
deviation (the mean over the three commands):

1. **bodies** -- ``mt16_m - mt16_c``: "help" below -2 SE, "hurt" above +2 SE, else "none";
2. **theory** -- ``mt16_d - mt16_m`` above +2 SE: "demonstrations worse"; and the asked duty's
   distance from the teacher's, ``mt16_m`` against ``mt16_c``: "closer" below -2 SE;
3. **latent** -- ``mt16_l - mt16_m``, as 1;
4. **zero-shot cost** -- each arm minus ``rough_b4``, reported;
5. **no regression** -- per arm against ``mt16_c``: falls at most two more over the three
   trackers, and every command's speed ratio at least 0.9.

    python transfer/scripts/w16_compare.py --output transfer/results/w16_multibody.json
    python transfer/scripts/w16_compare.py --leakage-gate
    python transfer/scripts/w16_compare.py --asked

``--asked`` prints the context of §10.17: the duty factor the same generator asks of each body,
and what the body does, on S's trackers' published captures (the task's units).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from w8_compare import RESULTS, condition  # noqa: E402

TRACKERS = {
    "g1": ("mt_g1_s41_sym", "mt_g1_s42_sym", "mt_g1_s43_sym"),
    "nao": ("mt_nao_s31_sym", "mt_nao_s32_sym", "mt_nao_s33_sym"),
}
ARMS = ("mt16_c", "mt16_m", "mt16_d", "mt16_l", "rough_b4")
COMMANDS = ("forward", "fast", "turning")
FALLS_MORE = 2
SPEED_MIN = 0.9
LEAK_SE = 3.0


def duties(body: str, tag: str, arm: str) -> dict | None:
    asked, executed, teacher = [], [], []
    for command in COMMANDS:
        path = RESULTS / f"zero_shot_{body}_{tag}_{arm}_own_auto_kin_l4_{command}.json"
        if not path.exists():
            return None
        runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
        teacher.append(float(np.mean(runs["teacher"]["signature"]["duty_factor"])))
        asked.append(float(np.mean(runs["encoder"]["asked"]["signature"]["duty_factor"])))
        executed.append(float(np.mean(runs["encoder"]["signature"]["duty_factor"])))
    return {
        "teacher": float(np.mean(teacher)),
        "asked": float(np.mean(asked)),
        "body": float(np.mean(executed)),
    }


def reading(body: str, tag: str, arm: str) -> dict | None:
    rows = [condition(body, tag, f"{arm}_own", c) for c in COMMANDS]
    if any(r is None for r in rows):
        return None
    return {
        "largest": float(np.mean([r["largest"] for r in rows])),
        "falls": int(sum(r["falls"] for r in rows)),
        "speed_ratio_min": float(min(r["speed_ratio"] for r in rows)),
        "duty": duties(body, tag, arm),
    }


def paired(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "se": float(v.std(ddof=1) / np.sqrt(len(v))),
        "values": v.tolist(),
    }


def sign(stat: dict) -> str:
    if stat["mean"] < -2 * stat["se"]:
        return "help"
    if stat["mean"] > 2 * stat["se"]:
        return "hurt"
    return "none"


def difference(rows: dict, a: str, b: str, key=lambda r: r["largest"]) -> dict | None:
    if any(rows[t].get(a) is None or rows[t].get(b) is None for t in rows):
        return None
    return paired([key(rows[t][a]) - key(rows[t][b]) for t in rows])


def generator_report(arm: str) -> dict | None:
    path = RESULTS / f"contract_{arm}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    bodies = report.get("bodies") or {}
    return {
        "gate_passes": report["gate"]["passes"],
        "calibration_error": report["gate"]["calibration_error"],
        "chain_fidelity": report["chain_fidelity"]["score"],
        "leakage": bodies.get("leakage"),
    }


def leakage_gate() -> tuple[bool, str]:
    entry = generator_report("mt16_l")
    if entry is None or entry["leakage"] is None:
        return False, "mt16_l has no leakage measurement"
    leak = entry["leakage"]
    margin = leak["balanced_accuracy"] - leak["control"]
    passes = margin <= LEAK_SE * leak["se"]
    return passes, (
        f"{'passes' if passes else 'fails'}: balanced accuracy {leak['balanced_accuracy']:.3f}, "
        f"control {leak['control']:.3f}, 3 SE {LEAK_SE * leak['se']:.3f}"
    )


ASKED = {
    "g1": ("warm_r3_logload_sym", "warm_r3_s42_log_sym", "warm_r3_s43_log_sym"),
    "nao": ("r1_3498_sym_v3", "warm_nao_r1_s32_sym", "warm_nao_r1_s33_sym"),
}


def asked_context() -> None:
    for generator in ("flat_dagger2", "rough_b4"):
        for body, tags in ASKED.items():
            rows = []
            for tag in tags:
                for command in COMMANDS:
                    name = f"zero_shot_{body}_{tag}_{generator}_auto_kin_l4_{command}.json"
                    runs = json.loads((RESULTS / name).read_text(encoding="utf-8"))["runs"]
                    rows.append(
                        [
                            np.mean(runs["teacher"]["signature"]["duty_factor"]),
                            np.mean(runs["encoder"]["asked"]["signature"]["duty_factor"]),
                            np.mean(runs["encoder"]["signature"]["duty_factor"]),
                        ]
                    )
            teacher, asked, executed = np.mean(rows, axis=0)
            print(
                f"{generator:13s} {body:4s} duty: teacher {teacher:.3f}, asked {asked:.3f}, "
                f"body {executed:.3f} (S's three trackers, three commands)"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--leakage-gate", action="store_true")
    parser.add_argument("--asked", action="store_true")
    args = parser.parse_args()
    if args.asked:
        asked_context()
        return 0
    if args.leakage_gate:
        passes, line = leakage_gate()
        print(line)
        return 0 if passes else 1
    report: dict = {
        "generators": {arm: generator_report(arm) for arm in ARMS if arm != "rough_b4"},
        "leakage_gate": leakage_gate()[1],
        "bodies": {},
    }
    summary = []
    for body, tags in TRACKERS.items():
        rows = {t: {arm: reading(body, t, arm) for arm in ARMS} for t in tags}
        bodies = difference(rows, "mt16_m", "mt16_c")
        theory = difference(rows, "mt16_d", "mt16_m")

        def distance(r: dict) -> float:
            return abs(r["duty"]["asked"] - r["duty"]["teacher"])

        asked = difference(rows, "mt16_m", "mt16_c", distance)
        latent = difference(rows, "mt16_l", "mt16_m")
        cost = {arm: difference(rows, arm, "rough_b4") for arm in ARMS if arm != "rough_b4"}
        regression = {}
        for arm in ARMS:
            if any(rows[t][arm] is None or rows[t]["mt16_c"] is None for t in tags):
                continue
            falls = sum(rows[t][arm]["falls"] for t in tags)
            control = sum(rows[t]["mt16_c"]["falls"] for t in tags)
            speed = min(rows[t][arm]["speed_ratio_min"] for t in tags)
            regression[arm] = {
                "falls": falls,
                "falls_control": control,
                "speed_ratio_min": speed,
                "passes": bool(falls <= control + FALLS_MORE and speed >= SPEED_MIN),
            }
        entry = {
            "rows": rows,
            "bodies": None if bodies is None else {**bodies, "verdict": sign(bodies)},
            "theory_demonstrations": None
            if theory is None
            else {**theory, "verdict": "demonstrations worse" if sign(theory) == "hurt" else "not"},
            "theory_asked_duty": None
            if asked is None
            else {**asked, "verdict": "closer" if sign(asked) == "help" else "not"},
            "latent": None if latent is None else {**latent, "verdict": sign(latent)},
            "zero_shot_cost": cost,
            "no_regression": regression,
        }
        report["bodies"][body] = entry
        part = [f"{body}: bodies {entry['bodies']['verdict'] if entry['bodies'] else 'incomplete'}"]
        if entry["latent"]:
            part.append(f"latent {entry['latent']['verdict']}")
        if entry["theory_demonstrations"]:
            part.append(f"demonstrations {entry['theory_demonstrations']['verdict']}")
        summary.append(", ".join(part))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print("; ".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
