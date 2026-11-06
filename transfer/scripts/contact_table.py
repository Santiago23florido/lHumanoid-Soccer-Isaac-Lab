"""Group ``contact_fidelity.py``'s report by body, method and skill.

Each run's metrics are averaged over the three L4 commands; a method trained with
several seeds is then given as the mean over its seeds plus or minus their standard
deviation over the root of their number (methods compare by seeds,
``transfer/paper/README.md``, table SE). The deployed path is read for the trackers
(run ``encoder``), the policy itself for the baselines (``imitate``, ``oracle``).

    python transfer/scripts/contact_table.py transfer/results/contact_fidelity_l4.json \
        --output transfer/results/contact_fidelity_table.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# (body, method, skill, seed) from a result file's name.
METHODS = [
    (
        r"zero_shot_g1_(warm_r3_logload_sym|warm_r3_s4[23]_log_sym)_(rough|flat)_",
        "g1",
        "zero-shot (S seeds)",
    ),
    (r"zero_shot_g1_g1_task_(rough|flat)(_s4[23])?_sym_(rough|flat)_", "g1", "B1, task tracker"),
    (r"zero_shot_nao_(r1_3498_sym_v3)_(rough|flat)_", "nao", "zero-shot (adopted)"),
    (r"zero_shot_nao_(warm_nao_r1_s3[123]_sym)_(rough|flat)_", "nao", "zero-shot (S seeds)"),
    (r"zero_shot_nao_nao_task_(rough|flat)(_s3[23])?_sym_(rough|flat)_", "nao", "B1, task tracker"),
    (r"retarget_nao_(rough|flat)_(copy|offset|range)(_s4[34])?_l4", "nao", "B3, retarget {1}"),
    (r"oracle_nao_(rough|flat)(_bounded)?(_s43)?_l4", "nao", "B2{1}, oracle"),
]
RUN = {"zero_shot": "encoder", "retarget": "imitate", "oracle": "oracle"}
METRICS = [
    "duty_deviation",
    "duty_body",
    "light_share_body",
    "airborne_when_light",
    "graze_in_swing",
    "liftoff_lag_mean",
    "touchdown_lag_mean",
    "clearance_body",
]


def classify(name: str) -> tuple[str, str, str, str] | None:
    for pattern, body, label in METHODS:
        m = re.search(pattern, name)
        if not m:
            continue
        groups = [g or "" for g in m.groups()]
        skill = next(g for g in groups if g in ("rough", "flat"))
        if label.startswith("B3"):
            return body, label.format(None, groups[1]), skill, groups[2] or "s42"
        if label.startswith("B2"):
            return body, label.format(None, "b" if groups[1] else ""), skill, groups[2] or "s42"
        seed = next((g for g in groups if re.fullmatch(r"(_?s\d+_?.*|.*_s\d+.*)", g)), "")
        found = re.search(r"s(\d+)", seed)
        return body, label, skill, f"s{found.group(1)}" if found else "first"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = json.loads(args.report.read_text(encoding="utf-8"))["reports"]

    per_seed: dict = defaultdict(lambda: defaultdict(list))
    teacher: dict = defaultdict(lambda: defaultdict(list))
    for report in reports:
        name = Path(report["result"]).name
        key = classify(name)
        if key is None:
            continue
        run_name = RUN[next(k for k in RUN if name.startswith(k))]
        run = report["runs"].get(run_name)
        if run is None or run.get("missing"):
            continue
        body, method, skill, seed = key
        for metric in METRICS:
            per_seed[(body, method, skill, seed)][metric].append(run.get(metric))
        for metric in ("duty_body", "light_share_body", "clearance_body"):
            teacher[skill][metric].append(report["teacher"][metric])

    table: dict = defaultdict(dict)
    for (body, method, skill, seed), metrics in per_seed.items():
        means = {}
        for metric, values in metrics.items():
            values = [v for v in values if v is not None]
            if values:
                means[metric] = np.nanmean(np.asarray(values, dtype=float), axis=0).tolist()
        table[(body, method, skill)][seed] = means

    rows = []
    for (body, method, skill), seeds in sorted(table.items()):
        row = {"body": body, "method": method, "skill": skill, "seeds": sorted(seeds)}
        for metric in METRICS:
            values = np.asarray([s[metric] for s in seeds.values() if metric in s], dtype=float)
            if values.size == 0:
                continue
            row[metric] = values.mean(axis=0).tolist()
            row[f"{metric}_se"] = (
                (values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist()
                if len(values) > 1
                else None
            )
        rows.append(row)
    teachers = {
        skill: {m: np.mean(np.asarray(v, dtype=float), axis=0).tolist() for m, v in metrics.items()}
        for skill, metrics in teacher.items()
    }
    args.output.write_text(
        json.dumps({"source": args.report.as_posix(), "teacher": teachers, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    def pair(row: dict, metric: str) -> str:
        v = row.get(metric)
        if v is None:
            return "-"
        return "/".join(f"{x:.3f}" for x in v)

    header = [
        "Body", "Skill", "Method", "Seeds", "Duty dev. L/R", "Light share",
        "Airborne when light", "Graze", "Lift-off lag (T)", "Touchdown lag (T)", "Clearance (L)",
    ]  # fmt: skip
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for skill, t in sorted(teachers.items()):
        light = "/".join(f"{x:.3f}" for x in t["light_share_body"])
        clear = "/".join(f"{x:.3f}" for x in t["clearance_body"])
        print(f"| teacher | {skill} | - | - | - | {light} | - | - | - | - | {clear} |")
    columns = [
        "duty_deviation", "light_share_body", "airborne_when_light", "graze_in_swing",
        "liftoff_lag_mean", "touchdown_lag_mean", "clearance_body",
    ]  # fmt: skip
    for row in rows:
        cells = [row["body"], row["skill"], row["method"], str(len(row["seeds"]))]
        cells += [pair(row, c) for c in columns]
        print("| " + " | ".join(cells) + " |")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
