"""The paper's tables, rebuilt from the result files into ``transfer/paper/README.md``.

Every cell is read from a JSON in ``transfer/results/`` (or another directory, with
``--results``); a row whose files do not exist yet is left out, so the tables grow as
the queue (``paper_queue.sh``) runs. The tables go between the README's
``<!-- tables:start -->`` and ``<!-- tables:end -->`` markers, and their data to
``transfer/paper/data/tables.json``.

Usage::

    python transfer/scripts/paper_tables.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "transfer" / "paper"
COMMANDS = ("forward", "fast", "turning")
GENERATOR = {"rough": "rough_b4", "flat": "flat_dagger2"}
MODEL = {"g1": "warm_r3_logload_sym", "nao": "r1_3498_sym_v3"}
"""The adopted model's tags (``incumbents.json``), deployed symmetrized."""
SEEDS = {
    "g1": [("41", MODEL["g1"]), ("42", "warm_r3_s42_log_sym"), ("43", "warm_r3_s43_log_sym")],
    "nao": [(s, f"warm_nao_r1_s{s}_sym") for s in ("31", "32", "33")],
}
ABLATIONS = [
    # (change, body, (tag, generator, kinematic) before, the same after)
    ("learned encoder -> kinematic estimator", "g1",
     ("r3_6194_sym", "rough_b3", False), ("r3_6194_sym", "rough_b3", True)),
    ("learned encoder -> kinematic estimator", "nao",
     ("r1_3498_sym", "rough_b3", False), ("r1_3498_sym", "rough_b3", True)),
    ("height at the sole point -> the sole's lowest corner", "g1",
     ("r3_6194_sym", "rough_b3", True), ("r3_6194_sym_v3", "rough_b3", True)),
    ("generator B3 -> B4 (noise-conditional)", "g1",
     ("r3_6194_sym_v3", "rough_b3", True), ("r3_6194_sym_v3", "rough_b4", True)),
    ("generator B3 -> B4 (noise-conditional)", "nao",
     ("r1_3498_sym_v3", "rough_b3", True), ("r1_3498_sym_v3", "rough_b4", True)),
    ("loads scored linearly -> in log space (same continuation, seed 41)", "g1",
     ("warm_r3", "rough_b4", True), ("warm_r3_logload", "rough_b4", True)),
    ("loads scored linearly -> in log space (same continuation, seed 31)", "nao",
     ("warm_nao_r1", "rough_b4", True), ("warm_nao_r1_logload", "rough_b4", True)),
    ("unsymmetrized -> symmetrized deployment", "g1",
     ("warm_r3_logload", "rough_b4", True), ("warm_r3_logload_sym", "rough_b4", True)),
]  # fmt: skip
MAPS = [("loads linear -> log space, 3 seeds each", "compare_seeds_logload_grid_and_corners.json")]
FAMILY = [("walk plans only -> 70 % procedural family", "nao", "nao2", "r1_3498_sym")]
"""Competence-map ablations: (change, body, tag before, tag after)."""
RETARGET = ("copy", "offset", "range")
"""The joint-target baseline's value maps (``transfer/retarget.py``)."""
ORACLE_SEEDS = ("42", "43")
"""The oracle's training seeds, also B2b's; 42's files carry no seed (``paper_baselines.sh``'s
``SEED``)."""
RETARGET_SEEDS = ("42", "43", "44")
"""B3's copy map's training seeds (B3s); the other maps have 42's only."""
GAIT = {
    "stride_length": "Stride length", "stride_period": "Stride period",
    "duty_factor_0": "Duty, left", "duty_factor_1": "Duty, right",
}  # fmt: skip
"""Gate C's signature features, as ``evaluate_contract.py``'s verdicts name them."""
DECODERS = [
    ("v1, distilled control", "warm_plain"),
    ("v2, physics prior", "warm_v2t_r8"),
    ("v2, deep-learning core", "warm_core"),
]

directories: list[Path] = [ROOT / "transfer" / "results"]


def find(name: str) -> Path | None:
    for directory in directories:
        if (directory / name).exists():
            return directory / name
    return None


def read(name: str) -> dict | None:
    path = find(name)
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    """``value`` with its non-finite numbers as None, which JSON can carry and NaN cannot."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def zero_shot_pattern(body: str, tag: str, generator: str, kinematic: bool = True) -> str:
    """The deployed path's L4 reports, ``{}`` standing for the command."""
    est = "_kin" if kinematic else ""
    return f"zero_shot_{body}_{tag}_{generator}_auto{est}_l4_{{}}.json"


def l4(body: str, tag: str, generator: str, kinematic: bool = True) -> dict | None:
    """The deployed path's L4 over the three commands, or None until all three exist."""
    return l4_files(zero_shot_pattern(body, tag, generator, kinematic), "encoder")


def l4_files(pattern: str, name: str) -> dict | None:
    """Run ``name`` of the L4 reports ``pattern.format(command)``, summed over the commands.

    ``deviation`` keeps each gate C feature's |relative deviation| per command.
    """
    total = {"falls": 0, "episodes": 0, "passes": 0, "speed_ratio": []}
    total["deviation"] = {f: [] for f in GAIT}
    for command in COMMANDS:
        report = read(pattern.format(command))
        if report is None:
            return None
        run = report["runs"][name]
        total["falls"] += run["falls"]
        total["episodes"] += run["episodes"]
        total["passes"] += bool(run["verdict"]["passes"])
        total["speed_ratio"].append(run["verdict"]["speed_ratio"])
        for f in GAIT:
            total["deviation"][f].append(abs(run["verdict"]["signature_deviation"][f]))
    return total


def oracle_pattern(skill: str, seed: str, variant: str = "") -> str:
    """B2's L4 reports on ``skill``, trained with ``seed``; ``variant`` "bounded" is B2b's."""
    tag = "" if seed == ORACLE_SEEDS[0] else f"_s{seed}"
    v = f"_{variant}" if variant else ""
    return f"oracle_nao_{skill}{v}{tag}_l4_{{}}.json"


def retarget_pattern(skill: str, value_map: str, seed: str = "42") -> str:
    tag = "" if seed == RETARGET_SEEDS[0] else f"_s{seed}"
    return f"retarget_nao_{skill}_{value_map}{tag}_l4_{{}}.json"


def oracle(skill: str, seed: str) -> dict | None:
    return l4_files(oracle_pattern(skill, seed), "oracle")


def nao_method_files(skill: str) -> dict[str, tuple[str, str]]:
    """Each of the NAO's methods on ``skill``: its L4 reports' pattern and its run's name.

    Zero-shot, B1, B2's and B2b's seeds, B3's maps and the copy map's other seeds.
    """
    generator = GENERATOR[skill]
    task_tracker = zero_shot_pattern("nao", f"nao_task_{skill}_sym", generator)
    files = {
        "zero-shot": (zero_shot_pattern("nao", MODEL["nao"], generator), "encoder"),
        "B1, task tracker": (task_tracker, "encoder"),
    }
    for seed in ORACLE_SEEDS:
        files[f"B2, oracle, seed {seed}"] = (oracle_pattern(skill, seed), "oracle")
    for seed in ORACLE_SEEDS:
        files[f"B2b, bounded, seed {seed}"] = (oracle_pattern(skill, seed, "bounded"), "oracle")
    for m in RETARGET:
        files[f"B3, {m}"] = (retarget_pattern(skill, m), "imitate")
    for seed in RETARGET_SEEDS[1:]:
        files[f"B3, copy, seed {seed}"] = (retarget_pattern(skill, "copy", seed), "imitate")
    return files


def nao_methods(skill: str) -> dict[str, dict | None]:
    """The NAO's L4 on ``skill`` by each method (:func:`nao_method_files`)."""
    return {name: l4_files(*files) for name, files in nao_method_files(skill).items()}


def walk(body: str, tag: str) -> dict | None:
    report = read(f"zero_shot_{body}_{tag}_walk_auto_kin.json")
    if report is None:
        return None
    run = report["runs"]["encoder"]
    return {"falls": run["falls"], "episodes": run["episodes"], "speed_ratio": run["verdict"]["speed_ratio"]}  # fmt: skip  # noqa: E501


def competence(body: str, tag: str, suffix: str, group: str) -> dict | None:
    report = read(f"competence_{body}_{tag}{suffix}.json")
    if report is None:
        return None
    return {k: report["summary"][group][k] for k in ("passed", "probes")}


def cell_l4(r: dict | None) -> str:
    if r is None:
        return "—"
    low, high = min(r["speed_ratio"]), max(r["speed_ratio"])
    return f"{r['falls']}/{r['episodes']}, passes {r['passes']}/3, speed {low:.2f}–{high:.2f}"


def cell_walk(r: dict | None) -> str:
    return "—" if r is None else f"{r['falls']}/{r['episodes']}, {r['speed_ratio']:.2f}"


def cell_competence(r: dict | None) -> str:
    return "—" if r is None else f"{r['passed']}/{r['probes']}"


def section(title: str, note: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"### {title}", "", note, ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build() -> tuple[str, dict]:
    sections = []
    data = {key: {} for key in ("M", "S", "B1", "B2", "B3", "gait", "SE", "T", "A", "D")}

    rows = []
    for body, tag in MODEL.items():
        for skill, generator in GENERATOR.items():
            r = data["M"][f"{body}/{skill}"] = l4(body, tag, generator)
            rows.append([body.upper(), skill, cell_l4(r)])
    sections.append(
        section(
            "M — the adopted model, zero-shot, deployed path",
            "Falls over the episodes of 3 commands × 64, commands that pass the teacher's gait "
            "signature, and speed over the teacher's. Source: "
            f"`zero_shot_<body>_<tag>_<generator>_auto_kin_l4_*.json`, tags `{MODEL['g1']}`, "
            f"`{MODEL['nao']}`.",
            ["Body", "Skill", "L4"],
            rows,
        )
    )

    rows = []
    for body, entries in SEEDS.items():
        for seed, tag in entries:
            r = {s: l4(body, tag, g) for s, g in GENERATOR.items()} | {"walk": walk(body, tag)}
            if all(v is None for v in r.values()):
                continue
            data["S"][f"{body}/{seed}"] = r
            cells = [cell_l4(r["rough"]), cell_l4(r["flat"]), cell_walk(r["walk"])]
            rows.append([body.upper(), seed, *cells])
    if rows:
        sections.append(
            section(
                "S — seeds of each body's final training stage, symmetrized",
                "Walk: S0's check, autonomous reference, falls and speed over the teacher's. "
                "Source: `paper_queue.sh` stages `s_g1`, `s_nao`.",
                ["Body", "Seed", "Rough L4", "Flat L4", "Walk"],
                rows,
            )
        )

    rows = []
    for body, tag in MODEL.items():
        for skill, generator in GENERATOR.items():
            task = l4(body, f"{body}_task_{skill}_sym", generator)
            if task is None:
                continue
            zero = l4(body, tag, generator)
            data["B1"][f"{body}/{skill}"] = {"zero_shot": zero, "task_specific": task}
            rows.append([body.upper(), skill, cell_l4(zero), cell_l4(task)])
    if rows:
        sections.append(
            section(
                "B1 — zero-shot against a tracker trained on the skill's own plans (N × M)",
                "Source: `paper_queue.sh` stages `b1_nao`, `b1_g1`.",
                ["Body", "Skill", "Zero-shot (adopted)", "Task-specific tracker"],
                rows,
            )
        )

    rows = []
    for skill, generator in GENERATOR.items():
        seeds = {s: oracle(skill, s) for s in ORACLE_SEEDS}
        if all(r is None for r in seeds.values()):
            continue
        zero = l4("nao", MODEL["nao"], generator)
        data["B2"][f"nao/{skill}"] = {"zero_shot": zero} | {f"oracle_seed_{s}": r for s, r in seeds.items()}  # fmt: skip  # noqa: E501
        rows.append(["NAO", skill, cell_l4(zero), *(cell_l4(r) for r in seeds.values())])
    if rows:
        sections.append(
            section(
                "B2 — zero-shot against the oracle: the NAO trained on the skill's own task",
                "The oracle is Isaac Lab's G1 recipe in the NAO's units "
                "(`nao/tasks/nao_velocity`), scored against the G1's teacher like the zero-shot "
                "tracker. Its gait is its own: it was rewarded for the task, not the G1's gait. "
                "One column per training seed. Source: `paper_baselines.sh` stage `b2`, "
                "`SEED=43 ORACLE_SKILLS=rough` for the second.",
                [
                    "Body",
                    "Skill",
                    "Zero-shot (adopted)",
                    *(f"Oracle, seed {s}" for s in ORACLE_SEEDS),
                ],
                rows,
            )
        )

    rows = []
    for skill, generator in GENERATOR.items():
        maps = {m: l4_files(f"retarget_nao_{skill}_{m}_l4_{{}}.json", "imitate") for m in RETARGET}
        if all(r is None for r in maps.values()):
            continue
        zero = l4("nao", MODEL["nao"], generator)
        data["B3"][f"nao/{skill}"] = {"zero_shot": zero, **maps}
        rows.append(["NAO", skill, cell_l4(zero), *(cell_l4(maps[m]) for m in RETARGET)])
    if rows:
        sections.append(
            section(
                "B3 — zero-shot against joint retargeting: the NAO imitating the G1's joints",
                "The teacher's joint trajectories through the correspondence, on the NAO's clock, "
                "imitated DeepMimic-style (`nao/tasks/nao_imitate`); one column per value map "
                "(`transfer/retarget.py`). At test time each imitates the teacher's own play "
                "capture at the command. Source: `paper_baselines.sh` stage `b3`.",
                ["Body", "Skill", "Zero-shot (adopted)", *(f"Retarget, {m}" for m in RETARGET)],
                rows,
            )
        )

    rows = []
    for skill in GENERATOR:
        for method, r in nao_methods(skill).items():
            if r is None:
                continue
            data["gait"][f"nao/{skill}/{method}"] = r["deviation"]
            # A command on which the NAO takes no stride has no stride length or period.
            cells = []
            for f in GAIT:
                values = r["deviation"][f]
                gone = [c for c, v in zip(COMMANDS, values, strict=True) if not math.isfinite(v)]
                cells.append(f"no stride on {', '.join(gone)}" if gone else f"{sum(values) / len(values):.2f}")  # fmt: skip  # noqa: E501
            rows.append(["NAO", skill, method, *cells])
    if any(row[2] != "zero-shot" for row in rows):
        sections.append(
            section(
                "B — the NAO's gait against the teacher's, by method",
                "Mean over the three L4 commands of each gate C feature's |relative deviation| "
                "(the gate allows 0.10). A method that fails gate C fails it here or on speed and "
                "falls, in the tables above. A command on which the NAO takes no stride has no "
                "stride length or period; the cell names it.",
                ["Body", "Skill", "Method", *GAIT.values()],
                rows,
            )
        )

    bootstrap = PAPER / "data" / "bootstrap.json"
    if bootstrap.exists():
        report = json.loads(bootstrap.read_text(encoding="utf-8"))
        data["SE"] = report["skills"]
        rows = []

        def value(v: float | None, se: float | None, sign: str = "") -> str:
            return "—" if v is None or se is None else f"{v:{sign}.3f} ± {se:.3f}"

        for skill, by in report["skills"].items():
            for group, e in by["seeds"].items():
                d = e.get("minus_baseline", {}).get("largest")
                minus = "—" if d is None else value(d["difference"], d["se"], "+")
                seeds = [by["runs"][n]["point"]["largest"] for n in e["runs"]]
                largest = value(e["mean"]["largest"], e["se"]["largest"])
                largest += f" (seeds {min(seeds):.3f}–{max(seeds):.3f})"
                label = "zero-shot, S's seeds" if group == "zero-shot" else group
                rows.append(["NAO", skill, label, str(len(e["runs"])), largest, minus])
            grouped = {n for e in by["seeds"].values() for n in e["runs"]}
            for name, e in by["runs"].items():
                if name not in grouped:
                    largest = value(e["point"]["largest"], e["se"]["largest"])
                    label = "zero-shot, adopted" if name == "zero-shot" else name
                    rows.append(["NAO", skill, label, "1", f"{largest} (environments)", "one seed"])
        sections.append(
            section(
                "SE — the NAO's methods against the zero-shot model, by seed",
                "Each command's largest gate C deviation, averaged over the three commands, as "
                "the figure reads it. A method trained with several seeds: the mean over its "
                "seeds ± their standard deviation over the root of their number, and its "
                "difference from the zero-shot model's seeds (S's; the adopted model is not "
                "among them) ± both errors combined. A method trained with one seed has only "
                "the bootstrap error of its 32 environments, far smaller than the spread "
                "between seeds (compare the zero-shot model's rows): it can only be read "
                "against the range of the zero-shot model's seeds. Source: "
                f"`paper_bootstrap.py`, {report['draws']} draws, "
                "`transfer/paper/data/bootstrap.json`.",
                ["Body", "Skill", "Method", "Seeds", "Largest deviation", "Minus zero-shot"],
                rows,
            )
        )

    decomposition = PAPER / "data" / "decomposition.json"
    if decomposition.exists():
        report = json.loads(decomposition.read_text(encoding="utf-8"))
        data["T"] = report["summary"]
        rows = [
            [key, term] + [f"{values[k]:.3f}" for k in report["features"]]
            for key, terms in report["summary"].items()
            for term, values in terms.items()
        ]
        sections.append(
            section(
                "T — the gait's deviation from the teacher, decomposed",
                "Mean over the three commands of the |signed relative deviation|. Signed, "
                "total = generator + tracker exactly; estimation = deployed path − true-state "
                "path. Source: `paper_decomposition.py`, `transfer/paper/data/decomposition.json`.",
                ["Body/skill", "Term", *report["features"]],
                rows,
            )
        )

    rows = []
    for change, body, before, after in ABLATIONS:
        a, b = l4(body, *before), l4(body, *after)
        if a is None or b is None:
            continue
        data["A"][f"{body}: {change}"] = {"before": a, "after": b}
        rows.append([body.upper(), change.replace("->", "→"), cell_l4(a), cell_l4(b)])
    for change, name in MAPS:
        r = read(name)
        if r is None:
            continue
        data["A"][f"g1 map: {change}"] = r
        per_seed = r["passed_per_seed"]
        before = ", ".join(f"{x:.0f}" for x in per_seed["reference"])
        after = ", ".join(f"{x:.0f}" for x in per_seed["candidate"])
        after += f" ({r['passed_difference']:+.2f} ± {r['passed_se']:.2f})"
        rows.append(["G1", f"map + corners passed: {change}".replace("->", "→"), before, after])
    for change, body, before, after in FAMILY:
        a, b = read(f"competence_{body}_{before}.json"), read(f"competence_{body}_{after}.json")
        if a is None or b is None:
            continue
        data["A"][f"{body} map: {change}"] = {"before": a["summary"], "after": b["summary"]}
        cells = [
            ", ".join(f"{g} {s[g]['passed']}/{s[g]['probes']}" for g in s)
            for s in (a["summary"], b["summary"])
        ]
        rows.append([body.upper(), f"map passed: {change}".replace("->", "→"), *cells])
    if rows:
        sections.append(
            section(
                "A — ablations",
                "Rough L4 on the deployed path, one change at a time, unless the row says "
                "otherwise. The tags are in `paper_tables.py`'s `ABLATIONS`.",
                ["Body", "Change", "Before", "After"],
                rows,
            )
        )

    rows = []
    for arm, tag in DECODERS:
        rough = l4("g1", tag, "rough_b4")
        if rough is None:
            continue
        grid = competence("g1", tag, "", "grid")
        corners = competence("g1", tag, "_corners", "corner")
        data["D"][arm] = {"rough": rough, "grid": grid, "corners": corners}
        rows.append([arm, cell_l4(rough), cell_competence(grid), cell_competence(corners)])
    if rows:
        sections.append(
            section(
                "D — decoder v2 at equal competence (G1, warm-started from the same incumbent)",
                "Grid and corners: probes mastered. Source: `warm_study.sh`, `core_and_nao.sh`.",
                ["Decoder", "Rough L4", "Grid", "Corners"],
                rows,
            )
        )
    return "\n\n".join(sections), data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, action="append", default=[])
    args = parser.parse_args()
    directories.extend(args.results)
    text, data = build()
    readme = PAPER / "README.md"
    start, end = "<!-- tables:start -->", "<!-- tables:end -->"
    head, rest = readme.read_text(encoding="utf-8").split(start, 1)
    tail = rest.split(end, 1)[1]
    readme.write_text(f"{head}{start}\n\n{text}\n\n{end}{tail}", encoding="utf-8", newline="\n")
    (PAPER / "data").mkdir(parents=True, exist_ok=True)
    (PAPER / "data" / "tables.json").write_text(json.dumps(finite(data), indent=2) + "\n", encoding="utf-8")  # fmt: skip  # noqa: E501
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
