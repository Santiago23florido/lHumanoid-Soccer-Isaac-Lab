"""A body's teacher and tracking conditions as shell assignments, from the registries.

``transfer/bodies/bodies.json`` describes the body; ``transfer/bodies/teachers.json`` names its
chosen teacher and the conditions its trackers train and run under. Scripts read every body the
same way instead of carrying one branch per robot::

    eval "$(python transfer/scripts/body_conditions.py h1 --check)" || exit 1

Assignments: the conditions (exported, absolute paths; a condition the body does not set is
unset), then, each name after ``--prefix``: ``run``, ``checkpoint``, ``policy`` (the teacher as
deployed, symmetrised when the entry says so), ``plain_policy``, ``symmetrised`` (1 or 0),
``contract`` (the walk contract,
repository-relative), ``variant`` and its audit captures ``probes_capture`` / ``train_capture``,
``play_task``, ``train_task``, ``rollouts``, ``mirror``, ``unit`` (the body's length unit,
metres) and ``speed_scale`` / ``turn_scale``, the Froude factors that carry a command written in
G1 units to the body (``sqrt(L / L_g1)`` and its inverse). ``--bodies`` prints the registered
bodies instead. With ``--check`` a missing teacher run or policy is an error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BODIES = REPO / "transfer/bodies/bodies.json"
TEACHERS = REPO / "transfer/bodies/teachers.json"
RUNS = REPO / "logs/rsl_rl"
AUDIT = "logs/pact/teachers"
# Every condition any body sets; a body that does not set one runs with it unset.
CONDITIONS = ("HUMANOID_CONTRACT", "HUMANOID_LOAD_SPACE", "HUMANOID_POSTURE_CONTRACT")
PATH_CONDITIONS = ("HUMANOID_CONTRACT", "HUMANOID_POSTURE_CONTRACT")
REFERENCE_BODY = "g1"  # commands are written in its units


def registry(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text("utf-8")).items() if not k.startswith("$")}


def teacher_bodies() -> list[str]:
    """The bodies with a chosen teacher, in the registry's order."""
    return list(registry(TEACHERS))


def rollouts_dir(body: str) -> Path:
    """Where a body's evaluation captures are (``<body>/rollouts`` for a body not registered)."""
    teachers = registry(TEACHERS)
    return REPO / (teachers[body]["rollouts"] if body in teachers else f"{body}/rollouts")


def length_unit(entry: dict) -> float:
    """The body's length unit: the registry's declared unit, else its card's leg length."""
    if entry.get("leg_length_unit") is not None:
        return float(entry["leg_length_unit"])
    return float(json.loads((REPO / entry["card"]).read_text("utf-8"))["leg_length"])


def task_name(body: str) -> str:
    """``h1`` -> ``H1Track``: the body's tracking task without its suffix."""
    return f"{body[:1].upper()}{body[1:]}Track"


def latest_run(body: str, tag: str, runs: Path = RUNS) -> Path | None:
    found = sorted((runs / f"{body}_track").glob(f"*_{tag}_{body}"))
    return found[-1] if found else None


def latest_checkpoint(run: Path) -> str | None:
    steps = [
        int(m.group(1)) for p in run.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))
    ]
    return f"model_{max(steps)}" if steps else None


def resolve(body: str, root: Path = REPO, runs: Path = RUNS) -> dict:
    """Everything a script needs about ``body``'s teacher (condition paths absolute under ``root``)."""
    bodies, teachers = registry(BODIES), registry(TEACHERS)
    if body not in bodies or body not in teachers:
        raise SystemExit(f"{body}: not in {BODIES.name} and {TEACHERS.name} (known: {sorted(teachers)})")
    entry, teacher = bodies[body], teachers[body]
    unknown = set(teacher["conditions"]) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"{body}: conditions not known to body_conditions.py: {sorted(unknown)}")
    conditions = {}
    for key in CONDITIONS:
        value = teacher["conditions"].get(key)
        if value is not None and key in PATH_CONDITIONS:
            value = (root / value).as_posix()
        conditions[key] = value
    run = latest_run(body, teacher["teacher"], runs)
    checkpoint = latest_checkpoint(run) if run is not None else None
    plain = policy = None
    if checkpoint is not None:
        plain = run / "exported" / f"{checkpoint}.pt"
        policy = run / "exported" / f"{checkpoint}_sym_pact.pt" if teacher["symmetrised"] else plain
    unit, reference = length_unit(entry), length_unit(bodies[REFERENCE_BODY])
    variant = teacher["variant"]
    return {
        "conditions": conditions,
        "run": run,
        "checkpoint": checkpoint,
        "policy": policy,
        "plain_policy": plain,
        "symmetrised": "1" if teacher["symmetrised"] else "0",
        "contract": teacher["conditions"]["HUMANOID_CONTRACT"],
        "variant": variant,
        "probes_capture": f"{AUDIT}/{body}_{variant}_probes.npz",
        "train_capture": f"{AUDIT}/{body}_{variant}_train.npz",
        "play_task": f"{task_name(body)}-Play-v0",
        "train_task": f"{task_name(body)}-v0",
        "rollouts": teacher["rollouts"],
        "mirror": teacher["mirror"],
        "unit": unit,
        "speed_scale": math.sqrt(unit / reference),
        "turn_scale": math.sqrt(reference / unit),
    }


def relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


TEXT = (
    "checkpoint", "symmetrised", "contract", "variant", "probes_capture", "train_capture",
    "play_task", "train_task", "rollouts", "mirror",
)


def shell(body: str, check: bool = False, prefix: str = "", runs: Path = RUNS) -> str:
    r = resolve(body, runs=runs)
    if check:
        if r["run"] is None:
            raise SystemExit(f"{body}: no run tagged {registry(TEACHERS)[body]['teacher']}")
        if r["policy"] is None or not r["policy"].exists():
            raise SystemExit(f"{body}: no teacher policy {r['policy']} (export and symmetrise it first)")
    lines = []
    for key, value in r["conditions"].items():
        lines.append(f"export {key}={shlex.quote(value)}" if value is not None else f"unset {key}")
    for key in ("run", "policy", "plain_policy"):
        lines.append(f"{prefix}{key}={shlex.quote(relative(r[key], REPO))}")
    for key in TEXT:
        lines.append(f"{prefix}{key}={shlex.quote(r[key] or '')}")
    for key in ("unit", "speed_scale", "turn_scale"):
        lines.append(f"{prefix}{key}={r[key]!r}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("body", nargs="?")
    parser.add_argument("--check", action="store_true", help="fail if the teacher run or policy is missing")
    parser.add_argument("--prefix", default="", help="prefix of the non-exported names")
    parser.add_argument("--bodies", action="store_true", help="print the registered bodies")
    args = parser.parse_args()
    if args.bodies:
        print(" ".join(teacher_bodies()))
        return 0
    if not args.body:
        parser.error("a body is needed (or --bodies)")
    if args.prefix and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.prefix):
        parser.error("--prefix must be a shell identifier")
    print(shell(args.body, args.check, args.prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
