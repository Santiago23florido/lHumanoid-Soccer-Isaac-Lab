"""Archive superseded captures as a compact summary, then (with ``--delete``) remove them.

A capture is about 0.5 GB. Once the run it belongs to has been read and the arm was not
adopted, or the protocol it served was superseded, only its summary is kept. Per capture:
size, environments, steps and falls. For a training-task capture, also the termination
classes (``pact_cap_test_read.classify``: task, procedural, after a fallen procedural biped).
For a probe capture, the scored result already in ``transfer/results`` is named. The summary is
appended to ``--output`` (one entry per capture path, never overwritten), so the archive keeps
every capture ever removed.

Kept, never archived:
* each body's chosen teacher's audit captures (``transfer/bodies/teachers.json``: variant)
  and every capture a running or planned script reads (``--keep``).

Usage::

    python transfer/scripts/pact_archive_captures.py "logs/pact/teachers/*.npz" \\
        "logs/pact/dagger/*/r*/*.npz" --output transfer/results/icml_capture_archive.json [--delete]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from pact_cap_test_read import classify  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty task masks in classify


def chosen_teacher_patterns() -> list[str]:
    teachers = json.loads((REPO / "transfer/bodies/teachers.json").read_text("utf-8"))
    return [
        f"logs/pact/teachers/{body}_{entry['variant']}_*.npz"
        for body, entry in teachers.items()
        if not body.startswith("$")
    ]


def summary(path: Path) -> dict:
    entry = {"bytes": path.stat().st_size}
    with np.load(path) as data:
        done = data["done"]
        entry["steps"], entry["envs"] = int(done.shape[0]), int(done.shape[1])
        if "terminated" in data.files:
            entry["falls"] = int(data["terminated"].sum())
        files = set(data.files)
    if {"terminated", "procedural", "biped_fallen", "contract_due"} <= files:
        c = classify(path)
        entry["terminations"] = {k: c[k] for k in ("terminations", "task", "procedural", "biped_fallen")}
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("patterns", nargs="+", help="repo-relative globs of captures")
    parser.add_argument("--keep", action="append", default=[], help="repo-relative glob never archived")
    parser.add_argument("--output", type=Path, default=REPO / "transfer/results/icml_capture_archive.json")
    parser.add_argument("--delete", action="store_true", help="remove each capture once summarised")
    args = parser.parse_args()
    keep = chosen_teacher_patterns() + args.keep
    paths = sorted({p for pattern in args.patterns for p in REPO.glob(pattern) if p.suffix == ".npz"})
    archive = json.loads(args.output.read_text("utf-8")) if args.output.exists() else {"captures": {}}
    freed = 0
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        if any(fnmatch.fnmatch(rel, k) for k in keep):
            continue
        if rel not in archive["captures"]:
            try:
                archive["captures"][rel] = summary(path)
            except Exception as error:  # an unreadable capture is recorded, not kept
                archive["captures"][rel] = {"bytes": path.stat().st_size, "unreadable": str(error)}
            args.output.write_text(json.dumps(archive, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        if args.delete:
            freed += path.stat().st_size
            path.unlink()
        print(f"{'removed' if args.delete else 'summarised'} {rel}", flush=True)
    print(f"{len(archive['captures'])} captures in {args.output}; freed {freed / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
