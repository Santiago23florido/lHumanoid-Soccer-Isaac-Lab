"""Remove the intermediate training checkpoints nothing reads, keeping each run's last.

A PPO run saves ``model_<iteration>.pt`` every few dozen iterations (5-10 MB each). Kept per run:
the last checkpoint, and every ``model_<n>.pt`` whose name ``model_<n>`` appears in a file under
``--references``: scripts, results and bundles that load a particular checkpoint. Exported
TorchScript policies (``exported/``) and training logs are never touched. Dry run unless
``--delete``.

    python transfer/scripts/prune_checkpoints.py logs/rsl_rl            # what would go
    python transfer/scripts/prune_checkpoints.py logs/rsl_rl --delete
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REFERENCES = ("transfer/scripts", "transfer/results", "nao/scripts", "nao/bundles", "nao/results", "g1/scripts")
CHECKPOINT = re.compile(r"model_(\d+)\.pt")


def referenced(roots: list[Path]) -> set[str]:
    names: set[str] = set()
    for root in roots:
        files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for path in files:
            if path.suffix not in (".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml"):
                continue
            names.update(re.findall(r"model_\d+", path.read_text("utf-8", errors="ignore")))
    return names


def prunable(runs_root: Path, keep: set[str]) -> list[Path]:
    out = []
    for run in sorted({p.parent for p in runs_root.rglob("model_*.pt") if p.parent.name != "exported"}):
        steps = sorted(
            (int(m.group(1)), p) for p in run.glob("model_*.pt") if (m := CHECKPOINT.fullmatch(p.name))
        )
        for step, path in steps[:-1]:  # the last one always stays
            if f"model_{step}" not in keep:
                out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", type=Path, help="e.g. logs/rsl_rl")
    parser.add_argument("--references", nargs="*", default=list(REFERENCES))
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    keep = referenced([REPO / r for r in args.references if (REPO / r).exists()])
    paths = prunable(REPO / args.runs if not args.runs.is_absolute() else args.runs, keep)
    size = sum(p.stat().st_size for p in paths)
    for p in paths:
        if args.delete:
            p.unlink()
    verb = "removed" if args.delete else "would remove"
    print(f"{verb} {len(paths)} checkpoints, {size / 1e9:.1f} GB (kept every run's last and {len(keep)} referenced names)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
