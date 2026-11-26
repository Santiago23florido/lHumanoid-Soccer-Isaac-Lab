"""The data the manifest's offline jobs read, listed or packed for a machine without Isaac.

The code comes from the repository; the data are each body's final-teacher audit captures (the
base data), the final co-executions (encoder and translator gates), and the walk contracts that
fix each body's units. Everything else an offline job reads (cards, mirrors, registries) is in
the repository. Models a job needs are made by earlier jobs on the same machine.

    python transfer/scripts/icml_cluster_data.py                    # list and total size
    python transfer/scripts/icml_cluster_data.py --tar logs/icml/cluster_data.tar
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

from body_conditions import TEACHERS, registry  # noqa: E402
from icml_runs import COEXEC  # noqa: E402


def files() -> list[str]:
    out: list[str] = []
    contracts: set[str] = set()
    for body, entry in registry(TEACHERS).items():
        v = entry["variant"]
        out += [f"logs/pact/teachers/{body}_{v}_probes.npz", f"logs/pact/teachers/{body}_{v}_train.npz"]
        out += [f"{COEXEC}/{body}_map.npz", f"{COEXEC}/{body}_replay.npz"]
        contracts.add(entry["conditions"]["HUMANOID_CONTRACT"])
    return out + sorted(contracts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tar", type=Path, default=None, help="write an uncompressed tar of them")
    args = parser.parse_args()
    paths = files()
    missing = [p for p in paths if not (REPO / p).exists()]
    size = sum((REPO / p).stat().st_size for p in paths if p not in missing)
    for p in paths:
        print(("MISSING " if p in missing else "") + p)
    print(f"{len(paths) - len(missing)}/{len(paths)} present, {size / 1e9:.1f} GB")
    if args.tar is not None:
        if missing:
            raise SystemExit("missing files: not packed")
        args.tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.tar, "w") as tar:  # npz are compressed already
            for p in paths:
                tar.add(REPO / p, arcname=p)
        print(f"written {args.tar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
