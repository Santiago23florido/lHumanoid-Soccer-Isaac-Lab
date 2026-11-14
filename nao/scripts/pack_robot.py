"""Pack what the robot needs into one zip: the runtime package and a deployment bundle.

The archive holds ``deploy/`` (the NumPy runtime and its tools, no tests, no caches)
and ``bundle/`` (runtime.json, body_model.json, actor.npz, plans/), so on the robot::

    unzip nao_robot.zip && python3 -m deploy.tools.check_sensors --lola --bundle bundle

The bundle's body model is derived from the URDF: the archive is for the robot, not
for sharing.

Usage::

    python nao/scripts/pack_robot.py nao/bundles/<name> --output nao/bundles/<name>_robot.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "source" / "humanoid_transfer" / "humanoid_transfer" / "nao" / "deploy"
BUNDLE_FILES = ("runtime.json", "body_model.json", "actor.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(DEPLOY.rglob("*.py")):
            archive.write(path, Path("deploy") / path.relative_to(DEPLOY))
            count += 1
        for name in BUNDLE_FILES:
            archive.write(args.bundle / name, Path("bundle") / name)
            count += 1
        for plan in sorted((args.bundle / "plans").glob("*.npz")):
            archive.write(plan, Path("bundle") / "plans" / plan.name)
            count += 1
    print(f"wrote {args.output} ({count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
