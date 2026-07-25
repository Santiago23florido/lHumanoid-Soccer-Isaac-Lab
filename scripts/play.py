"""Play entry point placeholder for the humanoid soccer task."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="HumanoidSoccer-Direct-v0")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(
        f"{args.task} is scaffolded only. Implement the Isaac Lab environment before play."
    )


if __name__ == "__main__":
    main()
