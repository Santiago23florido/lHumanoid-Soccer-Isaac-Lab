"""What every tool shares: choosing a backend from the command line, and saving records."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..backends import LolaBackend, RelayBackend


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lola", action="store_true", help="On the robot: LoLA's socket.")
    group.add_argument("--relay", metavar="ROBOT", help="From a laptop: relay.py on ROBOT.")
    parser.add_argument("--port", type=int, default=10500)


def backend_from(args: argparse.Namespace):
    if args.lola:
        return LolaBackend()
    return RelayBackend(args.relay, args.port)


def collect(backend, seconds: float) -> list:
    """Every reading for ``seconds`` (one per LoLA cycle)."""
    readings, end = [], time.perf_counter() + seconds
    while time.perf_counter() < end:
        readings.append(backend.read())
    return readings


def save(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


__all__ = ["add_backend_args", "backend_from", "collect", "save", "stamp"]
