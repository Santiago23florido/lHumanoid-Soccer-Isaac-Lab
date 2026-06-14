#!/usr/bin/env python
"""Run one headless Isaac Sim physics step on the CUDA device."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1, help="Number of physics steps to execute.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def main() -> None:
    sim = SimulationContext(SimulationCfg(device=args_cli.device or "cuda:0"))
    sim.reset()
    for _ in range(args_cli.steps):
        sim.step()
    print("ISAAC_SIM_CUDA_STEP_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if sys.platform == "win32":
            os._exit(1)
        simulation_app.close()
        raise
    else:
        simulation_app.close(skip_cleanup=sys.platform == "win32")
