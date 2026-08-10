"""Short Isaac Lab startup diagnostic.

This launches Isaac Sim, imports the Isaac Lab task extension, then exits. It is
used to verify environment/DLL problems without opening a long-running viewer.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose Isaac Lab startup imports.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401

print("[INFO]: isaaclab_tasks import ok.")
simulation_app.close()
