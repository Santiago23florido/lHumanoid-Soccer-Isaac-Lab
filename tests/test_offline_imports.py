"""The offline scripts the cluster runs import with torch and NumPy only (no Isaac, no rsl_rl)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OFFLINE = (
    "pact_gate_d", "pact_gate_x", "pact_gate_a3", "pact_gate_n1", "icml_r4_read", "icml_runs",
    "icml_tables", "icml_cluster_data", "body_conditions", "pact_archive_captures",
)  # fmt: skip
BLOCKED = ("isaaclab", "isaacsim", "omni", "pxr", "carb", "rsl_rl", "gymnasium")


def test_offline_scripts_import_without_isaac_or_rsl_rl():
    code = (
        "import sys, importlib\n"
        f"for m in {BLOCKED!r}: sys.modules[m] = None\n"
        "sys.path.insert(0, 'transfer/scripts'); sys.argv = ['x']\n"
        f"for m in {OFFLINE!r}: importlib.import_module(m)\n"
    )
    env = {"CUDA_VISIBLE_DEVICES": "-1", "PATH": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
