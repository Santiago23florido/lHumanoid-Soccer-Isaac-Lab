"""The paper's run manifest: unique ids, known needs, existing scripts, SLURM files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "transfer" / "scripts"))

import icml_runs  # noqa: E402


def test_ids_are_unique_and_needs_exist():
    jobs = icml_runs.manifest()
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids))
    for j in jobs:
        assert set(j.needs) <= set(ids), j.id
        assert j.kind in ("isaac", "offline"), j.id
        assert j.outputs and j.hours > 0, j.id


def test_every_command_names_a_script_that_exists():
    for j in icml_runs.manifest():
        for script in re.findall(r"transfer/scripts/[\w./]+\.(?:py|sh)", j.command):
            assert (REPO / script).exists(), (j.id, script)


def test_every_arm_and_seed_reaches_the_main_tables():
    main = {j.id for j in icml_runs.manifest() if j.group == "main"}
    assert main == {f"main/{a}_s{s}" for a in icml_runs.ARMS for s in icml_runs.SEEDS}


def test_slurm_writes_one_array_per_group(tmp_path):
    jobs = [
        icml_runs.Job("g/a", "g", "offline", "echo a", ["nope/a"], 1.0),
        icml_runs.Job("g/b", "g", "offline", "echo b", ["nope/b"], 1.0),
        icml_runs.Job("h/c", "h", "isaac", "echo c", ["nope/c"], 1.0),
    ]
    icml_runs.slurm(jobs, "offline", tmp_path, "gpu", 2)
    script = (tmp_path / "g.sbatch").read_text()
    assert "#SBATCH --array=0-1" in script and "#SBATCH --partition=gpu" in script
    assert (tmp_path / "g.commands").read_text().splitlines() == ["echo a", "echo b"]
    assert not (tmp_path / "h.sbatch").exists()
