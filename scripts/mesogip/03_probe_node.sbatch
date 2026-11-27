#!/bin/bash
#SBATCH --job-name=pact-probe
#SBATCH --partition=ENSTA-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=logs/icml/mesogip/%x_%j.out
#SBATCH --error=logs/icml/mesogip/%x_%j.err

# Short compute-node verification on Mesogip before submitting offline experiment arrays:
#   1. GPU / driver / CUDA availability
#   2. Offline unit test suite (verifying zero Isaac/rsl_rl dependency and numerical parity)
#   3. Compute-node outbound network check (informs whether L40S Apptainer Isaac runs can pull USDs)
#   4. Writes structured provenance JSON to logs/icml/mesogip_provenance/node_probe.json

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
VENV_DIR="${MESOGIP_VENV:-$HOME/venvs/pact-icml}"
cd "$REPO_ROOT"
mkdir -p logs/icml/mesogip logs/icml/mesogip_provenance

source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1

echo "=== Probe Job $SLURM_JOB_ID on $(hostname) (partition=$SLURM_JOB_PARTITION) at $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python -m pytest -q -p no:cacheprovider --basetemp "/tmp/pytest_pact_${SLURM_JOB_ID}" \
    tests/test_offline_imports.py \
    tests/test_drive_ratio.py \
    tests/test_icml_runs.py \
    tests/test_icml_r4_read.py \
    tests/test_pact_windows.py \
    tests/test_pact_mirror.py \
    tests/test_pact_model.py \
    tests/test_pact_descriptors.py

rm -rf "/tmp/pytest_pact_${SLURM_JOB_ID}"

python - <<'PY'
import json, os, platform, shutil, socket, subprocess, time, urllib.request
from pathlib import Path
import torch, numpy

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
drv = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
).strip().splitlines()[0]
omniverse_ok = False
try:
    req = urllib.request.Request(
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd",
        method="HEAD",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        omniverse_ok = resp.status == 200
except Exception:
    omniverse_ok = False

info = {
    "probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "gpu_name": gpu,
    "nvidia_driver": drv,
    "apptainer": shutil.which("apptainer"),
    "singularity": shutil.which("singularity"),
    "omniverse_usd_reachable": omniverse_ok,
    "offline_tests_passed": True,
}
out = Path("logs/icml/mesogip_provenance/node_probe.json")
out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
print(json.dumps(info, indent=2))
PY
echo "=== Probe finished at $(date -u) ==="
