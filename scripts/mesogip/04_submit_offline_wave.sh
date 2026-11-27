#!/usr/bin/env bash
# Generate and optionally submit Mesogip-compliant SLURM array jobs from transfer/scripts/icml_runs.py.
#
# Usage (from repository root on Mesogip):
#   bash scripts/mesogip/04_submit_offline_wave.sh <group> [partition] [max_concurrent] [walltime] [--submit]
#
# Examples:
#   bash scripts/mesogip/04_submit_offline_wave.sh cross ENSTA-h100 2 01:30:00 --submit
#   bash scripts/mesogip/04_submit_offline_wave.sh r4    ENSTA-h100 2 01:30:00 --submit
#   bash scripts/mesogip/04_submit_offline_wave.sh loo   ENSTA-h100 2 02:30:00 --submit
#
# Compliance with Mesogip cluster rules:
#   - Never runs compute on login node.
#   - Walltime always <= 03:50:00 with --signal=B:USR1@120.
#   - Concurrency throttled via --array=0-N%<max_concurrent> (default 2 GPUs) for fairshare politeness.
#   - Uses node-local /tmp for caches and writes structured provenance JSON per job for software reporting.
set -euo pipefail

GROUP="${1:?usage: $0 <group: cross|r4|loo|scaling|adapt> [partition] [max_concurrent] [walltime] [--submit]}"
PARTITION="${2:-ENSTA-h100}"
MAX_CONC="${3:-2}"
WALLTIME="${4:-02:30:00}"
DO_SUBMIT="${5:-}"
AFTER_JOB="${6:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${MESOGIP_VENV:-$HOME/venvs/pact-icml}"
OUT_DIR="$REPO_ROOT/logs/icml/mesogip"
PROV_DIR="$REPO_ROOT/logs/icml/mesogip_provenance"
mkdir -p "$OUT_DIR" "$PROV_DIR"

cd "$REPO_ROOT"
PYTHON="$VENV_DIR/bin/python"

# Build exact runnable task table (id, outputs, command) from the single source of truth icml_runs.py
TASK_JSON="$OUT_DIR/${GROUP}_tasks.json"
"$PYTHON" - "$GROUP" "$TASK_JSON" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "transfer/scripts")
import icml_runs

group, out_path = sys.argv[1], Path(sys.argv[2])
jobs = [j for j in icml_runs.runnable(icml_runs.manifest()) if j.kind == "offline" and j.group == group]
payload = [{"id": j.id, "group": j.group, "command": j.command, "outputs": j.outputs, "laptop_hours": j.hours} for j in jobs]
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"{group}: {len(payload)} runnable offline jobs written to {out_path}")
PY

COUNT="$("$PYTHON" -c "import json; print(len(json.load(open('$TASK_JSON'))))")"
if [ "$COUNT" -eq 0 ]; then
    echo "No runnable offline jobs left for group '$GROUP'."
    exit 0
fi
LAST=$((COUNT - 1))

SBATCH_FILE="$OUT_DIR/${GROUP}.sbatch"
cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=pact-${GROUP}
#SBATCH --partition=${PARTITION}
#SBATCH --array=0-${LAST}%${MAX_CONC}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=${WALLTIME}
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/icml/mesogip/${GROUP}_%A_%a.out
#SBATCH --error=logs/icml/mesogip/${GROUP}_%A_%a.err

set -euo pipefail
REPO_ROOT="\${SLURM_SUBMIT_DIR:-\$PWD}"
VENV_DIR="\${MESOGIP_VENV:-\$HOME/venvs/pact-icml}"
cd "\$REPO_ROOT"
source "\$VENV_DIR/bin/activate"

export PYTHON="\$VENV_DIR/bin/python"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export TORCH_HOME="/tmp/ensta_santiago_torch_\${SLURM_JOB_ID}"
mkdir -p "\$TORCH_HOME" logs/icml/models logs/icml/mesogip_provenance transfer/results

trap 'echo "[WARN] Approaching walltime limit (SIGUSR1) at \$(date -u). Re-run 04_submit_offline_wave.sh to resume unfinished tasks."' USR1

"\$PYTHON" - "$TASK_JSON" "\$SLURM_ARRAY_TASK_ID" <<'PYRUN'
import json, os, socket, subprocess, sys, time
from pathlib import Path
import torch

tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
idx = int(sys.argv[2])
task = tasks[idx]
job_id = task["id"]
outputs = task["outputs"]

if all(Path(o).exists() for o in outputs):
    print(f"[SKIP] {job_id}: all outputs already exist ({outputs})")
    raise SystemExit(0)

t0 = time.time()
started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"=== START {job_id} | host={socket.gethostname()} | gpu={gpu_name} | commit={git_commit[:8]} | {started_utc} ===")
print(f"CMD: {task['command']}")

rc = subprocess.run(["bash", "-c", task["command"]], cwd=os.getcwd()).returncode
elapsed = round(time.time() - t0, 1)
ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
all_exist = all(Path(o).exists() for o in outputs)

prov = {
    "job_id": job_id,
    "group": task["group"],
    "kind": "offline",
    "execution_site": "mesogip",
    "host": socket.gethostname(),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": idx,
    "gpu_model": gpu_name,
    "git_commit": git_commit,
    "started_utc": started_utc,
    "ended_utc": ended_utc,
    "wall_seconds": elapsed,
    "laptop_est_hours": task["laptop_hours"],
    "returncode": rc,
    "outputs": outputs,
    "outputs_present": all_exist,
    "status": "done" if (rc == 0 and all_exist) else "failed",
}
safe = job_id.replace("/", "__")
Path(f"logs/icml/mesogip_provenance/{safe}.json").write_text(
    json.dumps(prov, indent=2) + "\n", encoding="utf-8"
)
print(f"=== END {job_id} | status={prov['status']} | wall={elapsed}s ===")
raise SystemExit(0 if prov["status"] == "done" else (rc or 1))
PYRUN

rm -rf "\$TORCH_HOME"
EOF

chmod +x "$SBATCH_FILE"
echo "Generated $SBATCH_FILE ($COUNT jobs, partition=$PARTITION, max_concurrent=$MAX_CONC, walltime=$WALLTIME)"

if [ "$DO_SUBMIT" = "--submit" ]; then
    if [ -n "$AFTER_JOB" ]; then
        sbatch --dependency="afterany:${AFTER_JOB}" "$SBATCH_FILE"
    else
        sbatch "$SBATCH_FILE"
    fi
fi
