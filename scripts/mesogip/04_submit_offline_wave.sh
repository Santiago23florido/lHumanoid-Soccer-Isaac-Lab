#!/usr/bin/env bash
# Generate and optionally submit Mesogip-compliant SLURM jobs from transfer/scripts/icml_runs.py.
#
# Supports two QOS-aware submission modes:
#   1. Packed mode (default, ideal under QOSMaxJobsPerUserLimit = 1):
#      Packs multiple short offline tasks sequentially into ONE resumable <= 03:50:00 job,
#      stopping cleanly with a safety margin before walltime and skipping already-completed outputs.
#   2. Array mode (--array):
#      Submits a throttled SLURM array (--array=0-N%<max_concurrent>).
#
# Usage (from repository root on Mesogip):
#   bash scripts/mesogip/04_submit_offline_wave.sh <groups_csv> [partition] [walltime] [--submit] [--after <jobid>] [--array <max_conc>]
#
# Examples:
#   bash scripts/mesogip/04_submit_offline_wave.sh cross,r4 ENSTA-h100 03:50:00 --submit
#   bash scripts/mesogip/04_submit_offline_wave.sh cross,r4 ENSTA-h100 03:50:00 --submit --after 20350
#   bash scripts/mesogip/04_submit_offline_wave.sh loo      ENSTA-h100 03:50:00 --submit --array 1
set -euo pipefail

WAVE_GROUPS="${1:?usage: $0 <groups_csv: cross,r4|loo|scaling|adapt> [partition] [walltime] [--submit] [--after <jobid>] [--array <max_conc>]}"
PARTITION="${2:-ENSTA-h100}"
WALLTIME="${3:-03:50:00}"
shift $(( $# < 3 ? $# : 3 ))

DO_SUBMIT=0
AFTER_JOB=""
ARRAY_CONC=0
SAFETY_MIN=15
while [ $# -gt 0 ]; do
    case "$1" in
        --submit) DO_SUBMIT=1; shift ;;
        --after) AFTER_JOB="$2"; shift 2 ;;
        --array) ARRAY_CONC="$2"; shift 2 ;;
        --safety-min) SAFETY_MIN="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${MESOGIP_VENV:-$HOME/venvs/pact-icml}"
OUT_DIR="$REPO_ROOT/logs/icml/mesogip"
PROV_DIR="$REPO_ROOT/logs/icml/mesogip_provenance"
mkdir -p "$OUT_DIR" "$PROV_DIR"

cd "$REPO_ROOT"
PYTHON="$VENV_DIR/bin/python"
WAVE_TAG="${WAVE_GROUPS//,/_}"
TASK_JSON="$OUT_DIR/wave_${WAVE_TAG}_tasks.json"

"$PYTHON" - "$WAVE_GROUPS" "$TASK_JSON" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "transfer/scripts")
import icml_runs

groups = [g.strip() for g in sys.argv[1].split(",") if g.strip()]
out_path = Path(sys.argv[2])
order = {g: i for i, g in enumerate(groups)}
jobs = [j for j in icml_runs.runnable(icml_runs.manifest()) if j.kind == "offline" and j.group in order]
jobs.sort(key=lambda j: (order[j.group], j.id))
payload = [{"id": j.id, "group": j.group, "command": j.command, "outputs": j.outputs, "laptop_hours": j.hours} for j in jobs]
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"wave[{','.join(groups)}]: {len(payload)} runnable offline jobs written to {out_path}")
PY

COUNT="$("$PYTHON" -c "import json; print(len(json.load(open('$TASK_JSON'))))")"
if [ "$COUNT" -eq 0 ]; then
    echo "No runnable offline jobs left for groups '$WAVE_GROUPS'."
    exit 0
fi

# Convert HH:MM:SS walltime to total seconds for soft pre-deadline stop in packed mode
IFS=: read -r WH WM WS <<< "$WALLTIME"
WALL_SEC=$((10#$WH * 3600 + 10#$WM * 60 + 10#$WS))
DEADLINE_SEC=$((WALL_SEC - SAFETY_MIN * 60))
[ "$DEADLINE_SEC" -lt 600 ] && DEADLINE_SEC=600

SBATCH_FILE="$OUT_DIR/wave_${WAVE_TAG}.sbatch"
ARRAY_HEADER=""
OUT_PATTERN="logs/icml/mesogip/wave_${WAVE_TAG}_%j"
if [ "$ARRAY_CONC" -gt 0 ]; then
    LAST=$((COUNT - 1))
    ARRAY_HEADER="#SBATCH --array=0-${LAST}%${ARRAY_CONC}"
    OUT_PATTERN="logs/icml/mesogip/wave_${WAVE_TAG}_%A_%a"
fi

cat > "$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=pact-${WAVE_TAG}
#SBATCH --partition=${PARTITION}
${ARRAY_HEADER}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=${WALLTIME}
#SBATCH --signal=B:USR1@120
#SBATCH --output=${OUT_PATTERN}.out
#SBATCH --error=${OUT_PATTERN}.err

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

trap 'echo "[WARN] Approaching walltime limit (SIGUSR1) at \$(date -u). Completed tasks are saved and resumable."' USR1

"\$PYTHON" - "$TASK_JSON" "$DEADLINE_SEC" <<'PYRUN'
import json, os, socket, subprocess, sys, time
from pathlib import Path
import torch

tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
deadline_sec = int(sys.argv[2])
array_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
if array_idx is not None:
    indices = [int(array_idx)]
else:
    indices = list(range(len(tasks)))

job_start = time.time()
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
done_now = 0
failed_now = 0

for idx in indices:
    task = tasks[idx]
    job_id = task["id"]
    outputs = task["outputs"]
    if all(Path(o).exists() for o in outputs):
        print(f"[SKIP] {job_id}: all outputs already exist")
        continue
    if array_idx is None and (time.time() - job_start) >= deadline_sec:
        print(f"[STOP-SOFT] Reached soft walltime budget ({deadline_sec}s) before {job_id}; safe to resubmit.")
        break

    t0 = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"=== START [{idx+1}/{len(tasks)}] {job_id} | host={socket.gethostname()} | gpu={gpu_name} | commit={git_commit[:8]} | {started_utc} ===")
    rc = subprocess.run(["bash", "-c", task["command"]], cwd=os.getcwd()).returncode
    elapsed = round(time.time() - t0, 1)
    ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    all_exist = all(Path(o).exists() for o in outputs)
    status = "done" if (rc == 0 and all_exist) else "failed"

    prov = {
        "job_id": job_id,
        "group": task["group"],
        "kind": "offline",
        "execution_site": "mesogip",
        "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": idx if array_idx is not None else None,
        "gpu_model": gpu_name,
        "git_commit": git_commit,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "wall_seconds": elapsed,
        "laptop_est_hours": task["laptop_hours"],
        "returncode": rc,
        "outputs": outputs,
        "outputs_present": all_exist,
        "status": status,
    }
    safe = job_id.replace("/", "__")
    Path(f"logs/icml/mesogip_provenance/{safe}.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    print(f"=== END {job_id} | status={status} | wall={elapsed}s ===")
    if status == "done":
        done_now += 1
    else:
        failed_now += 1

print(f"=== WAVE SUMMARY: done_now={done_now}, failed_now={failed_now}, elapsed={round(time.time()-job_start,1)}s ===")
raise SystemExit(1 if failed_now else 0)
PYRUN

rm -rf "\$TORCH_HOME"
EOF

chmod +x "$SBATCH_FILE"
echo "Generated $SBATCH_FILE ($COUNT jobs, mode=$([ "$ARRAY_CONC" -gt 0 ] && echo "array%$ARRAY_CONC" || echo "packed"), partition=$PARTITION, walltime=$WALLTIME)"

if [ "$DO_SUBMIT" -eq 1 ]; then
    if [ -n "$AFTER_JOB" ]; then
        sbatch --dependency="afterany:${AFTER_JOB}" "$SBATCH_FILE"
    else
        sbatch "$SBATCH_FILE"
    fi
fi
