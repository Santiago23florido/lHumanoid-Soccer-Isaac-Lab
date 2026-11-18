#!/usr/bin/env bash
# Train the contract trackers of the new bodies, one Isaac process at a time, with the
# adopted trackers' conditions (W20: mirror loss, procedural share 0.5, the cross-fitted
# base-task set, the G1 walk contract, loads scored in log space).
#
# Usage: bash transfer/scripts/pact_train_trackers.sh "h1 cassie" [iterations=3000] [seed=41]
# Run it from a snapshot copy (PACT_ROOT=<repo> bash <copy>), so the repository's script can
# be edited while it runs. Status: logs/pact/train_trackers.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
bodies="${1:-h1 cassie}"
iterations="${2:-3000}"
seed="${3:-41}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/train_trackers.status
mkdir -p logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }
declare -A task_id=([h1]=H1Track-v0 [cassie]=CassieTrack-v0)
declare -A experiment=([h1]=h1_track [cassie]=cassie_track)

note "start bodies=$bodies iterations=$iterations seed=$seed"
for body in $bodies; do
  name="pact_${body}_s${seed}"
  if [ ! -f "transfer/bodies/$body/mirror.json" ]; then
    note "$name SKIPPED: no transfer/bodies/$body/mirror.json"
    continue
  fi
  export HUMANOID_SYMMETRY=loss HUMANOID_PROCEDURAL=0.5 HUMANOID_LOAD_SPACE=log
  export HUMANOID_TASKS="$root/transfer/results/base_tasks_xf.json"
  export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
  unset HUMANOID_POSTURE_CONTRACT HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  note "$name training"
  "$python" -u scripts/train.py --task "${task_id[$body]}" --headless --num_envs 4096 \
    --max_iterations "$iterations" --seed "$seed" --run_name "$name" \
    > "logs/pact/train_${name}.log" 2>&1
  code=$?
  run="$(ls -d logs/rsl_rl/${experiment[$body]}/*_"$name" 2>/dev/null | tail -1)"
  if [ $code -ne 0 ] || [ -z "$run" ]; then
    note "$name FAILED (exit $code), see logs/pact/train_${name}.log"
    continue
  fi
  note "$name trained: $run"
done
note "done"
