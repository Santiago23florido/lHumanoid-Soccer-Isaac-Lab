#!/usr/bin/env bash
# What a body costs from scratch: the probe competence of its tracker's checkpoints along
# training (the PACT trackers save one every 100 iterations). Each checkpoint follows the
# 71 gait probes (3 envs each) exactly as the co-execution map stage does
# (pact_coexecute.sh), scored by evaluate_competence.py; the capture is deleted once
# scored. The final checkpoint's score is the map stage's (icml_competence_<body>.json).
#
#   PACT_ROOT=<repo> bash <snapshot copy> ["h1 cassie"] ["100 200 ..."] [file to wait for]...
#
# Scores: transfer/results/icml_curve_<body>_<iteration>.json;
# status: logs/pact/tracker_curve.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
bodies="${1:-h1 cassie}"
iterations="${2:-100 200 300 500 700 1000 1500 2000 2500}"
shift $(($# < 2 ? $# : 2))
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/tracker_curve.status
out=logs/pact/curve
mkdir -p "$out"
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

for wait_for in "$@"; do
  note "waiting for $wait_for"
  until [ -f "$wait_for" ]; do sleep 60; done
done

assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}

note "start bodies=$bodies iterations=$iterations"
for body in $bodies; do
  run="$(ls -d "logs/rsl_rl/${body}_track"/*_"pact_${body}_s41" 2>/dev/null | tail -1)"
  if [ -z "$run" ]; then note "$body: no tracker run, skipped"; continue; fi
  task="$(echo "${body:0:1}" | tr a-z A-Z)${body:1}Track-Play-v0"
  unset HUMANOID_POSTURE_CONTRACT HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL
  export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
  for it in $iterations; do
    result="transfer/results/icml_curve_${body}_${it}.json"
    file="$out/${body}_${it}.npz"
    log="logs/pact/curve_${body}_${it}.log"
    if [ -f "$result" ]; then note "$body $it: scored already"; continue; fi
    if [ ! -f "$run/model_${it}.pt" ]; then note "$body $it: no checkpoint"; continue; fi
    "$python" transfer/scripts/export_actor.py "$run/model_${it}.pt" > /dev/null \
      || { note "$body $it: export FAILED"; continue; }
    for attempt in 1 2 3; do
      until assets_online; do sleep 60; done
      "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task" \
        --driver-policy "$run/exported/model_${it}.pt" --steps 900 --settle-steps 100 \
        --contract-from-truth --reference autonomous --num-envs 213 \
        --gait-probes transfer/probes/gait_probes.json --output "$file" > "$log" 2>&1
      grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || break
      note "$body $it: an asset could not be opened (attempt $attempt); retrying"
      rm -f "$file"
    done
    if [ ! -f "$file" ]; then note "$body $it: capture FAILED, see $log"; continue; fi
    "$python" transfer/scripts/evaluate_competence.py --capture "$file" --output "$result" \
      > "logs/pact/competence_curve_${body}_${it}.log" 2>&1 \
      && note "$body $it: $(tail -1 "$log" | cut -c1-160)" \
      || note "$body $it: scoring FAILED"
    rm -f "$file"
  done
done
note "done"
