#!/usr/bin/env bash
# Co-executions for PACT: every body's contract tracker follows the same fixed gaits of the
# procedural family (transfer/probes/gait_probes.json, 71 probes, 3 envs each), recorded with
# the joints and the tracker's observation (capture_contract.py --joints --policy-obs). The
# probes are dimensionless, so (probe, biped phase) pairs states across bodies by construction.
# The same captures score each tracker's competence (evaluate_competence.py); the new bodies'
# trackers are also scored on the random schedules of training (l1) and the family's corners.
#
#   PACT_ROOT=<repo> bash <snapshot copy> ["h1 cassie g1 nao"] [status file to wait for "done"]
#
# Trackers are deployed as trained (not symmetrised). Every stage is skipped when its output
# exists. Captures: logs/pact/coexec/<body>_<stage>.npz; scores:
# transfer/results/icml_competence_<body>[_<stage>].json; status: logs/pact/coexecute.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
bodies="${1:-h1 cassie g1 nao}"
wait_for="${2:-}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/coexecute.status
out=logs/pact/coexec
mkdir -p "$out" logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

if [ -n "$wait_for" ]; then
  note "waiting for done in $wait_for"
  until grep -q "^done" "$wait_for" 2>/dev/null; do sleep 60; done
fi

latest_run() { ls -d "$1"/*_"$2" 2>/dev/null | tail -1; }
latest_checkpoint() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}

# Isaac Sim opens the robots' and the scene's USD from the Omniverse asset server at every launch.
assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}

capture() {  # capture <body> <stage> <task id> <policy> args...
  local body="$1" stage="$2" task="$3" policy="$4" attempt log file
  shift 4
  file="$out/${body}_${stage}.npz"
  log="logs/pact/coexec_${body}_${stage}.log"
  if [ -f "$file" ]; then note "$body $stage: capture exists"; return 0; fi
  for attempt in 1 2 3; do
    until assets_online; do sleep 60; done
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task" \
      --driver-policy "$policy" --steps 900 --settle-steps 100 --contract-from-truth \
      --reference autonomous "$@" --output "$file" > "$log" 2>&1
    grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || break
    note "$body $stage: an asset could not be opened (attempt $attempt); retrying"
    rm -f "$file"
  done
  if [ ! -f "$file" ]; then note "$body $stage: capture FAILED, see $log"; return 1; fi
  note "$body $stage: $(tail -1 "$log" | cut -c1-200)"
}

score() {  # score <body> <stage> <suffix>
  local body="$1" stage="$2" suffix="$3" result
  result="transfer/results/icml_competence_${body}${suffix}.json"
  [ -f "$out/${body}_${stage}.npz" ] || return 0
  "$python" transfer/scripts/evaluate_competence.py --capture "$out/${body}_${stage}.npz" \
    --output "$result" > "logs/pact/competence_${body}_${stage}.log" 2>&1 \
    && note "$body $stage scored: $result" || note "$body $stage: scoring FAILED"
}

note "start bodies=$bodies"
for body in $bodies; do
  unset HUMANOID_POSTURE_CONTRACT HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL
  case "$body" in
    g1)
      run=logs/rsl_rl/g1_track/2026-11-08_18-22-20_xs_g1_s41 checkpoint=model_7193
      task=G1Track-Play-v0
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      ;;
    nao)
      run=logs/rsl_rl/nao_track/2026-11-08_21-02-51_xs_nao_s31 checkpoint=model_4497
      task=NaoTrack-Play-v0
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
      export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
      ;;
    h1 | cassie)
      run="$(latest_run "logs/rsl_rl/${body}_track" "pact_${body}_s41")"
      if [ -z "$run" ]; then note "$body: no tracker run, skipped"; continue; fi
      checkpoint="$(latest_checkpoint "$run")"
      task="$(echo "${body:0:1}" | tr a-z A-Z)${body:1}Track-Play-v0"
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      ;;
    *) note "$body: unknown body, skipped"; continue ;;
  esac
  "$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt" > /dev/null \
    || { note "$body: export FAILED ($run/$checkpoint.pt)"; continue; }
  policy="$run/exported/$checkpoint.pt"
  note "$body: $policy on $task"
  capture "$body" map "$task" "$policy" --num-envs 213 --joints --policy-obs \
    --gait-probes transfer/probes/gait_probes.json && score "$body" map ""
  if [ "$body" = h1 ] || [ "$body" = cassie ]; then
    capture "$body" random "$task" "$policy" --num-envs 128 --procedural-fraction 1 \
      && score "$body" random _random
    capture "$body" corners "$task" "$policy" --num-envs 108 \
      --gait-probes transfer/probes/corner_probes.json && score "$body" corners _corners
  fi
done
note "done"
