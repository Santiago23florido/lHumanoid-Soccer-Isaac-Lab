#!/usr/bin/env bash
# Co-executions for PACT: every body's contract tracker follows the same fixed gaits of the
# procedural family (transfer/probes/gait_probes.json, 71 probes, 3 envs each), recorded with
# the joints and the tracker's observation (capture_contract.py --joints --policy-obs). The
# probes are dimensionless, so (probe, biped phase) pairs states across bodies by construction.
# The same captures score each tracker's competence (evaluate_competence.py). Then every body
# replays the same fixed plans of the base tasks (logs/pact/replays/base_tasks.npz, made by
# pact_make_replays.py; env i follows chain i, paired by (chain, replay_clock)), recorded the
# same way. Last, the new bodies' trackers are also scored on the random schedules of training
# (l1) and the family's corners.
#
#   PACT_ROOT=<repo> bash <snapshot copy> ["<bodies>"] [status file to wait for "done"]
#
# Each body is driven by its final teacher (transfer/bodies/teachers.json, symmetrised as
# registered; default every registered body; until 2026-11-22 the first trackers, as trained).
# PACT_COEXEC_STAGES picks the stages (default "map replay"; also "random corners"); the map
# stage links the teacher's audit probe capture, the same capture, when it exists. Every stage
# is skipped when its output exists. Captures: ${PACT_COEXEC_DIR:-logs/pact/coexec}/
# <body>_<stage>.npz; scores: transfer/results/icml_competence_<body>[_<stage>].json (with
# PACT_COEXEC_DIR set, <dir>/competence_<body>[_<stage>].json); status: logs/pact/coexecute.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
bodies="${1:-$("$python" transfer/scripts/body_conditions.py --bodies)}"
wait_for="${2:-}"
status=logs/pact/coexecute.status
out="${PACT_COEXEC_DIR:-logs/pact/coexec}"
stages="${PACT_COEXEC_STAGES:-map replay}"
want() { [[ " $stages " == *" $1 "* ]]; }
mkdir -p "$out" logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

if [ -n "$wait_for" ]; then
  note "waiting for done in $wait_for"
  until grep -q "^done" "$wait_for" 2>/dev/null; do sleep 60; done
fi

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
  [ -n "${PACT_COEXEC_DIR:-}" ] && result="$out/competence_${body}${suffix}.json"
  [ -f "$out/${body}_${stage}.npz" ] || return 0
  "$python" transfer/scripts/evaluate_competence.py --capture "$out/${body}_${stage}.npz" \
    --output "$result" > "logs/pact/competence_${body}_${stage}.log" 2>&1 \
    && note "$body $stage scored: $result" || note "$body $stage: scoring FAILED"
}

replay=logs/pact/replays/base_tasks.npz
replay_envs="$("$python" -c "import numpy, sys; print(numpy.load(sys.argv[1])['command'].shape[0])" "$replay" 2>/dev/null)"

setup() {  # setup <body>: the body's final teacher, task and conditions (teachers.json)
  unset HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL HUMANOID_PROCEDURAL_DUTY
  local vars
  vars="$("$python" transfer/scripts/body_conditions.py "$1" --check 2>&1)" \
    || { note "$1: $vars; skipped"; return 1; }
  eval "$vars"  # run, checkpoint, policy (symmetrised as registered), play_task, probes_capture
  task="$play_task"
}

note "start bodies=$bodies stages=$stages (replay: ${replay_envs:-none} chains)"
for body in $bodies; do
  setup "$body" || continue
  note "$body: $policy on $task"
  if want map; then
    if [ ! -f "$out/${body}_map.npz" ] && [ -f "$probes_capture" ]; then
      ln "$probes_capture" "$out/${body}_map.npz" && note "$body map: linked $probes_capture"
    fi
    capture "$body" map "$task" "$policy" --num-envs 213 --joints --policy-obs \
      --gait-probes transfer/probes/gait_probes.json && score "$body" map ""
  fi
  if want replay && [ -n "$replay_envs" ]; then
    capture "$body" replay "$task" "$policy" --num-envs "$replay_envs" --joints --policy-obs \
      --replay-plans "$replay" && score "$body" replay _replay
  fi
  want random && capture "$body" random "$task" "$policy" --num-envs 128 --procedural-fraction 1 \
    && score "$body" random _random
  want corners && capture "$body" corners "$task" "$policy" --num-envs 108 \
    --gait-probes transfer/probes/corner_probes.json && score "$body" corners _corners
done
note "done"
