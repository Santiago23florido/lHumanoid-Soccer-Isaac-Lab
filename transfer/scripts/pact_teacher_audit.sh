#!/usr/bin/env bash
# The teachers before any distillation: each body's tracker as trained ("plain") and its
# mirror-symmetrised version ("sym", symmetrize_actor.py), driving alone, on the tasks PACT
# is asked to learn and to be evaluated on.
#
#   PACT_ROOT=<repo> bash <snapshot copy> ["g1 nao h1 cassie"] ["sym"]
#
# Per body and variant:
#   probes  the 71 gait probes in <Body>Track-Play-v0 (as the co-executions' map stage),
#           scored by evaluate_competence.py -> transfer/results/icml_teacher_<body>_<v>_probes.json
#   train   the body's training task <Body>Track-v0 with its trackers' plans (procedural share
#           0.5, the base tasks), as pact_dagger.sh round 0 -> logs/pact/teachers/<body>_<v>_train.npz
# The plain variant's captures already exist (logs/pact/coexec/<body>_map.npz,
# logs/pact/dagger/train/r0/<body>.npz); pact_teacher_audit_read.py reads all of them, with
# W20's base-task and held-out ladders for the G1 and the NAO. Status: logs/pact/teachers.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
bodies="${1:-g1 nao h1 cassie}"
variants="${2:-sym}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/teachers.status
mkdir -p logs/pact/teachers
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

latest_run() { ls -d "$1"/*_"$2" 2>/dev/null | tail -1; }
latest_checkpoint() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}

setup() {  # setup <body> <variant>: the teacher and conditions of the co-executions
  unset HUMANOID_POSTURE_CONTRACT HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL
  case "$1" in
    g1)
      run=logs/rsl_rl/g1_track/2026-11-08_18-22-20_xs_g1_s41 checkpoint=model_7193 task=G1Track-Play-v0
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      ;;
    nao)
      run=logs/rsl_rl/nao_track/2026-11-08_21-02-51_xs_nao_s31 checkpoint=model_4497 task=NaoTrack-Play-v0
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
      export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
      ;;
    h1 | cassie)
      run="$(latest_run "logs/rsl_rl/${1}_track" "pact_${1}_s41")"
      checkpoint="$(latest_checkpoint "$run")"
      task="$(echo "${1:0:1}" | tr a-z A-Z)${1:1}Track-Play-v0"
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      ;;
    *) note "$1: unknown body"; return 1 ;;
  esac
  policy="$run/exported/$checkpoint.pt"
  [ "$2" = sym ] && policy="$run/exported/${checkpoint}_sym_pact.pt"
  [ -f "$policy" ] || { note "$1 $2: no policy $policy"; return 1; }
}

capture() {  # capture <log> <args...>: retried while the asset server is down
  local log="$1" attempt
  shift
  for attempt in 1 2 3; do
    until assets_online; do sleep 60; done
    "$python" -u transfer/scripts/capture_contract.py --headless "$@" > "$log" 2>&1
    grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || return 0
    note "asset could not be opened (attempt $attempt); retrying"
  done
}

note "start bodies=$bodies variants=$variants"
for body in $bodies; do
  for v in $variants; do
    setup "$body" "$v" || continue
    probes="logs/pact/teachers/${body}_${v}_probes.npz"
    result="transfer/results/icml_teacher_${body}_${v}_probes.json"
    if [ ! -f "$result" ]; then
      capture "logs/pact/teachers/${body}_${v}_probes.log" --task-id "$task" --driver-policy "$policy" \
        --steps 900 --settle-steps 100 --contract-from-truth --reference autonomous --num-envs 213 \
        --gait-probes transfer/probes/gait_probes.json --joints --policy-obs --output "$probes"
      "$python" transfer/scripts/evaluate_competence.py --capture "$probes" --output "$result" \
        > "logs/pact/teachers/${body}_${v}_probes_score.log" 2>&1 \
        && note "$body $v probes: $(tail -1 "logs/pact/teachers/${body}_${v}_probes.log" | cut -c1-160)" \
        || note "$body $v probes FAILED"
    fi
    train="logs/pact/teachers/${body}_${v}_train.npz"
    if [ ! -f "$train" ]; then
      (
        export HUMANOID_PROCEDURAL=0.5 HUMANOID_TASKS="$root/transfer/results/base_tasks_xf.json"
        capture "logs/pact/teachers/${body}_${v}_train.log" --task-id "${task%-Play-v0}-v0" \
          --driver-policy "$policy" --steps 900 --settle-steps 100 --contract-from-truth \
          --reference autonomous --num-envs 213 --joints --policy-obs --output "$train"
      )
      [ -f "$train" ] && note "$body $v train: $(tail -1 "logs/pact/teachers/${body}_${v}_train.log" | cut -c1-160)" \
        || note "$body $v train FAILED"
    fi
  done
done
note "done"
