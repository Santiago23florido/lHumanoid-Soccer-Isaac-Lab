#!/usr/bin/env bash
# G-D1: the shared PACT policy in closed loop, improved by DAgger (Ross et al. 2011).
#
# Each round, PACT drives every body through the 71 gait probes (3 envs each), exactly as
# the co-execution map stage ran each tracker (pact_coexecute.sh), while the body's tracker
# labels every visited state (capture_contract.py --pact-model). The capture is scored as a
# map stage is (evaluate_competence.py): that is the round's closed-loop result. The rounds'
# captures then join the map captures and the policy goes on training from the round's
# model (pact_gate_d.py --dagger --init). The last round is scored only.
#
#   PACT_ROOT=<repo> bash <snapshot copy> <start model> <rounds> ["g1 nao h1 cassie"] \
#       [tag] [file to wait for]...
#
# Captures: logs/pact/dagger/<tag>/r<k>/<body>.npz; scores:
# transfer/results/icml_gate_d1_<tag>_r<k>_<body>.json; models: logs/pact/models/d1_<tag>_r<k>.pt;
# status: logs/pact/dagger_<tag>.status. PACT_DAGGER_FIRST=<k> numbers the first round k (to
# score a model trained on rounds 1..k-1 as round k, e.g. the aggregate's final policy).
#
# PACT_DAGGER_ENV=train captures in the body's training task (<Body>Track-v0) with the plans
# its tracker trained on (procedural share 0.5, the base tasks, starts inside a plan, the
# contract_left termination) instead of the gait probes; those rounds are not scored by probe
# (the status line keeps the capture's falls and contract error).
# PACT_DAGGER_TEACHER_SHARE=<f>: the tracker drives that share of the environments.
# PACT_DAGGER_GLOB: the DAgger captures training aggregates (default: this tag's rounds).
# PACT_DAGGER_TRAIN_LAST=1 also trains after the last round.
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
model="$1"
rounds="$2"
bodies="${3:-g1 nao h1 cassie}"
tag="${4:-main}"
shift $(($# < 4 ? $# : 4))
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
train_steps="${PACT_DAGGER_STEPS:-8000}"
first="${PACT_DAGGER_FIRST:-1}"
last=$((first + rounds - 1))
where="${PACT_DAGGER_ENV:-probes}"
share="${PACT_DAGGER_TEACHER_SHARE:-0}"
aggregate="${PACT_DAGGER_GLOB:-logs/pact/dagger/$tag/r*/{body}.npz}"
train_last="${PACT_DAGGER_TRAIN_LAST:-0}"
status="logs/pact/dagger_${tag}.status"
mkdir -p logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

for wait_for in "$@"; do
  note "waiting for $wait_for"
  until [ -f "$wait_for" ]; do sleep 60; done
done

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

setup() {  # setup <body>: the tracker (teacher), task and contract the co-executions used
  unset HUMANOID_POSTURE_CONTRACT HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL
  case "$1" in
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
      run="$(latest_run "logs/rsl_rl/${1}_track" "pact_${1}_s41")"
      if [ -z "$run" ]; then note "$1: no tracker run, skipped"; return 1; fi
      checkpoint="$(latest_checkpoint "$run")"
      task="$(echo "${1:0:1}" | tr a-z A-Z)${1:1}Track-Play-v0"
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      ;;
    *) note "$1: unknown body, skipped"; return 1 ;;
  esac
  policy="$run/exported/$checkpoint.pt"
  [ -f "$policy" ] || "$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt" > /dev/null \
    || { note "$1: export FAILED ($run/$checkpoint.pt)"; return 1; }
}

note "start model=$model rounds=$first..$last bodies=$bodies steps=$train_steps env=$where share=$share"
for round in $(seq "$first" "$last"); do
  out="logs/pact/dagger/$tag/r$round"
  mkdir -p "$out"
  for body in $bodies; do
    setup "$body" || continue
    file="$out/$body.npz"
    log="logs/pact/dagger_${tag}_r${round}_${body}.log"
    result="transfer/results/icml_gate_d1_${tag}_r${round}_${body}.json"
    if [ ! -f "$file" ]; then
      for attempt in 1 2 3; do
        until assets_online; do sleep 60; done
        if [ "$where" = train ]; then
          (
            export HUMANOID_PROCEDURAL=0.5 HUMANOID_TASKS="$root/transfer/results/base_tasks_xf.json"
            "$python" -u transfer/scripts/capture_contract.py --headless \
              --task-id "${task%-Play-v0}-v0" --driver-policy "$policy" --pact-model "$model" \
              --pact-body "$body" --teacher-share "$share" --steps 900 --settle-steps 100 \
              --contract-from-truth --reference autonomous --num-envs 213 --joints --policy-obs \
              --output "$file" > "$log" 2>&1
          )
        else
          "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task" \
            --driver-policy "$policy" --pact-model "$model" --pact-body "$body" \
            --teacher-share "$share" --steps 900 --settle-steps 100 --contract-from-truth \
            --reference autonomous --num-envs 213 --gait-probes transfer/probes/gait_probes.json \
            --joints --policy-obs --output "$file" > "$log" 2>&1
        fi
        grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || break
        note "$body r$round: an asset could not be opened (attempt $attempt); retrying"
        rm -f "$file"
      done
    fi
    if [ ! -f "$file" ]; then note "$body r$round: capture FAILED, see $log"; continue; fi
    if [ "$where" = train ]; then
      note "$body r$round (train env): $(tail -1 "$log" | cut -c1-200)"
      continue
    fi
    "$python" transfer/scripts/evaluate_competence.py --capture "$file" --output "$result" \
      > "logs/pact/dagger_${tag}_r${round}_${body}_score.log" 2>&1 \
      && note "$body r$round: $(tail -1 "$log" | cut -c1-200)" \
      || note "$body r$round: scoring FAILED"
  done
  if [ "$round" -lt "$last" ] || [ "$train_last" = 1 ]; then
    next="logs/pact/models/d1_${tag}_r${round}.pt"
    "$python" -u transfer/scripts/pact_gate_d.py --bodies "$(echo $bodies | tr ' ' ',')" \
      --dagger "$aggregate" --init "$model" --steps "$train_steps" \
      --arms hold,pact_d --save "$next" \
      --output "transfer/results/icml_gate_d1_${tag}_train_r${round}.json" \
      > "logs/pact/dagger_${tag}_train_r${round}.log" 2>&1 \
      && { model="$next"; note "trained $next: $(grep -a 'pact_d R' "logs/pact/dagger_${tag}_train_r${round}.log" | cut -c1-200)"; } \
      || { note "training r$round FAILED"; break; }
  fi
done
note "done"
