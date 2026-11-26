#!/usr/bin/env bash
# G-N2: a held-out body added to the shared PACT policy by PPO, against its tracker's
# from-scratch cost (registered in the prereg before it runs).
#
# Per held-out body:
#   1. the shared policy trained without it (pact_gate_d.py --hold-out, on the map
#      captures and G-D1's DAgger captures of the other bodies);
#   2. that policy driving the body zero-shot, scored as G-D1 scores a round (iteration 0);
#   3. PPO on each arm's parameters in the body's Track task (pact_adapt.py);
#   4. the adapted checkpoints scored the same way.
#
#   PACT_ROOT=<repo> bash <snapshot copy> ["h1 cassie"] ["etune bitfit"] [iterations] [dagger tag]
#
# PACT_GN2_START: the start model ({body} is filled; default logs/pact/models/gn2_without_<body>.pt,
# trained here if missing), e.g. the manifest's logs/icml/models/loo_pact_d_{body}_s0.pt. The
# captures' labeller is the body's final teacher (transfer/bodies/teachers.json); PPO runs on the
# walking family (HUMANOID_PROCEDURAL_DUTY, default 0.55,0.95).
#
# Models: logs/pact/models/gn2_without_<body>.pt; runs: logs/rsl_rl/pact_adapt_<body>/;
# scores: transfer/results/icml_gate_n2_<body>_<arm>_<it>.json (arm "zero" at it 0);
# status: logs/pact/gn2.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
bodies="${1:-h1 cassie}"
arms="${2:-etune bitfit}"
iterations="${3:-1000}"
dagger="${4:-main}"
score_at="${PACT_GN2_SCORE_AT:-50 100 200 300 400 500 600 700 800 900 1000}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/gn2.status
mkdir -p logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}
latest_run() { ls -d "$1"/*_"$2" 2>/dev/null | tail -1; }

setup() {  # setup <body>: the body's final teacher (the captures' labeller) and conditions
  unset HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL HUMANOID_PROCEDURAL_DUTY \
    HUMANOID_SYMMETRY HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT
  local vars
  vars="$("$python" transfer/scripts/body_conditions.py "$1" --check 2>&1)" \
    || { note "$1: $vars; skipped"; return 1; }
  eval "$vars"  # policy, play_task, train_task, conditions (transfer/bodies/teachers.json)
}

score() {  # score <body> <model> <result>: the model drives the 71 probes, scored as G-D1
  local body="$1" model="$2" result="$3" file log
  file="logs/pact/gn2/$(basename "$result" .json).npz"
  log="logs/pact/gn2/$(basename "$result" .json).log"
  mkdir -p logs/pact/gn2
  [ -f "$result" ] && return 0
  for attempt in 1 2 3; do
    until assets_online; do sleep 60; done
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$play_task" \
      --driver-policy "$policy" --pact-model "$model" --pact-body "$body" \
      --steps 900 --settle-steps 100 --contract-from-truth --reference autonomous \
      --num-envs 213 --gait-probes transfer/probes/gait_probes.json --joints --policy-obs \
      --output "$file" > "$log" 2>&1
    grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || break
    note "$body: an asset could not be opened (attempt $attempt); retrying"
    rm -f "$file"
  done
  [ -f "$file" ] || { note "$body: capture FAILED for $model, see $log"; return 1; }
  "$python" transfer/scripts/evaluate_competence.py --capture "$file" --output "$result" \
    > "${log%.log}_score.log" 2>&1 || { note "$body: scoring FAILED for $model"; return 1; }
  note "scored $result: $(tail -1 "$log" | cut -c1-160)"
  rm -f "$file"  # the score keeps what G-N2 reads; captures are 1-2 GB each
}

note "start bodies=$bodies arms=$arms iterations=$iterations dagger=$dagger"
for body in $bodies; do
  setup "$body" || continue
  model="${PACT_GN2_START:-logs/pact/models/gn2_without_${body}.pt}"
  model="${model//\{body\}/$body}"
  if [ ! -f "$model" ]; then
    note "$body: training the shared policy without it"
    "$python" -u transfer/scripts/pact_gate_d.py --hold-out "$body" --arms hold,pact_d \
      --dagger "logs/pact/dagger/$dagger/r*/{body}.npz" --save "$model" \
      --output "transfer/results/icml_gate_n2_offline_${body}.json" \
      > "logs/pact/gn2_offline_${body}.log" 2>&1 \
      || { note "$body: training without it FAILED"; continue; }
  fi
  score "$body" "$model" "transfer/results/icml_gate_n2_${body}_zero_0.json"
  for arm in $arms; do
    tag="gn2_${body}_${arm}"
    adapted="$(latest_run "logs/rsl_rl/pact_adapt_${body}" "$tag")"
    if [ -z "$adapted" ] || ! ls "$adapted"/pact_*.pt > /dev/null 2>&1; then
      note "$tag: PPO, $iterations iterations"
      # The from-scratch trackers' training conditions (pact_train_trackers.sh), mirror loss
      # aside; the Play captures that score run without them, as the curve's did.
      (
        export HUMANOID_PROCEDURAL=0.5 HUMANOID_TASKS="$root/transfer/results/base_tasks_xf.json"
        export HUMANOID_PROCEDURAL_DUTY="${HUMANOID_PROCEDURAL_DUTY:-0.55,0.95}"
        "$python" -u transfer/scripts/pact_adapt.py --headless --task "$train_task" \
          --body "$body" --pact-model "$model" --arm "$arm" --max-iterations "$iterations" \
          --run-name "$tag" > "logs/pact/${tag}.log" 2>&1
      ) || { note "$tag: PPO FAILED"; continue; }
      adapted="$(latest_run "logs/rsl_rl/pact_adapt_${body}" "$tag")"
      note "$tag: $(grep -a 'trained in' "logs/pact/${tag}.log" | tail -1)"
    fi
    last="$(ls "$adapted" | grep -E '^pact_[0-9]+\.pt$' | sed 's/pact_//; s/\.pt//' | sort -n | tail -1)"
    for it in $score_at $last; do
      [ -f "$adapted/pact_${it}.pt" ] || continue
      score "$body" "$adapted/pact_${it}.pt" "transfer/results/icml_gate_n2_${body}_${arm}_${it}.json"
    done
  done
done
note "done"
