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
# PACT_DAGGER_ARM: the shared arm trained between rounds (pact_gate_d.py; default pact_d, or a
# baseline or ablation: urma_d, hpt_d, pact_na, pact_nb), from a start model of its kind.
# PACT_DAGGER_CAPTURES: the base captures (pact_gate_d.py --captures; default the co-executions'
# map); PACT_DAGGER_MIN_DUTY: drop probes asking a lower mean duty (0.55: walking only);
# PACT_DAGGER_SEED: the trainer's seed; PACT_DAGGER_DIR / _RESULTS / _MODELS: where captures,
# scores and models go (default logs/pact/dagger, transfer/results, logs/pact/models);
# PACT_DAGGER_ENVS / _CAPTURE_STEPS / _SETTLE: capture size (213 = 71 probes x 3, 900, 100; a
# smoke run shrinks them). Every body's labeller is its final teacher
# (transfer/bodies/teachers.json, body_conditions.py); training-task rounds draw the procedural
# family on HUMANOID_PROCEDURAL_DUTY (default 0.55,0.95: the walking family).
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
model="$1"
rounds="$2"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
bodies="${3:-$("$python" transfer/scripts/body_conditions.py --bodies)}"
tag="${4:-main}"
shift $(($# < 4 ? $# : 4))
train_steps="${PACT_DAGGER_STEPS:-8000}"
first="${PACT_DAGGER_FIRST:-1}"
last=$((first + rounds - 1))
where="${PACT_DAGGER_ENV:-probes}"
share="${PACT_DAGGER_TEACHER_SHARE:-0}"
# (A brace inside ${...:-...} would end the expansion early: the default is set apart.)
captures_root="${PACT_DAGGER_DIR:-logs/pact/dagger}"
results="${PACT_DAGGER_RESULTS:-transfer/results}"
models="${PACT_DAGGER_MODELS:-logs/pact/models}"
envs="${PACT_DAGGER_ENVS:-213}"
capture_steps="${PACT_DAGGER_CAPTURE_STEPS:-900}"
settle="${PACT_DAGGER_SETTLE:-100}"
this_tag="$captures_root/$tag/r*/{body}.npz"
aggregate="${PACT_DAGGER_GLOB:-$this_tag}"
train_last="${PACT_DAGGER_TRAIN_LAST:-0}"
arm="${PACT_DAGGER_ARM:-pact_d}"
# The base data the aggregate starts from, the probes kept (asked mean duty) and the seed.
base_captures="logs/pact/coexec/{body}_map.npz"
captures="${PACT_DAGGER_CAPTURES:-$base_captures}"
min_duty="${PACT_DAGGER_MIN_DUTY:-0}"
seed="${PACT_DAGGER_SEED:-0}"
duty="${HUMANOID_PROCEDURAL_DUTY:-0.55,0.95}"
status="logs/pact/dagger_${tag}.status"
mkdir -p logs/pact "$results" "$models"
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

setup() {  # setup <body>: the body's final teacher (the labeller), task and conditions
  unset HUMANOID_OBSERVE_UNITS HUMANOID_TASKS HUMANOID_PROCEDURAL HUMANOID_PROCEDURAL_DUTY
  local vars
  vars="$("$python" transfer/scripts/body_conditions.py "$1" --check 2>&1)" \
    || { note "$1: $vars; skipped"; return 1; }
  eval "$vars"  # transfer/bodies/teachers.json: run, checkpoint, policy, play_task, conditions
  task="$play_task"
}

note "start model=$model rounds=$first..$last bodies=$bodies steps=$train_steps env=$where share=$share"
for round in $(seq "$first" "$last"); do
  out="$captures_root/$tag/r$round"
  mkdir -p "$out"
  for body in $bodies; do
    setup "$body" || continue
    file="$out/$body.npz"
    log="logs/pact/dagger_${tag}_r${round}_${body}.log"
    result="$results/icml_gate_d1_${tag}_r${round}_${body}.json"
    if [ ! -f "$file" ]; then
      for attempt in 1 2 3; do
        until assets_online; do sleep 60; done
        if [ "$where" = train ]; then
          (
            export HUMANOID_PROCEDURAL=0.5 HUMANOID_TASKS="$root/transfer/results/base_tasks_xf.json"
            export HUMANOID_PROCEDURAL_DUTY="$duty"
            "$python" -u transfer/scripts/capture_contract.py --headless \
              --task-id "${task%-Play-v0}-v0" --driver-policy "$policy" --pact-model "$model" \
              --pact-body "$body" --teacher-share "$share" --steps "$capture_steps" --settle-steps "$settle" \
              --contract-from-truth --reference autonomous --num-envs "$envs" --joints --policy-obs \
              --output "$file" > "$log" 2>&1
          )
        else
          "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task" \
            --driver-policy "$policy" --pact-model "$model" --pact-body "$body" \
            --teacher-share "$share" --steps "$capture_steps" --settle-steps "$settle" --contract-from-truth \
            --reference autonomous --num-envs "$envs" --gait-probes transfer/probes/gait_probes.json \
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
    next="$models/d1_${tag}_r${round}.pt"
    "$python" -u transfer/scripts/pact_gate_d.py --bodies "$(echo $bodies | tr ' ' ',')" \
      --captures "$captures" --min-duty "$min_duty" --seed "$seed" \
      --dagger "$aggregate" --init "$model" --steps "$train_steps" \
      --arms "hold,$arm" --save "$next" \
      --output "$results/icml_gate_d1_${tag}_train_r${round}.json" \
      > "logs/pact/dagger_${tag}_train_r${round}.log" 2>&1 \
      && { model="$next"; note "trained $next ($arm): $(grep -a -e 'pact_d R' -e '^saved' "logs/pact/dagger_${tag}_train_r${round}.log" | tail -1 | cut -c1-200)"; } \
      || { note "training r$round FAILED"; break; }
  fi
done
note "done"
