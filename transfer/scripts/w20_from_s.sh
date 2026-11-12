#!/usr/bin/env bash
# W20: W13's decoder recipe with W19's cross-fitted plans, trained once
# from S. W13's trackers (the same incumbent, seed, conditions and length, without the
# cross-fitted plans) are the paired control; their rows are already on disk.
#
#   1. train      S's incumbent continued 1000 iterations in W13's conditions, with
#                 base_tasks_xf.json instead of base_tasks.json: xs_<body>_s<seed>
#   2. evaluate   per tracker, on W13's paths:
#                 - the encoder refitted on its own states, and S0's walk;
#                 - flat's L4 and jog's at its middle command (the G1 in the task's units as W13,
#                   the NAO in its own as W14);
#                 - sidestep's L4 under mt18_no_sidestep and _m, and rough's under rough_b4
#                   (own units, as W16-W19)
#   3. screen     seeds 41 (G1) and 31 (NAO) first; the others run if, on either body, W13's
#                 tracker falls more than twice on sidestep under mt18_no_sidestep and xs's at
#                 most half as often
#   4. full       seeds 42 43 (G1) and 32 33 (NAO)
#   5. compare    w20_compare.py -> transfer/results/w20_from_s.json
#
# Sidestep and rough stay out of every gradient step. One Isaac process at a time; every stage
# is skipped when its output exists. It always ends with "w20 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w20_from_s.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results
ITERATIONS=1000
tasks_file="$root/$results/base_tasks_xf.json"

# S's incumbents and checkpoints, as w13_decoders.sh
declare -A incumbent=(
  [g1]=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
  [nao]=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
)
declare -A checkpoint=([g1]=model_6194 [nao]=model_3498)
declare -A walk=([g1]=logs/contract/walk_dagger3e/generator.pt [nao]=logs/contract/walk_dagger2e/generator.pt)
declare -A task_id=([g1]=G1Track-v0 [nao]=NaoTrack-v0)
declare -A play_id=([g1]=G1Track-Play-v0 [nao]=NaoTrack-Play-v0)
# The units each body's base-task rows are read in (paper_tables.OBSERVED)
declare -A observe=([g1]=task [nao]=own)
declare -A own=([g1]="" [nao]=_own)
sidestep="left 0 0.25 0
left_fast 0 0.45 0
right 0 -0.35 0"

rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
isaac_busy() {  # no brackets or backslashes in the pattern (bash -> PowerShell quoting)
  local busy
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [ "$busy" != 0 ]
}
w19_done() {  # in the current run only
  awk '/^w19 start/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
    logs/contract/w19_crossfit.status 2>/dev/null | grep -q "^w19 done"
}
walk_contract() {  # <body>: the walk contract and its load space, as W13
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/${walk[g1]}" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/${walk[nao]}" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}
training_conditions() {  # <body>: W13's, with the cross-fitted task set
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.5
  unset HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  walk_contract "$1"
  export HUMANOID_TASKS="$tasks_file"
}
conditions() {  # <body>: the evaluation conditions of W13-W19
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7
  unset HUMANOID_TASKS HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  walk_contract "$1"
}

# <body> <run> <tag> <observe: task|own> <zs task> <generator> <commands> <last> <stages...>
ladder() {
  local body="$1" run="$2" tag="$3" units="$4" task="$5" generator="$6" commands="$7" last="$8"
  shift 8
  local suffix="" stages="$*"
  [ "$units" = own ] && suffix=_own
  if [ "$*" = l4 ]; then
    [ -f "$results/zero_shot_${body}_${tag}_${generator}${suffix}_auto_kin_l4_${last}.json" ] && return 0
  fi
  while isaac_busy; do sleep 60; done
  ( conditions "$body"
    [ "$units" = own ] && export ZS_OBSERVE=own
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID="${play_id[$body]}" \
    ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" ZS_COMMANDS="$commands" \
    ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$(latest "$run")" "$tag" "$@" ) \
    < /dev/null > "logs/contract/ladder_${tag}_${generator}${suffix}_${task}_${stages// /_}.log" 2>&1 \
    || { note "$tag $generator $task FAILED"; return 1; }
  note "$task $tag $generator$suffix $*"
}

# <body> <seed>: trains xs_<body>_s<seed> from S and evaluates it
tracker() {
  local body="$1" seed="$2" name="xs_${1}_s${2}" run tag
  if ! ls -d logs/rsl_rl/"${body}"_track/*_"$name" > /dev/null 2>&1; then
    while isaac_busy; do sleep 60; done
    note "$name training start"
    ( training_conditions "$body"
      "$python" -u scripts/train.py --task "${task_id[$body]}" --headless --num_envs 4096 \
        --max_iterations "$ITERATIONS" --seed "$seed" --run_name "$name" --resume \
        --load_run "$(basename "${incumbent[$body]}")" --checkpoint "${checkpoint[$body]}.pt" \
        "agent.algorithm.learning_rate=$(rate "${incumbent[$body]}/${checkpoint[$body]}.pt")" \
        < /dev/null > "logs/contract/train_${name}.log" 2>&1 ) || { note "$name training FAILED"; return 1; }
    note "$name trained"
  fi
  run="$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" | tail -1)"
  tag="${name}_sym"
  # The encoder refitted on this tracker's own states, and S0's walk (W13's stages).
  [ -f "$results/zero_shot_${body}_${tag}_walk_auto_kin.json" ] \
    || ladder "$body" "$run" "$tag" task flat flat_dagger2 "" turning encoder walk || return 1
  ladder "$body" "$run" "$tag" "${observe[$body]}" flat flat_dagger2 "" turning l4
  ladder "$body" "$run" "$tag" "${observe[$body]}" base_jog base_jog_b4 "mid 1.25 0 0" mid l4
  for generator in mt18_no_sidestep mt18_no_sidestep_m; do
    ladder "$body" "$run" "$tag" own base_sidestep "$generator" "$sidestep" right l4
  done
  ladder "$body" "$run" "$tag" own rough rough_b4 "" turning l4
}

until w19_done; do sleep 120; done
note "w20 start"
[ -f "$tasks_file" ] || { note "no base_tasks_xf.json FAILED"; note "w20 done"; exit 1; }

# 1-2. Screen seeds.
tracker g1 41
tracker nao 31
"$python" transfer/scripts/w20_compare.py --screen > logs/contract/w20_screen.out 2>&1
note "w20 screen: $(tail -1 logs/contract/w20_screen.out)"

# 3-4. The other seeds, if the screen's rule holds.
if grep -q "^continue" logs/contract/w20_screen.out; then
  for spec in "g1 42" "g1 43" "nao 32" "nao 33"; do
    # shellcheck disable=SC2086
    tracker $spec
  done
fi

# 5. Compare.
"$python" transfer/scripts/w20_compare.py --output "$results/w20_from_s.json" \
  > logs/contract/w20_compare.out 2>&1 || note "w20 compare FAILED"
note "w20 reading: $(tail -1 logs/contract/w20_compare.out)"
note "w20 done"
