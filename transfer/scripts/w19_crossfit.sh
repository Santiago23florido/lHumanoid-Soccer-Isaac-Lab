#!/usr/bin/env bash
# W19: each decoder also practises plans that a generator composes for
# a base task it never saw (cross-fitted plans). Sidestep and rough stay out of every gradient step.
#
#   1. generators  mt19_xf_flat (walk, backward, jog experts) and mt19_xf_jog (walk, flat,
#                  backward), W17's recipe as W18's; the task set base_tasks_xf.json from them
#   2. signal      W13's G1 s41 tracker under them at flat's and jog's commands (instrumentation)
#   3. screen      W13's G1 s41 and NAO s31 trackers continued 300 iterations in W13's conditions:
#                  xf (base_tasks_xf.json) and ctl (base_tasks.json); each deployed with its W13
#                  tracker's refitted encoder; sidestep L4 under mt18_no_sidestep and _m
#   4. full        if the screen's rule holds: seeds 42 43 (G1) and 32 33 (NAO), the same
#   5. regression  rough L4 under rough_b4 on every new tracker
#   6. compare     w19_compare.py -> transfer/results/w19_crossfit.json
#
# One Isaac process at a time; every stage is skipped when its output exists. It always ends
# with "w19 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w19_crossfit.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results
ITERATIONS=300

declare -A w13=(
  [g1_41]=logs/rsl_rl/g1_track/2026-11-06_23-26-29_mt_g1_s41
  [g1_42]=logs/rsl_rl/g1_track/2026-11-07_01-30-35_mt_g1_s42
  [g1_43]=logs/rsl_rl/g1_track/2026-11-07_03-30-28_mt_g1_s43
  [nao_31]=logs/rsl_rl/nao_track/2026-11-07_05-30-28_mt_nao_s31
  [nao_32]=logs/rsl_rl/nao_track/2026-11-07_08-28-04_mt_nao_s32
  [nao_33]=logs/rsl_rl/nao_track/2026-11-07_11-24-19_mt_nao_s33
)
declare -A w13_checkpoint=([g1]=model_7193 [nao]=model_4497)
declare -A walk=([g1]=logs/contract/walk_dagger3e/generator.pt [nao]=logs/contract/walk_dagger2e/generator.pt)
declare -A expert=(
  [walk]=g1/rollouts/teacher_dart.npz
  [flat]=g1/rollouts/flat_dart.npz
  [backward]=g1/rollouts/base_backward_dart.npz
  [jog]=g1/rollouts/base_jog_dart.npz
)
recipe="--encoder-from logs/contract/walk_zs_g1_r3_6194_sym/generator.pt --sigma-from-body \
--warp-loads --solver midpoint --flow-steps 8 --horizon 20 --commit 13 --noise-conditional \
--gait g1/rollouts/teacher.npz --relabel-span 50"
sidestep="left 0 0.25 0
left_fast 0 0.45 0
right 0 -0.35 0"
flat_commands="forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0.2 0.3"
jog_commands="slow 1.0 0 0
mid 1.25 0 0
fast 1.5 0 0"

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
# <body>: W13's training conditions (w13_decoders.sh), without the task set
training_conditions() {
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.5
  unset HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/${walk[g1]}" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/${walk[nao]}" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}
# <body>: the evaluation conditions of W16-W18
conditions() {
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7
  unset HUMANOID_TASKS HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/${walk[g1]}" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/${walk[nao]}" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}

# <name> <held-out tasks...>: W18's recipe on every base expert but these (sidestep never)
generator() {
  local name="$1" data="" task
  shift
  [ -f "logs/contract/${name}/generator.pt" ] && return 0
  for task in walk flat backward jog; do
    [[ " $* " == *" $task "* ]] || data="$data --data ${expert[$task]}"
  done
  note "$name training start"
  # shellcheck disable=SC2086
  "$python" -u transfer/scripts/train_contract.py --name "$name" $data $recipe --epochs 200 \
    --results "$results/contract_${name}.json" > "logs/contract/train_${name}.log" 2>&1 \
    || { note "$name training FAILED"; return 1; }
  note "$name: $(grep -E '^gate|chain fidelity' "logs/contract/train_${name}.log" | tr '\n' ' ')"
}

# <body> <run> <checkpoint> <tag> <generator> <zs task> <commands> <last command>
ladder() {
  local body="$1" run="$2" checkpoint="$3" tag="$4" name="$5" task="$6" commands="$7" last="$8"
  [ -f "$results/zero_shot_${body}_${tag}_${name}_own_auto_kin_l4_${last}.json" ] && return 0
  while isaac_busy; do sleep 60; done
  ( conditions "$body"
    ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK="$task" \
    ZS_COMMANDS="$commands" ZS_GENERATOR="logs/contract/${name}/generator.pt" ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_${name}_${task}.log" 2>&1 \
    || { note "$tag $name $task FAILED"; return 1; }
  note "$task $tag $name"
}
rough() {  # <body> <run> <checkpoint> <tag>: rough_b4 on rough, pinned units, as W16
  local body="$1" run="$2" checkpoint="$3" tag="$4"
  [ -f "$results/zero_shot_${body}_${tag}_rough_b4_own_auto_kin_l4_turning.json" ] && return 0
  while isaac_busy; do sleep 60; done
  ( conditions "$body"
    ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK=rough \
    ZS_GENERATOR=logs/contract/rough_b4/generator.pt ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_rough_b4_rough.log" 2>&1 \
    || { note "$tag rough FAILED"; return 1; }
  note "rough $tag rough_b4"
}

# <body> <seed> <arm: xf|ctl>: W13's tracker continued; prints the run directory
continue_tracker() {
  local body="$1" seed="$2" arm="$3" name="${3}_${1}_s${2}" source tasks run
  source="${w13[${body}_${seed}]}"
  tasks="$root/$results/base_tasks.json"
  [ "$arm" = xf ] && tasks="$root/$results/base_tasks_xf.json"
  if ! ls -d logs/rsl_rl/"${body}"_track/*_"$name" > /dev/null 2>&1; then
    while isaac_busy; do sleep 60; done
    note "$name training start"
    ( training_conditions "$body"
      export HUMANOID_TASKS="$tasks"
      "$python" -u scripts/train.py --task "$([ "$body" = g1 ] && echo G1Track-v0 || echo NaoTrack-v0)" \
        --headless --num_envs 4096 --max_iterations "$ITERATIONS" --seed "$seed" --run_name "$name" \
        --resume --load_run "$(basename "$source")" --checkpoint "${w13_checkpoint[$body]}.pt" \
        "agent.algorithm.learning_rate=$(rate "$source/${w13_checkpoint[$body]}.pt")" \
        < /dev/null > "logs/contract/train_${name}.log" 2>&1 ) || { note "$name training FAILED"; return 1; }
    note "$name trained"
  fi
  run="$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" | tail -1)"
  # The deployed estimator stays W13's: the tracker's refitted encoder, copied, in both arms.
  local encoder="logs/contract/walk_zs_${body}_${name}_sym"
  [ -d "$encoder" ] || cp -r "logs/contract/walk_zs_${body}_mt_${body}_s${seed}_sym" "$encoder"
  echo "$run"
}

# <body> <seed>: both arms trained, then sidestep's L4 under both held-out generators
pair() {
  local body="$1" seed="$2" arm run
  for arm in xf ctl; do
    continue_tracker "$body" "$seed" "$arm" > /dev/null || return 1
  done
  for arm in xf ctl; do
    run="$(ls -d logs/rsl_rl/"${body}"_track/*_"${arm}_${body}_s${seed}" | tail -1)"
    for name in mt18_no_sidestep mt18_no_sidestep_m; do
      ladder "$body" "$run" "$(latest "$run")" "${arm}_${body}_s${seed}_sym" "$name" base_sidestep \
        "$sidestep" right
    done
  done
}

note "w19 start"

# 1. The cross-fitted generators and the task set.
generator mt19_xf_flat flat || { note "w19 done"; exit 1; }
generator mt19_xf_jog jog || { note "w19 done"; exit 1; }
"$python" transfer/scripts/crossfit_task_set.py --output "$results/base_tasks_xf.json" \
  > logs/contract/w19_tasks.out 2>&1 || { note "task set FAILED"; note "w19 done"; exit 1; }
note "task set: $(awk '{print $1}' logs/contract/w19_tasks.out | tr '\n' ' ')"

# 2. Signal: does W13's G1 s41 tracker fall under the cross-fitted plans of tasks it trained on?
ladder g1 "${w13[g1_41]}" model_7193 mt_g1_s41_sym mt19_xf_flat flat "$flat_commands" turning
ladder g1 "${w13[g1_41]}" model_7193 mt_g1_s41_sym mt19_xf_jog base_jog "$jog_commands" fast

# 3. Screen.
pair g1 41
pair nao 31
"$python" transfer/scripts/w19_compare.py --screen > logs/contract/w19_screen.out 2>&1
note "w19 screen: $(tail -1 logs/contract/w19_screen.out)"

# 4. The other seeds, if the screen's rule holds.
if grep -q "^continue" logs/contract/w19_screen.out; then
  for spec in "g1 42" "g1 43" "nao 32" "nao 33"; do
    # shellcheck disable=SC2086
    pair $spec
  done
fi

# 5. No regression on the published deployment.
for run in $(ls -d logs/rsl_rl/{g1,nao}_track/*_{xf,ctl}_{g1,nao}_s* 2>/dev/null); do
  name="${run##*_track/}"; name="${name#*_*_}"
  body=g1; [[ "$name" == *nao* ]] && body=nao
  rough "$body" "$run" "$(latest "$run")" "${name}_sym"
done

# 6. Compare.
"$python" transfer/scripts/w19_compare.py --output "$results/w19_crossfit.json" \
  > logs/contract/w19_compare.out 2>&1 || note "w19 compare FAILED"
note "w19 reading: $(tail -1 logs/contract/w19_compare.out)"
note "w19 done"
