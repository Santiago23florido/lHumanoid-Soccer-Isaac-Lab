#!/usr/bin/env bash
# W13 of skill_space.md §10.14: each body's decoder trained once on the base tasks, step 3 of
# method_revision.md §6. S's recipe exactly (paper_queue.sh: each body's incumbent continued
# 1000 iterations in its own conditions, one run per seed), with the base tasks' plans added
# (HUMANOID_TASKS, transfer/results/base_tasks.json) and the family's share at 0.5. S's
# trackers, the same continuation without the tasks, are the paired control.
#
#   1. task set   base_task_set.py, after W12 (rough is refused there)
#   2. smoke      3 iterations per body: no error, and a task_<name> metric per task in the log
#   3. train      G1 seeds 41 42 43 (mt_g1_s<seed>), NAO 31 32 33 (mt_nao_s<seed>)
#   4. evaluate   each W13 tracker: encoder, walk, flat L4 (as S); every tracker of both arms:
#                 each new base task's L4 at its middle command (ZS_COMMANDS)
#   5. compare    w13_compare.py -> transfer/results/w13_decoders.json
# Rough is never trained on or evaluated here. Room, per the project's rule: one Isaac process.
# Every stage is skipped when its output exists. It always ends with "w13 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w13_decoders.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results
tasks_file=$results/base_tasks.json

g1_incumbent=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
g1_checkpoint=model_6194
nao_incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
nao_checkpoint=model_3498
declare -A middle=([backward]="mid -0.35 0 0" [sidestep]="left 0 0.25 0" [jog]="mid 1.25 0 0")

rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
conditions() {  # <body>: S's conditions (paper_queue.sh), then the base tasks
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
  export HUMANOID_PROCEDURAL=0.5 HUMANOID_TASKS="$root/$tasks_file"
  unset HUMANOID_TASK_WEIGHTS
}
task_of() { [ "$1" = g1 ] && echo G1Track-v0 || echo NaoTrack-v0; }
new_tasks() {
  "$python" -c "
import json, sys
print(' '.join(t['name'] for t in json.load(open(sys.argv[1]))['tasks'] if t['name'] != 'flat'))" \
    "$tasks_file"
}
ladder() {  # <body> <run dir> <tag> <task> <stages...>; deployed as S is (symmetrized)
  local body="$1" run="$2" tag="$3" task="$4"
  shift 4
  local generator=flat_dagger2 commands=""
  if [ "$task" != flat ]; then generator="base_${task}_b4" commands="${middle[$task]}"; task="base_${task}"; fi
  # S was evaluated under its training conditions (paper_queue.sh); so is every arm here.
  ( conditions "$body"
    unset HUMANOID_TASKS
    export HUMANOID_PROCEDURAL=0.7
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 \
    ZS_TASK_ID="$( [ "$body" = g1 ] && echo G1Track-Play-v0 || echo NaoTrack-Play-v0 )" \
    ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" ZS_COMMANDS="$commands" \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$(latest "$run")" "$tag" "$@" ) \
    > "logs/contract/ladder_${tag}_${task}.log" 2>&1 || note "$tag $task ladder FAILED"
  note "evaluated $tag $task"
}

until grep -q "^w12 done" logs/contract/w12_generators.status 2>/dev/null; do sleep 120; done
note "w13 start"
"$python" transfer/scripts/base_task_set.py --output "$tasks_file" > logs/contract/w13_tasks.out 2>&1 \
  || { note "task set FAILED"; note "w13 done"; exit 1; }
note "task set: $(grep -v excluded logs/contract/w13_tasks.out | awk '{print $1}' | tr '\n' ' ')"

for body in g1 nao; do
  incumbent=$g1_incumbent checkpoint=$g1_checkpoint
  [ "$body" = nao ] && incumbent=$nao_incumbent checkpoint=$nao_checkpoint
  smoke="mt_${body}_smoke"
  ls -d logs/rsl_rl/"${body}"_track/*_"$smoke" > /dev/null 2>&1 && continue
  ( conditions "$body"
    "$python" -u scripts/train.py --task "$(task_of "$body")" --headless --num_envs 64 \
      --max_iterations 3 --seed 1 --run_name "$smoke" --resume --load_run "$(basename "$incumbent")" \
      --checkpoint "$checkpoint.pt" > "logs/contract/train_${smoke}.log" 2>&1 )
  missing=""
  for name in contract $(grep -v excluded logs/contract/w13_tasks.out | awk '{print $1}'); do
    grep -q "task_${name}" "logs/contract/train_${smoke}.log" || missing="$missing $name"
  done
  if grep -q "Traceback" "logs/contract/train_${smoke}.log" || [ -n "$missing" ]; then
    note "smoke $body FAILED (missing:${missing:- none})"; note "w13 stopped"; note "w13 done"; exit 1
  fi
  note "smoke $body: every task logged"
done

for body in g1 nao; do
  incumbent=$g1_incumbent checkpoint=$g1_checkpoint seeds="41 42 43"
  [ "$body" = nao ] && incumbent=$nao_incumbent checkpoint=$nao_checkpoint seeds="31 32 33"
  for seed in $seeds; do
    name="mt_${body}_s${seed}"
    if ! ls -d logs/rsl_rl/"${body}"_track/*_"$name" > /dev/null 2>&1; then
      note "$name training start"
      ( conditions "$body"
        "$python" -u scripts/train.py --task "$(task_of "$body")" --headless --num_envs 4096 \
          --max_iterations 1000 --seed "$seed" --run_name "$name" --resume \
          --load_run "$(basename "$incumbent")" --checkpoint "$checkpoint.pt" \
          "agent.algorithm.learning_rate=$(rate "$incumbent/$checkpoint.pt")" \
          > "logs/contract/train_${name}.log" 2>&1 ) || { note "$name training FAILED"; continue; }
      note "$name trained"
    fi
    run="$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" | tail -1)"
    [ -f "$results/zero_shot_${body}_${name}_sym_flat_dagger2_auto_kin_l4_turning.json" ] \
      || ladder "$body" "$run" "${name}_sym" flat encoder walk l4
  done
done

# Each new base task at its middle command, on both arms (S's trackers are the control).
for task in $(new_tasks); do
  read -r cname _ <<< "${middle[$task]}"
  for pair in "g1 41 warm_r3_logload warm_r3_logload_sym" "g1 42 warm_r3_s42_log warm_r3_s42_log_sym" \
    "g1 43 warm_r3_s43_log warm_r3_s43_log_sym" "nao 31 zs_nao_r1 r1_3498_sym_v3" \
    "nao 32 warm_nao_r1_s32 warm_nao_r1_s32_sym" "nao 33 warm_nao_r1_s33 warm_nao_r1_s33_sym"; do
    read -r body seed s_run s_tag <<< "$pair"
    for arm in mt s; do
      if [ "$arm" = mt ]; then
        run="$(ls -d logs/rsl_rl/"${body}"_track/*_mt_"${body}"_s"${seed}" 2>/dev/null | tail -1)"
        tag="mt_${body}_s${seed}_sym"
      else
        run="$(ls -d logs/rsl_rl/"${body}"_track/*_"${s_run}" 2>/dev/null | tail -1)" tag="$s_tag"
      fi
      [ -n "$run" ] || continue
      out="$results/zero_shot_${body}_${tag}_base_${task}_b4_auto_kin_l4_${cname}.json"
      [ -f "$out" ] || ladder "$body" "$run" "$tag" "$task" l4
    done
  done
done

"$python" transfer/scripts/w13_compare.py --output "$results/w13_decoders.json" \
  > logs/contract/w13_compare.out 2>&1 || note "w13 compare FAILED"
note "w13 decision: $(tail -1 logs/contract/w13_compare.out)"
note "w13 done"
