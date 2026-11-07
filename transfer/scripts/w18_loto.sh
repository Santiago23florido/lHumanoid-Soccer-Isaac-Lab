#!/usr/bin/env bash
# W18 of skill_space.md §10.19: leave one base task out where no other task defines its commands
# (sidestep), on both bodies' frozen W13 decoders; and the same offline for the other base tasks.
#
#   1. reset       the state after a reset (no settle steps), G1 s41 and NAO s31, under mt17_c
#                  copied as w18_reset; then the proxy calibrated from it on rough's four arms
#   2. generators  mt18_no_sidestep (walk, flat, backward, jog), relabelled as W17; from it
#                  mt18_no_sidestep_m with the bodies' histories (W16's step-4 captures)
#   3. sidestep    L4 at left, left_fast and right on the six W13 trackers, pinned units, for
#                  mt18_no_sidestep, mt18_no_sidestep_m, mt17_c and mt17_m
#   4. offline     mt18_no_walk/_no_flat/_no_backward/_no_jog; every leave-one-out generator and
#                  mt17_c chained on its held-out task, from the teacher's history and the reset
#   5. compare     w18_compare.py -> transfer/results/w18_loto.json
#
# One Isaac process at a time; every stage is skipped when its output exists. It always ends
# with "w18 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w18_loto.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

declare -A runs=(
  [mt_g1_s41_sym]=logs/rsl_rl/g1_track/2026-11-06_23-26-29_mt_g1_s41
  [mt_g1_s42_sym]=logs/rsl_rl/g1_track/2026-11-07_01-30-35_mt_g1_s42
  [mt_g1_s43_sym]=logs/rsl_rl/g1_track/2026-11-07_03-30-28_mt_g1_s43
  [mt_nao_s31_sym]=logs/rsl_rl/nao_track/2026-11-07_05-30-28_mt_nao_s31
  [mt_nao_s32_sym]=logs/rsl_rl/nao_track/2026-11-07_08-28-04_mt_nao_s32
  [mt_nao_s33_sym]=logs/rsl_rl/nao_track/2026-11-07_11-24-19_mt_nao_s33
)
declare -A checkpoint=([g1]=model_7193 [nao]=model_4497)
declare -A walk=([g1]=logs/contract/walk_dagger3e/generator.pt [nao]=logs/contract/walk_dagger2e/generator.pt)
declare -A expert=(
  [walk]=g1/rollouts/teacher_dart.npz
  [flat]=g1/rollouts/flat_dart.npz
  [backward]=g1/rollouts/base_backward_dart.npz
  [sidestep]=g1/rollouts/base_sidestep_dart.npz
  [jog]=g1/rollouts/base_jog_dart.npz
)
# Each held-out task's teacher captures for the offline chains: <task> <rollout> <commands...>
declare -A chains=(
  [flat]="flat play forward fast turning"
  [backward]="base_backward train slow mid fast"
  [sidestep]="base_sidestep play left left_fast right"
  [jog]="base_jog play slow mid fast"
)
recipe="--encoder-from logs/contract/walk_zs_g1_r3_6194_sym/generator.pt --sigma-from-body \
--warp-loads --solver midpoint --flow-steps 8 --horizon 20 --commit 13 --noise-conditional \
--gait g1/rollouts/teacher.npz --relabel-span 50"
sidestep_commands="left 0 0.25 0
left_fast 0 0.45 0
right 0 -0.35 0"

body_of() { case "$1" in *g1*) echo g1 ;; *) echo nao ;; esac; }
conditions() {  # <body>: each body's evaluation conditions, as W16's and W17's
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
isaac_busy() {  # no brackets or backslashes in the pattern (bash -> PowerShell quoting)
  local busy
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [ "$busy" != 0 ]
}

# <name> <held-out task> <extra arguments...>: the experts of every base task but one
train() {
  local name="$1" out="$2" data="" task
  shift 2
  [ -f "logs/contract/${name}/generator.pt" ] && return 0
  for task in walk flat backward sidestep jog; do
    [ "$task" = "$out" ] || data="$data --data ${expert[$task]}"
  done
  note "$name training start"
  # shellcheck disable=SC2086
  "$python" -u transfer/scripts/train_contract.py --name "$name" $data $recipe "$@" \
    --results "$results/contract_${name}.json" > "logs/contract/train_${name}.log" 2>&1 \
    || { note "$name training FAILED"; return 1; }
  note "$name: $(grep -E '^\[leakage\]|^gate|chain fidelity' "logs/contract/train_${name}.log" | tr '\n' ' ')"
}

# <tag> <generator directory name> <task> <rollout> <commands> <last command name>
ladder() {
  local tag="$1" name="$2" task="$3" rollout="$4" commands="$5" last="$6" body
  body="$(body_of "$tag")"
  [ -f "$results/zero_shot_${body}_${tag}_${name}_own_auto_kin_l4_${last}.json" ] && return 0
  while isaac_busy; do sleep 60; done
  ( conditions "$body"
    ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK="$task" \
    ZS_COMMANDS="$commands" ZS_GENERATOR="logs/contract/${name}/generator.pt" \
    ZS_TEACHER_ROLLOUT="$rollout" \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "${runs[$tag]}" "${checkpoint[$body]}" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_${name}_${task}.log" 2>&1 \
    || { note "$tag $name $task FAILED"; return 1; }
  note "$task $tag $name"
}

note "w18 start"

# 1. The state after a reset, and the proxy calibrated from it.
[ -f logs/contract/w18_reset/generator.pt ] || cp -r logs/contract/mt17_c logs/contract/w18_reset
for tag in mt_g1_s41_sym mt_nao_s31_sym; do
  body="$(body_of "$tag")"
  reset="$body/rollouts/zs_${tag}_w18_reset_own_auto_kin_forward_encoder.npz"
  if [ ! -f "$reset" ]; then
    while isaac_busy; do sleep 60; done
    ( conditions "$body"
      ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK=rough \
      ZS_COMMANDS="forward 0.25 0 0" ZS_SETTLE=0 ZS_STEPS=40 \
      ZS_GENERATOR=logs/contract/w18_reset/generator.pt ZS_TEACHER_ROLLOUT=play \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "${runs[$tag]}" "${checkpoint[$body]}" "$tag" l4 ) \
      < /dev/null > "logs/contract/ladder_${tag}_w18_reset.log" 2>&1
    # 40 steps are no L4 reading: only the capture is kept.
    rm -f "$results/zero_shot_${body}_${tag}_w18_reset_own_auto_kin_l4_forward.json"
  fi
  [ -f "$reset" ] || { note "$tag reset capture FAILED"; continue; }
  out="$results/held_out_chains_w18_reset_${body}.json"
  [ -f "$out" ] || "$python" transfer/scripts/held_out_chains.py --start body --start-capture "$reset" \
    --start-step 2 --generators mt16_c mt17_c mt17_m rough_b4 --output "$out" \
    > "logs/contract/w18_calibration_${body}.log" 2>&1 || note "$body calibration FAILED"
  note "reset $body: $(grep 'mean largest' "logs/contract/w18_calibration_${body}.log" | tr -s ' ' | tr '\n' ';')"
done

# 2. The generators that never saw sidestep.
bodies=""
for f in g1/rollouts/w16_mt_g1_s41_sym_*.npz nao/rollouts/w16_mt_nao_s31_sym_*.npz; do
  [ -f "$f" ] && bodies="$bodies --body-data $f"
done
[ -n "$bodies" ] || { note "no body capture FAILED"; note "w18 done"; exit 1; }
# shellcheck disable=SC2086
train mt18_no_sidestep sidestep --epochs 200 $bodies --body-epochs 0 \
  || { note "w18 done"; exit 1; }
# shellcheck disable=SC2086
train mt18_no_sidestep_m sidestep --epochs 200 --init-from logs/contract/mt18_no_sidestep/generator.pt \
  $bodies --body-targets teacher --pair-command target

# 3. Sidestep's L4.
tags="mt_g1_s41_sym mt_g1_s42_sym mt_g1_s43_sym mt_nao_s31_sym mt_nao_s32_sym mt_nao_s33_sym"
for name in mt18_no_sidestep mt17_c mt18_no_sidestep_m mt17_m; do
  [ -f "logs/contract/${name}/generator.pt" ] || continue
  for tag in $tags; do
    ladder "$tag" "$name" base_sidestep play "$sidestep_commands" right
  done
done

# 4. The other leave-one-out generators, and every one chained on its held-out task.
for task in walk flat backward jog; do
  # shellcheck disable=SC2086
  train "mt18_no_${task}" "$task" --epochs 200 $bodies --body-epochs 0
done
reset_g1="g1/rollouts/zs_mt_g1_s41_sym_w18_reset_own_auto_kin_forward_encoder.npz"
for task in flat backward sidestep jog; do
  read -r source rollout commands <<< "${chains[$task]}"
  arms=""
  for name in "mt18_no_${task}" mt17_c; do
    [ -f "logs/contract/${name}/generator.pt" ] && arms="$arms $name"
  done
  out="$results/held_out_chains_w18_${task}_teacher.json"
  # shellcheck disable=SC2086
  [ -f "$out" ] || "$python" transfer/scripts/held_out_chains.py --task "$source" --rollout "$rollout" \
    --commands $commands --generators $arms --output "$out" \
    > "logs/contract/w18_chains_${task}_teacher.log" 2>&1 || note "$task teacher chains FAILED"
  out="$results/held_out_chains_w18_${task}_reset.json"
  if [ -f "$reset_g1" ] && [ ! -f "$out" ]; then
    # shellcheck disable=SC2086
    "$python" transfer/scripts/held_out_chains.py --task "$source" --rollout "$rollout" \
      --commands $commands --generators $arms --start body --start-capture "$reset_g1" \
      --start-step 2 --output "$out" > "logs/contract/w18_chains_${task}_reset.log" 2>&1 \
      || note "$task reset chains FAILED"
  fi
done
note "offline chains"

# 5. Compare.
"$python" transfer/scripts/w18_compare.py --output "$results/w18_loto.json" \
  > logs/contract/w18_compare.out 2>&1 || note "w18 compare FAILED"
note "w18 reading: $(tail -1 logs/contract/w18_compare.out)"
note "w18 done"
