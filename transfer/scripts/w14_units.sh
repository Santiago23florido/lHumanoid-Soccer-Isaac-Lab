#!/usr/bin/env bash
# W14 of skill_space.md §10.15: the NAO's frozen trackers observing each z* in the units of the
# walk contract they trained under (ZS_OBSERVE=own -> HUMANOID_OBSERVE_UNITS), L4 on the
# deployed path, paired with the same trackers' captures in the task's units (w14_compare.py).
# Nothing is trained, and the G1 is not run: its map is the identity.
#
#   S's NAO trackers    flat_dagger2 and rough_b4 at three commands, jog at its middle one
#   W13's NAO trackers  flat_dagger2 at three commands, jog at its middle one (rough waits)
#
# Waits for the current W13 run's "done". Each tracker runs in the conditions W13's ladder()
# and paper_queue.sh give it, so the units are the only difference. One Isaac process at a
# time; every stage is skipped when its last result exists. It always ends with "w14 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w14_units.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
conditions() {  # the NAO's, as paper_queue.sh and w13_decoders.sh's ladder() evaluate it
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7
  export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
  export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  unset HUMANOID_TASKS HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
}
isaac_busy() {  # no brackets or backslashes in the pattern (bash -> PowerShell quoting)
  local busy
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [ "$busy" != 0 ]
}
w13_done() {  # in the current run only: the status file keeps an earlier attempt's "done"
  awk '/^w13 start/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
    logs/contract/w13_decoders.status 2>/dev/null | grep -q "^w13 done"
}
own_l4() {  # <run dir> <checkpoint> <tag> <flat|rough|jog>
  local run="$1" checkpoint="$2" tag="$3" task="$4" zs_task generator commands="" last=turning
  case "$task" in
    flat) zs_task=flat generator=flat_dagger2 ;;
    rough) zs_task=rough generator=rough_b4 ;;
    jog) zs_task=base_jog generator=base_jog_b4 commands="mid 1.25 0 0" last=mid ;;
  esac
  [ -f "$results/zero_shot_nao_${tag}_${generator}_own_auto_kin_l4_${last}.json" ] && return 0
  while isaac_busy; do sleep 60; done
  note "$tag $task start"
  ( conditions
    ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 \
    ZS_TASK_ID=NaoTrack-Play-v0 ZS_TASK="$zs_task" \
    ZS_GENERATOR="logs/contract/${generator}/generator.pt" ZS_COMMANDS="$commands" \
    ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$checkpoint" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_${task}_own.log" 2>&1 \
    || { note "$tag $task FAILED"; return 1; }
  note "$tag $task done"
}

until w13_done; do sleep 120; done
note "w14 start"
for entry in "zs_nao_r1 model_3498 r1_3498_sym_v3" \
  "warm_nao_r1_s32 model_4497 warm_nao_r1_s32_sym" "warm_nao_r1_s33 model_4497 warm_nao_r1_s33_sym"; do
  read -r name checkpoint tag <<< "$entry"
  run="$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)"
  for task in flat rough jog; do own_l4 "$run" "$checkpoint" "$tag" "$task"; done
done
for seed in 31 32 33; do
  run="$(ls -d logs/rsl_rl/nao_track/*_mt_nao_s"$seed" 2>/dev/null | tail -1)"
  [ -n "$run" ] || { note "mt_nao_s$seed: no run FAILED"; continue; }
  for task in flat jog; do own_l4 "$run" "$(latest "$run")" "mt_nao_s${seed}_sym" "$task"; done
done
"$python" transfer/scripts/w14_compare.py --output "$results/w14_units.json" \
  > logs/contract/w14_compare.out 2>&1 || note "w14 compare FAILED"
note "w14 reading: $(tail -1 logs/contract/w14_compare.out)"
note "w14 done"
