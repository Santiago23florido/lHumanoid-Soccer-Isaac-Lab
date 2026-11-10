#!/usr/bin/env bash
# W15 of skill_space.md §10.16: the family's share alone, on the NAO. S's NAO recipe
# (paper_queue.sh: the incumbent continued 1000 iterations, seed 31) with the family's share
# at 0.5 and no task plans, the one W13 change that is not the G1's task plans.
#
#   1. train     fam05_nao_s31
#   2. evaluate  encoder refit and S0's walk, as W13's ladder(); then flat's L4 at the three
#                commands and jog's at its middle one, on the NAO's pinned path (W14)
#   3. compare   w15_compare.py -> transfer/results/w15_family.json
#
# Waits for w14_paper.sh's "done" (S's seed 31, pinned, is the reading's lower anchor). One
# Isaac process at a time; every stage is skipped when its output exists. It always ends
# with "w15 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w15_family.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
checkpoint=model_3498
name=fam05_nao_s31
tag="${name}_sym"

rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
conditions() {  # S's NAO conditions (paper_queue.sh), the family's share given
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL="$1"
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
ladder() {  # <run dir> <task: flat|jog> <observe: task|own> <stages...>; as W13 evaluates
  local run="$1" task="$2" observe="$3" generator=flat_dagger2 commands="" zs_task=flat truth=1
  shift 3
  if [ "$task" = jog ]; then
    generator=base_jog_b4 commands="mid 1.25 0 0" zs_task=base_jog truth=0
  fi
  ( conditions 0.7  # S was evaluated under its training conditions, and so was W13
    ZS_OBSERVE="$observe" ZS_OWN_TRUTH="$truth" ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic \
    ZS_SYMMETRIZE=1 \
    ZS_TASK_ID=NaoTrack-Play-v0 ZS_TASK="$zs_task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
    ZS_COMMANDS="$commands" ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$(latest "$run")" "$tag" "$@" ) \
    < /dev/null > "logs/contract/ladder_${tag}_${task}_${observe}.log" 2>&1
}

until grep -q "^w14 paper done" logs/contract/w14_paper.status 2>/dev/null; do sleep 120; done
note "w15 start"
if ! ls -d logs/rsl_rl/nao_track/*_"$name" > /dev/null 2>&1; then
  while isaac_busy; do sleep 60; done
  note "$name training start"
  ( conditions 0.5
    "$python" -u scripts/train.py --task NaoTrack-v0 --headless --num_envs 4096 \
      --max_iterations 1000 --seed 31 --run_name "$name" --resume \
      --load_run "$(basename "$incumbent")" --checkpoint "$checkpoint.pt" \
      "agent.algorithm.learning_rate=$(rate "$incumbent/$checkpoint.pt")" \
      > "logs/contract/train_${name}.log" 2>&1 ) || { note "$name training FAILED"; note "w15 done"; exit 1; }
  note "$name trained"
fi
run="$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)"
if [ ! -f "$results/zero_shot_nao_${tag}_walk_auto_kin.json" ]; then
  while isaac_busy; do sleep 60; done
  ladder "$run" flat task encoder walk || note "$tag encoder walk FAILED"
  note "evaluated $tag encoder walk"
fi
if [ ! -f "$results/zero_shot_nao_${tag}_flat_dagger2_own_auto_kin_l4_turning.json" ]; then
  while isaac_busy; do sleep 60; done
  ladder "$run" flat own l4 || note "$tag flat FAILED"
  note "evaluated $tag flat"
fi
if [ ! -f "$results/zero_shot_nao_${tag}_base_jog_b4_own_auto_kin_l4_mid.json" ]; then
  while isaac_busy; do sleep 60; done
  ladder "$run" jog own l4 || note "$tag jog FAILED"
  note "evaluated $tag jog"
fi
"$python" transfer/scripts/w15_compare.py --output "$results/w15_family.json" \
  > logs/contract/w15_compare.out 2>&1 || note "w15 compare FAILED"
note "w15 reading: $(tail -1 logs/contract/w15_compare.out)"
note "w15 done"
