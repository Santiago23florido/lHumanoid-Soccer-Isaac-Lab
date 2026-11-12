#!/usr/bin/env bash
# What correcting the paper's NAO figures to the pinned reading needs beyond W14's captures
# (reading 1 "corrected"). It enters no W14 reading.
#
#   S's seed 31 (warm_nao_r1_s31_sym)  the paper's "S's seeds" rows hold it; W13 paired the
#                                      incumbent instead. Pinned L4 on flat_dagger2 and rough_b4,
#                                      deployed and true-state runs.
#   the adopted model (r1_3498_sym_v3) its pinned true-state runs on both skills, which the
#                                      paper's decomposition reads. Its deployed captures are
#                                      W14's, kept and scored again unchanged.
#
# The same conditions as w14_units.sh. One Isaac process at a time; a stage is skipped when
# its last true-state capture exists. It always ends with "w14 paper done".
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w14_paper.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }

conditions() {  # the NAO's, as w14_units.sh evaluates it
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
pinned() {  # <run dir> <checkpoint> <tag> <flat|rough>
  local run="$1" checkpoint="$2" tag="$3" task="$4" generator
  case "$task" in
    flat) generator=flat_dagger2 ;;
    rough) generator=rough_b4 ;;
  esac
  [ -f "nao/rollouts/zs_${tag}_${generator}_own_auto_turning_truth.npz" ] && return 0
  while isaac_busy; do sleep 60; done
  note "$tag $task start"
  ( conditions
    ZS_OBSERVE=own ZS_OWN_TRUTH=1 ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 \
    ZS_TASK_ID=NaoTrack-Play-v0 ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
    ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$checkpoint" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_${task}_own_paper.log" 2>&1 \
    || { note "$tag $task FAILED"; return 1; }
  note "$tag $task done"
}

note "w14 paper start"
for entry in "warm_nao_r1 model_4497 warm_nao_r1_s31_sym" "zs_nao_r1 model_3498 r1_3498_sym_v3"; do
  read -r name checkpoint tag <<< "$entry"
  run="$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)"
  for task in flat rough; do pinned "$run" "$checkpoint" "$tag" "$task"; done
done
note "w14 paper done"
