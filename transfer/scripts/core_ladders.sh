#!/usr/bin/env bash
# The rest of step 2's ladder (transfer/docs/skill_space.md §9.1): its map, corners, encoder and
# walk captures were taken on 2026-10-23, but evaluate_latent.py resolved the worktree's linked
# rollouts out of its root and stopped the ladder before L4. L4 on both tasks, on the trained run.
#
#   bash transfer/scripts/core_ladders.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
status=logs/contract/core_and_nao.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
export HUMANOID_CONTRACT="$PWD/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
run="$(ls -d logs/rsl_rl/g1_track_core/*_warm_core | tail -1)"
checkpoint="$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID=G1TrackCore-Play-v0
for task in rough flat; do
  if [ "$task" = rough ]; then generator=rough_b4; else generator=flat_dagger2; fi
  export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
  bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "model_${checkpoint}" warm_core l4 \
    > "logs/contract/ladder_warm_core_${task}_l4.log" 2>&1 || note "warm_core $task L4 FAILED"
  note "evaluated warm_core $task L4"
done
