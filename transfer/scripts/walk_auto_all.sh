#!/usr/bin/env bash
# S0's walk as amended on 2026-10-24 (transfer/docs/skill_space.md §9): the deployed path under
# the autonomous reference, for the incumbents and every warm-study arm. The ladder's walk stage
# now captures it beside the physical reference; each arm runs from the tree it was trained with.
#
#   bash transfer/scripts/walk_auto_all.sh
set -uo pipefail
main="$(cd "$(dirname "$0")/../.." && pwd)"
worktree="$main/../lHumanoid-wt-core"
status="$main/logs/contract/walk_auto_all.status"
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
export ZS_ESTIMATOR=kinematic ZS_GENERATOR=logs/contract/rough_b4/generator.pt ZS_TASK=rough

# tree | body | run | checkpoint | tag | task id | heads+trust (0/1) | symmetrize
arms="main|g1|logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss|model_6194|r3_6194_sym_v3|G1Track-Play-v0|0|1
main|g1|warm_r3_r8||warm_r3_r8|G1Track-Play-v0|0|0
main|g1|warm_v2t_r8||warm_v2t_r8|G1TrackPhysics-Play-v0|1|0
worktree|g1|warm_plain||warm_plain|G1Track-Play-v0|0|0
worktree|g1|warm_core||warm_core|G1TrackCore-Play-v0|0|0
main|nao|logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1|model_3498|r1_3498_sym_v3|NaoTrack-Play-v0|0|1
main|nao|warm_nao_r1||warm_nao_r1|NaoTrack-Play-v0|0|0
worktree|nao|warm_nao_v2t||warm_nao_v2t|NaoTrackPhysics-Play-v0|1|0"

note "walk (autonomous) for every arm: start"
while IFS='|' read -r tree body run checkpoint tag task_id physics symmetrize; do
  if [ "$tree" = worktree ]; then cd "$worktree"; else cd "$main"; fi
  if [ -z "$checkpoint" ]; then
    experiment="${body}_track"
    case "$task_id" in *Physics*) experiment="${body}_track_physics" ;; *Core*) experiment="${body}_track_core" ;; esac
    run="$(ls -d "logs/rsl_rl/$experiment/"*"_$tag" | tail -1)"
    checkpoint="model_$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
  fi
  HUMANOID_HEADS="$physics" HUMANOID_PLAN_TRUST="$physics" HUMANOID_FEEDFORWARD_OBS="$physics" \
    ZS_TASK_ID="$task_id" ZS_SYMMETRIZE="$symmetrize" \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" walk \
    > "$main/logs/contract/walk_auto_${tag}.log" 2>&1 || note "$tag walk FAILED"
  note "$tag: $(grep -E '^encoder' "$main/logs/contract/walk_auto_${tag}.log" | sed -n 2p | cut -c1-110)"
done <<< "$arms"
note "walk (autonomous) for every arm: done"
