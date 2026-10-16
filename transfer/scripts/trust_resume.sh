#!/usr/bin/env bash
# Resume trust_study.sh after a pause, without repeating what finished: v2t is trained and its
# ladder ran l1, map, corners, encoder and walk on rough. What is left: L4 on rough and flat, then
# the incumbents' re-baseline (as trust_study.sh's stage 4). Runs at the priority it is started
# with; no keeper raises it.
#
#   transfer/scripts/trust_resume.sh
set -uo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/trust_study.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1

note "trust study resumed"
export HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1
export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID=G1TrackPhysics-Play-v0
run="$(ls -d logs/rsl_rl/g1_track_physics/*_decoder_v2t | tail -1)"
for task in rough flat; do
  if [ "$task" = rough ]; then generator=rough_b3; else generator=flat_dagger2; fi
  export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
  bash transfer/scripts/zero_shot_ladder.sh g1 "$run" model_1999 dec_v2t l4 \
    > "logs/contract/ladder_decoder_v2t_${task}_l4.log" 2>&1 || note "v2t ${task} l4 FAILED"
  note "evaluate v2t ${task} l4"
done
unset HUMANOID_HEADS HUMANOID_PLAN_TRUST ZS_TASK_ID

export ZS_SYMMETRIZE=1
for body in g1 nao; do
  if [ "$body" = g1 ]; then run=$g1_run ckpt=model_6194 old=r3_6194_sym; else
    run=$nao_run ckpt=model_3498 old=r1_3498_sym; fi
  new="${old}_v3"
  for generator in walk rough_b3 flat_dagger2; do
    source_dir="logs/contract/${generator}_zs_${body}_${old}"
    target_dir="logs/contract/${generator}_zs_${body}_${new}"
    [ -d "$source_dir" ] && [ ! -d "$target_dir" ] && cp -r "$source_dir" "$target_dir"
  done
  if [ "$body" = g1 ]; then
    bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "$ckpt" "$new" corners \
      > logs/contract/ladder_rebaseline_g1_corners.log 2>&1 || note "rebaseline g1 corners FAILED"
  fi
  for generator in rough_b3 rough_b4 flat_dagger2 flat_b4; do
    export ZS_TASK="${generator%%_*}" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$ckpt" "$new" l4 \
      > "logs/contract/ladder_rebaseline_${body}_${generator}.log" 2>&1 \
      || note "rebaseline ${body} ${generator} FAILED"
  done
  note "rebaseline ${body} done"
done
note "trust study done"
