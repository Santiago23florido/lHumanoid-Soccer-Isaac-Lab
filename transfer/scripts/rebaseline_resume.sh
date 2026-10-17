#!/usr/bin/env bash
# What the paused re-baseline left (trust_resume.sh, stage 2): flat_b4's L4 on the G1, whose
# turning command did not finish, and the NAO's L4 with rough_b3, rough_b4, flat_dagger2 and
# flat_b4, the incumbent tracker under this code (kinematic estimator, sole corners).
#
#   transfer/scripts/rebaseline_resume.sh
set -uo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/trust_study.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1 ZS_ESTIMATOR=kinematic

note "re-baseline resumed"
export ZS_TASK=flat ZS_GENERATOR=logs/contract/flat_b4/generator.pt
bash transfer/scripts/zero_shot_ladder.sh g1 logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss \
  model_6194 r3_6194_sym_v3 l4 > logs/contract/ladder_rebaseline_g1_flat_b4.log 2>&1 \
  || note "rebaseline g1 flat_b4 FAILED"
note "rebaseline g1 done"

run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
for generator in walk rough_b3 flat_dagger2; do
  source_dir="logs/contract/${generator}_zs_nao_r1_3498_sym"
  target_dir="logs/contract/${generator}_zs_nao_r1_3498_sym_v3"
  [ -d "$source_dir" ] && [ ! -d "$target_dir" ] && cp -r "$source_dir" "$target_dir"
done
for generator in rough_b3 rough_b4 flat_dagger2 flat_b4; do
  export ZS_TASK="${generator%%_*}" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
  bash transfer/scripts/zero_shot_ladder.sh nao "$run" model_3498 r1_3498_sym_v3 l4 \
    > "logs/contract/ladder_rebaseline_nao_${generator}.log" 2>&1 \
    || note "rebaseline nao ${generator} FAILED"
  note "rebaseline nao ${generator}"
done
note "trust study done"
