#!/usr/bin/env bash
# F2 of transfer/docs/skill_space.md §9: the deployed path reads the body's state with its
# kinematics -- every field of z* but the feet's loads -- instead of the learned encoder alone.
#
#   transfer/scripts/estimator_study.sh [g1|nao|both]
#
# The amended L4 (autonomous reference, frozen symmetrized trackers) with the incumbents,
# rough_b3 and flat_dagger2 (the regression), ZS_ESTIMATOR=kinematic; the true-state
# captures are reused. Then the estimate each capture recorded is scored against the true
# state (encoder_error.py --recorded).
set -euo pipefail

bodies="${1:-both}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/estimator_study.status
export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1 ZS_ESTIMATOR=kinematic

run_body() {
  local body="$1" run checkpoint tag rollouts
  if [ "$body" = g1 ]; then
    run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss checkpoint=model_6194
    tag=r3_6194_sym rollouts=g1/rollouts
  else
    run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1 checkpoint=model_3498
    tag=r1_3498_sym rollouts=nao/rollouts
  fi
  for generator in rough_b3 flat_dagger2; do
    export ZS_TASK="${generator%%_*}" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" l4 \
      > "logs/contract/l4_${generator}_kin_${body}.log" 2>&1
    echo "l4 ${generator} kinematic ${body} $(date +%H:%M)" >> "$status"
  done
  local captures=()
  for generator in rough_b3 flat_dagger2; do
    for name in forward fast turning; do
      captures+=("$rollouts/zs_${tag}_${generator}_auto_kin_${name}_encoder.npz")
    done
  done
  "$python" transfer/scripts/encoder_error.py "logs/contract/rough_b3_zs_${body}_${tag}/generator.pt" \
    "${captures[@]}" --recorded --device cpu \
    --output "transfer/results/estimator_error_${body}_${tag}_kin.json" >> "$status" 2>&1
}

echo "estimator start $(date +%H:%M)" >> "$status"
if [ "$bodies" = g1 ] || [ "$bodies" = both ]; then run_body g1; fi
if [ "$bodies" = nao ] || [ "$bodies" = both ]; then run_body nao; fi
echo "estimator done $(date +%H:%M)" >> "$status"
