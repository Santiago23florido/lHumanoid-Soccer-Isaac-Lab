#!/usr/bin/env bash
# F1 of transfer/docs/skill_space.md §9: the noise-conditional generators (B4) on both bodies.
#
#   transfer/scripts/b4_on_bodies.sh
#
# The amended L4 as step 3 ran it (autonomous reference, frozen symmetrized trackers, the
# learned encoder) with rough_b4 and flat_b4, so each compares with its B3 and with the
# incumbents in transfer/results/incumbents.json.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/b4_on_bodies.status
export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1
g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
echo "b4 l4 start $(date +%H:%M)" >> "$status"
for task in rough flat; do
  export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${task}_b4/generator.pt"
  bash transfer/scripts/zero_shot_ladder.sh g1 "$g1_run" model_6194 r3_6194_sym l4 \
    > "logs/contract/l4_${task}_b4_g1.log" 2>&1
  echo "l4 ${task}_b4 g1 $(date +%H:%M)" >> "$status"
  bash transfer/scripts/zero_shot_ladder.sh nao "$nao_run" model_3498 r1_3498_sym l4 \
    > "logs/contract/l4_${task}_b4_nao.log" 2>&1
  echo "l4 ${task}_b4 nao $(date +%H:%M)" >> "$status"
done
echo "b4 l4 done $(date +%H:%M)" >> "$status"
