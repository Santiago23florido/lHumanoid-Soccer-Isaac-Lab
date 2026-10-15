#!/usr/bin/env bash
# The task side's correction after the blind test (transfer/docs/skill_space.md §8.4).
#
#   transfer/scripts/chain_study.sh <step1|stepA|c1|b1|b2|b3|l4>
#
# step1  the published policies' generators retrained with training on their own chains
#        (Data as Demonstrator): rough_dad and flat_dad, on the data of rough_dagger2 and
#        flat_dagger2, the G1's frozen encoder, the chains held by the data loss's own error.
# stepA  the same, the chains held by the plans' timing calibration against its gate instead:
#        rough_dadA and flat_dadA. Each reports its chain fidelity.
# c1     stepA with phase-consistent targets: each written history trained toward the
#        teacher's future from its nearest window, not the same instant -> rough_c1, flat_c1.
# b1     the rhythm of the generator's own chain matched to its teacher's, differentiably
#        through the chain, held by calibration -> rough_b1, flat_b1.
# b2     twice the history (52 samples), the data alone, no chain training -> rough_b2, flat_b2.
# b3     the data alone, under the body encoder's own precision instead of its error on the
#        teacher's states (--sigma-from-body) -> rough_b3, flat_b3.
# l4     step 3: the amended L4 with rough_b3 and flat_b3 on the G1 (round 3) and the NAO
#        (round 1), both frozen and deployed symmetrized. Rough is no longer blind.
set -euo pipefail

stage="${1:-step1}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
body=logs/contract/walk_zs_g1_r3_6194_sym/generator.pt
status=logs/contract/chain_study.status

train() {
  local name="$1" task="$2"
  shift 2
  "$python" -u transfer/scripts/train_contract.py --name "$name" \
    --data "g1/rollouts/${task}_dart.npz" --gait "g1/rollouts/${task}_play_forward.npz" \
    --encoder-from "$body" --warp-loads --solver midpoint --flow-steps 8 --horizon 20 \
    --commit 13 --chain-rounds 8 --results "transfer/results/contract_${name}.json" "$@" \
    > "logs/contract/train_${name}.log" 2>&1
  echo "$name: $(grep -E 'chained:|gate' "logs/contract/train_${name}.log" | tr '\n' ' ') $(date +%H:%M)" \
    | tee -a "$status"
}

if [ "$stage" = l4 ]; then
  echo "l4 start $(date +%H:%M)" >> "$status"
  export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1
  g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
  nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
  for task in rough flat; do
    export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${task}_b3/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh g1 "$g1_run" model_6194 r3_6194_sym l4       > "logs/contract/l4_${task}_b3_g1.log" 2>&1
    echo "l4 ${task}_b3 g1 $(date +%H:%M)" >> "$status"
    bash transfer/scripts/zero_shot_ladder.sh nao "$nao_run" model_3498 r1_3498_sym l4       > "logs/contract/l4_${task}_b3_nao.log" 2>&1
    echo "l4 ${task}_b3 nao $(date +%H:%M)" >> "$status"
  done
  echo "l4 done $(date +%H:%M)" >> "$status"
fi
if [ "$stage" = b3 ]; then
  echo "b3 start $(date +%H:%M)" >> "$status"
  train rough_b3 rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz --chain-rounds 0 --sigma-from-body
  train flat_b3 flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz --chain-rounds 0 --sigma-from-body
  echo "b3 done $(date +%H:%M)" >> "$status"
fi
if [ "$stage" = b2 ]; then
  echo "b2 start $(date +%H:%M)" >> "$status"
  train rough_b2 rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz --chain-rounds 0 --history 52
  train flat_b2 flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz --chain-rounds 0 --history 52
  echo "b2 done $(date +%H:%M)" >> "$status"
fi
if [ "$stage" = b1 ]; then
  echo "b1 start $(date +%H:%M)" >> "$status"
  train rough_b1 rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz --chain-objective rhythm
  train flat_b1 flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz --chain-objective rhythm
  echo "b1 done $(date +%H:%M)" >> "$status"
fi
if [ "$stage" = c1 ]; then
  echo "c1 start $(date +%H:%M)" >> "$status"
  train rough_c1 rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz --chain-constraint calibration --chain-targets phase
  train flat_c1 flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz --chain-constraint calibration --chain-targets phase
  echo "c1 done $(date +%H:%M)" >> "$status"
fi
if [ "$stage" = stepA ]; then
  echo "stepA start $(date +%H:%M)" >> "$status"
  train rough_dadA rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz     --chain-constraint calibration
  train flat_dadA flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz     --chain-constraint calibration
  echo "stepA done $(date +%H:%M)" >> "$status"
fi
if [ "$stage" = step1 ]; then
  echo "step1 start $(date +%H:%M)" >> "$status"
  train rough_dad rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz     --chain-constraint data
  train flat_dad flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz     --chain-constraint data
  echo "step1 done $(date +%H:%M)" >> "$status"
fi
