#!/usr/bin/env bash
# The task side's correction after the blind test (transfer/docs/skill_space.md §8.4).
#
#   transfer/scripts/chain_study.sh <step1|stepA|c1|b1>
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
    --data "g1/rollouts/${task}_dart.npz" "$@" --gait "g1/rollouts/${task}_play_forward.npz" \
    --encoder-from "$body" --warp-loads --solver midpoint --flow-steps 8 --horizon 20 \
    --commit 13 --chain-rounds 8 --results "transfer/results/contract_${name}.json" \
    > "logs/contract/train_${name}.log" 2>&1
  echo "$name: $(grep -E 'chained:|gate' "logs/contract/train_${name}.log" | tr '\n' ' ') $(date +%H:%M)" \
    | tee -a "$status"
}

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
