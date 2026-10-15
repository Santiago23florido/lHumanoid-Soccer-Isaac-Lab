#!/usr/bin/env bash
# The task side's correction after the blind test (transfer/docs/skill_space.md §8.4).
#
#   transfer/scripts/chain_study.sh step1
#
# step1  the published policies' generators retrained with training on their own chains
#        (Data as Demonstrator): rough_dad and flat_dad, on the data of rough_dagger2 and
#        flat_dagger2, the G1's frozen encoder. Each reports its chain gate.
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
    --commit 13 --chain-rounds 3 --results "transfer/results/contract_${name}.json" \
    > "logs/contract/train_${name}.log" 2>&1
  echo "$name: $(grep -E 'chained:|gate' "logs/contract/train_${name}.log" | tr '\n' ' ') $(date +%H:%M)" \
    | tee -a "$status"
}

if [ "$stage" = step1 ]; then
  echo "step1 start $(date +%H:%M)" >> "$status"
  train rough_dad rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz
  train flat_dad flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz
  echo "step1 done $(date +%H:%M)" >> "$status"
fi
