#!/usr/bin/env bash
# Cycle B4 of the task side's correction (transfer/docs/skill_space.md §8.4): one generator per
# task trained across history noise levels, reading the level as an input -- a denoiser built in,
# with no level fixed in training -- and deployed at the body encoder's own error.
#
#   transfer/scripts/chain_study_b4.sh
#
# Trains rough_b4 and flat_b4 on the data of rough_dagger2 and flat_dagger2 under the body
# encoder's precision (--sigma-from-body --noise-conditional), then measures every chain, B3's
# and B4's, at 1, 3, 10 and 20 times that error: B4 should match B3 at 1 and hold up above it.
set -euo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
body=logs/contract/walk_zs_g1_r3_6194_sym/generator.pt
status=logs/contract/chain_study_b4.status

train() {
  local name="$1" task="$2"
  shift 2
  "$python" -u transfer/scripts/train_contract.py --name "$name" \
    --data "g1/rollouts/${task}_dart.npz" --gait "g1/rollouts/${task}_play_forward.npz" \
    --encoder-from "$body" --warp-loads --solver midpoint --flow-steps 8 --horizon 20 \
    --commit 13 --results "transfer/results/contract_${name}.json" "$@" \
    > "logs/contract/train_${name}.log" 2>&1
  echo "$name: $(grep -E 'chain fidelity|gate' "logs/contract/train_${name}.log" | tr '\n' ' ') $(date +%H:%M)" \
    | tee -a "$status"
}

echo "b4 start $(date +%H:%M)" > "$status"
train rough_b4 rough --data g1/rollouts/rough_dagger_1.npz --data g1/rollouts/rough_dagger_2.npz \
  --sigma-from-body --noise-conditional
train flat_b4 flat --data g1/rollouts/flat_dagger_1.npz --data g1/rollouts/flat_dagger_2.npz \
  --sigma-from-body --noise-conditional
for task in rough flat; do
  for name in "${task}_b3" "${task}_b4"; do
    for scale in 1 3 10 20; do
      "$python" transfer/scripts/chain_fidelity.py "logs/contract/${name}/generator.pt" \
        --gait "g1/rollouts/${task}_play_forward.npz" --noise-scale "$scale" \
        --output "transfer/results/chain_fidelity_${name}_x${scale}.json" | tee -a "$status"
    done
  done
done
echo "b4 done $(date +%H:%M)" >> "$status"
