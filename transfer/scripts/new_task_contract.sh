#!/usr/bin/env bash
# A new task on a body that already has a contract tracker, with nothing of the body retrained.
#
#   transfer/scripts/new_task_contract.sh <tracker policy.pt> <body contract> <tag> [generator]
#
# The new task is Isaac Lab's published Isaac-Velocity-Flat-G1-v0 policy
# (g1/scripts/export_published.py): forward to 1 m/s, sideways, turning.
#  1. its rollouts: DART with starts and stops for the generator, and play rollouts
#     under three fixed commands to compare with;
#  2. its contract generator, on the body's encoder and units (--encoder-from),
#     saved as logs/contract/<generator>/generator.pt (default flat_h20);
#  3. the body's tracker, unchanged, driven by that generator under the same commands --
#     plans drawn from the body's history as deployed, through the body's encoder and,
#     to separate the tracker from the estimate, from the true state;
#  4. evaluate_contract.py per command, the published policy as the reference.
set -euo pipefail

policy="$1"
body_contract="$2"
tag="$3"
generator="${4:-flat_h20}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
published="logs/published/Isaac-Velocity-Flat-G1-v0/policy.pt"
capture() { "$python" -u g1/scripts/capture_rollouts.py --headless "$@"; }

if [ ! -f g1/rollouts/flat_dart.npz ]; then
  capture --task-id Isaac-Velocity-Flat-G1-v0 --checkpoint "$published" --num-envs 128 \
    --steps 1000 --action-noise 0.1 --standing-fraction 0.3 --resample-seconds 2 6 \
    --output g1/rollouts/flat_dart.npz > logs/contract/capture_flat_dart.log 2>&1
fi

declare -A commands=( [forward]="0.25 0 0" [fast]="0.6 0 0" [turning]="0.3 0.2 0.3" )
for name in forward fast turning; do
  if [ ! -f "g1/rollouts/flat_play_${name}.npz" ]; then
    capture --task-id Isaac-Velocity-Flat-G1-Play-v0 --checkpoint "$published" \
      --num-envs 32 --steps 900 --settle-steps 100 --command ${commands[$name]} \
      --output "g1/rollouts/flat_play_${name}.npz" > "logs/contract/capture_flat_${name}.log" 2>&1
  fi
done

if [ ! -f "logs/contract/${generator}/generator.pt" ]; then
  "$python" -u transfer/scripts/train_contract.py --name "$generator" \
    --data g1/rollouts/flat_dart.npz --gait g1/rollouts/flat_play_forward.npz \
    --encoder-from "$body_contract" --warp-loads --solver midpoint --flow-steps 8 \
    > "logs/contract/train_${generator}.log" 2>&1
fi

export HUMANOID_CONTRACT="$root/logs/contract/${generator}/generator.pt"
for name in forward fast turning; do
  for path in encoder truth; do
    extra=()
    if [ "$path" = truth ]; then extra=(--contract-from-truth); fi
    capture --task-id G1Track-Play-v0 --driver-policy "$policy" --reference physical \
      "${extra[@]}" --num-envs 32 --steps 900 --settle-steps 100 \
      --command ${commands[$name]} --output "g1/rollouts/track_flat_${tag}_${name}_${path}.npz" \
      > "logs/contract/capture_track_flat_${tag}_${name}_${path}.log" 2>&1
  done
  "$python" transfer/scripts/evaluate_contract.py \
    --run "teacher=g1/rollouts/flat_play_${name}.npz" \
    --run "encoder=g1/rollouts/track_flat_${tag}_${name}_encoder.npz" \
    --run "truth=g1/rollouts/track_flat_${tag}_${name}_truth.npz" \
    --output "transfer/results/contract_new_task_${tag}_${name}.json"
done
