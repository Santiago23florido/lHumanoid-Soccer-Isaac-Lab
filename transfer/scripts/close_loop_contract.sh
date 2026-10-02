#!/usr/bin/env bash
# Close the loop on a trained contract tracker, on the G1, against the teacher.
#
#   transfer/scripts/close_loop_contract.sh <run dir> <checkpoint> <contract> <tag>
#
# Exports the actor, captures 32 environments x 900 steps at the teacher's play
# command twice -- the contract fed by the true state, and by the body's encoder
# (the deployed path) -- and applies the pre-registered gates with
# evaluate_contract.py, writing transfer/results/contract_g1_closed_loop_<tag>.json.
set -euo pipefail

run="$1"
checkpoint="$2"
contract="$3"
tag="$4"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

"$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt"
policy="$run/exported/$checkpoint.pt"
export HUMANOID_CONTRACT="$root/$contract"

capture() {
  "$python" -u g1/scripts/capture_rollouts.py --headless --task-id G1Track-Play-v0 \
    --driver-policy "$policy" --num-envs 32 --steps 900 --settle-steps 100 "$@"
}
capture --contract-from-truth --output "g1/rollouts/closed_loop_track_${tag}_truth.npz" \
  > "logs/contract/capture_${tag}_truth.log" 2>&1
capture --output "g1/rollouts/closed_loop_track_${tag}.npz" \
  > "logs/contract/capture_${tag}.log" 2>&1

"$python" transfer/scripts/evaluate_contract.py \
  --run teacher=g1/rollouts/closed_loop_teacher.npz \
  --run "track_${tag}_truth=g1/rollouts/closed_loop_track_${tag}_truth.npz" \
  --run "track_${tag}_encoder=g1/rollouts/closed_loop_track_${tag}.npz" \
  --output "transfer/results/contract_g1_closed_loop_${tag}.json"
