#!/usr/bin/env bash
# Close the loop on a trained contract tracker, on the G1, against the teacher.
#
#   transfer/scripts/close_loop_contract.sh <run dir> <checkpoint> <contract> <tag>
#
# Exports the actor and captures 32 environments x 900 steps at the teacher's
# play command three times:
#
#   _encoder  the deployed path: the generator reads the body's history, as the
#             body's encoder estimates it (ancestral sampling of the task);
#   _truth    the same, from the true state: separates the tracker from the estimate;
#   _auto     the generator reading its own committed plans, as in training.
#
# The pre-registered gates are applied by evaluate_contract.py, which writes
# transfer/results/contract_g1_closed_loop_<tag>.json.
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
capture --reference physical --output "g1/rollouts/closed_loop_track_${tag}_encoder.npz" \
  > "logs/contract/capture_${tag}_encoder.log" 2>&1
capture --reference physical --contract-from-truth \
  --output "g1/rollouts/closed_loop_track_${tag}_truth.npz" \
  > "logs/contract/capture_${tag}_truth.log" 2>&1
capture --reference autonomous --contract-from-truth \
  --output "g1/rollouts/closed_loop_track_${tag}_auto.npz" \
  > "logs/contract/capture_${tag}_auto.log" 2>&1

"$python" transfer/scripts/evaluate_contract.py \
  --run teacher=g1/rollouts/closed_loop_teacher.npz \
  --run "track_${tag}_encoder=g1/rollouts/closed_loop_track_${tag}_encoder.npz" \
  --run "track_${tag}_truth=g1/rollouts/closed_loop_track_${tag}_truth.npz" \
  --run "track_${tag}_auto=g1/rollouts/closed_loop_track_${tag}_auto.npz" \
  --output "transfer/results/contract_g1_closed_loop_${tag}.json"
