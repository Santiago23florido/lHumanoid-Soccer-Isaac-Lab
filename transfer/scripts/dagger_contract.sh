#!/usr/bin/env bash
# One round of DAgger on a task's contract generator (Ross, Gordon and Bagnell, 2011).
#
#   transfer/scripts/dagger_contract.sh <run dir> <checkpoint> <contract> <tag> <round> \
#       <name> <data.npz>...
#
# A generator calibrated on the teacher's histories can still drift once it reads
# histories the closed loop wrote -- measured on the walk: forecasts of lift-off
# within 0.03 of the data at every stance age, yet stances a third shorter when
# chained on its own plans. The remedy for compounding error is to ask the expert
# in the states the learner reaches. Here the learner is the loop of generator
# and tracker, and the expert is still the teacher:
#
# 1. the closed-loop captures of `close_loop_contract.sh <tag>` (deployed,
#    true-state and autonomous references), plus two more with the same tracker
#    over the rest of the task's command range -- starts and stops, half speed;
# 2. `teacher_from_states.py` lets the pinned teacher act from those states;
# 3. the generator is retrained on everything so far (the given data and this
#    round's), as `logs/contract/<name>/generator.pt`.
set -euo pipefail

run="$1"
checkpoint="$2"
contract="$3"
tag="$4"
round="$5"
name="$6"
shift 6
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

# The tracker as close_loop_contract.sh exported it, with the contract it follows.
policy="$run/exported/$checkpoint.pt"
export HUMANOID_CONTRACT="$root/$contract"

capture() {
  "$python" -u g1/scripts/capture_rollouts.py --headless --task-id G1Track-Play-v0 \
    --driver-policy "$policy" --num-envs 32 --steps 900 --settle-steps 100 \
    --reference physical --contract-from-truth "$@"
}
capture --standing-fraction 0.3 --resample-seconds 2 5 \
  --output "g1/rollouts/closed_loop_track_${tag}_starts.npz" \
  > "logs/contract/capture_${tag}_starts.log" 2>&1
capture --command 0.12 0 0 --output "g1/rollouts/closed_loop_track_${tag}_slow.npz" \
  > "logs/contract/capture_${tag}_slow.log" 2>&1

states=()
for kind in encoder truth auto starts slow; do
  states+=(--states "g1/rollouts/closed_loop_track_${tag}_${kind}.npz")
done
"$python" -u g1/scripts/teacher_from_states.py --headless "${states[@]}" \
  --output "g1/rollouts/dagger_contract_${round}.npz" \
  > "logs/contract/dagger_contract_${round}.log" 2>&1
tail -1 "logs/contract/dagger_contract_${round}.log"

data=()
for path in "$@" "g1/rollouts/dagger_contract_${round}.npz"; do
  data+=(--data "$path")
done
"$python" -u transfer/scripts/train_contract.py --name "$name" "${data[@]}" \
  --warp-loads --solver midpoint --flow-steps 8 \
  --results "transfer/results/contract_${name}.json" \
  > "logs/contract/train_${name}.log" 2>&1
tail -4 "logs/contract/train_${name}.log"
