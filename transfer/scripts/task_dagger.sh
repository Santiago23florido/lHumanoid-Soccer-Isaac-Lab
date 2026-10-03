#!/usr/bin/env bash
# DAgger on a published task's generator, on the G1 -- the task side, nothing of a body trained.
#
#   transfer/scripts/task_dagger.sh <flat|rough> <G1 run dir> <checkpoint> <G1 body contract> \
#       <round> <name> [extra data.npz...]
#
# A generator trained on its teacher's own rollouts meets, in a closed loop, histories its
# teacher never wrote -- the G1's tracker standing at rest, then starting -- and its plans
# there are meaningless (the published flat generator's chain diverged from a standing
# start). The remedy for compounding error is to ask the expert in the states the learner
# reaches (Ross, Gordon and Bagnell, 2011). The learner is the loop of this generator and the
# G1's frozen tracker; the expert, the published policy itself:
#   1. the loop captured on the G1 (true state, the body's own history) under commands from
#      the task's range, and with starts and stops;
#   2. the published policy acting from those states (teacher_from_states.py);
#   3. the generator retrained on its teacher's data and this round's (recipe of flat_h20),
#      on the G1 body's encoder: logs/contract/<name>.
# The G1 is the task's own body; the NAO is never involved, so a new task is still trained once.
set -euo pipefail

task="$1"
run="$2"
checkpoint="$3"
body_contract="$4"
round="$5"
name="$6"
shift 6
extra=("$@")
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

if [ "$task" = rough ]; then
  teacher_task=G1RoughOnFlat-v0
  published=logs/published/Isaac-Velocity-Rough-G1-v0/policy.pt
  commands="0.25 0 0
0.6 0 0
0.3 0 0.3
0.45 0 -0.4
0.1 0 0.6"
else
  teacher_task=Isaac-Velocity-Flat-G1-v0
  published=logs/published/Isaac-Velocity-Flat-G1-v0/policy.pt
  commands="0.25 0 0
0.6 0 0
0.3 0.2 0.3
0.45 -0.15 -0.4
0.1 0 0.6"
fi
generator="logs/contract/${task}_zs_g1_dagger_input/generator.pt"
previous="${TASK_GENERATOR:-logs/contract/${task}_h20/generator.pt}"
"$python" transfer/scripts/body_encoder.py "$previous" "$body_contract" --output "$generator"
"$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt" > /dev/null
policy="$run/exported/$checkpoint.pt"
export HUMANOID_CONTRACT="$root/$generator"

capture() {
  local tag="$1"
  shift
  "$python" -u g1/scripts/capture_rollouts.py --headless --task-id G1Track-Play-v0 \
    --driver-policy "$policy" --num-envs 32 --steps 900 --settle-steps 100 \
    --reference physical --contract-from-truth "$@" \
    --output "g1/rollouts/${task}_loop_${round}_${tag}.npz" \
    > "logs/contract/capture_${task}_loop_${round}_${tag}.log" 2>&1
  tail -1 "logs/contract/capture_${task}_loop_${round}_${tag}.log"
}

states=()
i=0
while read -r vx vy wz; do
  capture "c${i}" --command "$vx" "$vy" "$wz"
  states+=(--states "g1/rollouts/${task}_loop_${round}_c${i}.npz")
  i=$((i + 1))
done <<< "$commands"
capture starts --standing-fraction 0.3 --resample-seconds 2 5
states+=(--states "g1/rollouts/${task}_loop_${round}_starts.npz")

"$python" -u g1/scripts/teacher_from_states.py --headless "${states[@]}" \
  --checkpoint "$published" --task-id "$teacher_task" --fixed-command \
  --output "g1/rollouts/${task}_dagger_${round}.npz" > "logs/contract/${task}_dagger_${round}.log" 2>&1
tail -1 "logs/contract/${task}_dagger_${round}.log"

data=(--data "g1/rollouts/${task}_dart.npz")
for path in "${extra[@]}" "g1/rollouts/${task}_dagger_${round}.npz"; do
  data+=(--data "$path")
done
"$python" -u transfer/scripts/train_contract.py --name "$name" "${data[@]}" \
  --gait "g1/rollouts/${task}_play_forward.npz" --encoder-from "$body_contract" \
  --warp-loads --solver midpoint --flow-steps 8 --horizon 20 --commit 13 \
  --results "transfer/results/contract_${name}.json" > "logs/contract/train_${name}.log" 2>&1
tail -4 "logs/contract/train_${name}.log"
