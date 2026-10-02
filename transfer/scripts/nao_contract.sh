#!/usr/bin/env bash
# The NAO following contracts made by G1 policies, held to those policies in body units.
#
#   transfer/scripts/nao_contract.sh <run dir> <checkpoint> <walk contract> <new contract> <tag>
#
# The NAO's tracker was trained by RL on the walk's contract only.
#  1. Walk, from the true state, at the teacher's play command in the NAO's units: plans
#     drawn from the NAO's own history and from the autonomous reference.
#  2. The NAO's encoder, fitted on those captures (the anchor alone): the walk's contract
#     with it is logs/contract/walk_nao_<tag>, and the walk is captured again through it --
#     the deployed path.
#  3. The published task's contract, unchanged, with the NAO's encoder moved over
#     (body_encoder.py), under the three commands the G1 was tested with, each in the
#     NAO's units (speeds scale with V = sqrt(g L), turn rates with 1 / T, T = sqrt(L / g)),
#     from the true state and through the encoder.
# Each capture is held to the G1 policy that produced the contract (evaluate_contract.py
# compares speeds in V and gaits in T and L).
set -euo pipefail

run="$1"
checkpoint="$2"
walk="$3"
new="$4"
tag="$5"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

"$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt"
policy="$run/exported/$checkpoint.pt"
mkdir -p nao/rollouts

capture() {
  "$python" -u transfer/scripts/capture_contract.py --headless --task-id NaoTrack-Play-v0 \
    --driver-policy "$policy" --num-envs 32 --steps 900 --settle-steps 100 "$@"
}

export HUMANOID_CONTRACT="$root/$walk"
for reference in physical autonomous; do
  capture --contract-from-truth --reference "$reference" \
    --output "nao/rollouts/track_walk_${tag}_${reference}.npz" \
    > "logs/contract/capture_nao_walk_${tag}_${reference}.log" 2>&1
  tail -1 "logs/contract/capture_nao_walk_${tag}_${reference}.log"
done

nao_walk="logs/contract/walk_nao_${tag}/generator.pt"
"$python" transfer/scripts/train_body_encoder.py "$walk" \
  --data "nao/rollouts/track_walk_${tag}_physical.npz" \
  --data "nao/rollouts/track_walk_${tag}_autonomous.npz" \
  --output "$nao_walk" --results "transfer/results/encoder_nao_${tag}.json"

export HUMANOID_CONTRACT="$root/$nao_walk"
capture --reference physical --output "nao/rollouts/track_walk_${tag}_encoder.npz" \
  > "logs/contract/capture_nao_walk_${tag}_encoder.log" 2>&1
tail -1 "logs/contract/capture_nao_walk_${tag}_encoder.log"

"$python" transfer/scripts/evaluate_contract.py \
  --run teacher=g1/rollouts/closed_loop_teacher.npz \
  --run "nao_${tag}_encoder=nao/rollouts/track_walk_${tag}_encoder.npz" \
  --run "nao_${tag}_truth=nao/rollouts/track_walk_${tag}_physical.npz" \
  --run "nao_${tag}_autonomous=nao/rollouts/track_walk_${tag}_autonomous.npz" \
  --output "transfer/results/contract_nao_walk_${tag}.json"

# The published task on the NAO, with nothing of the NAO retrained.
nao_new="logs/contract/new_nao_${tag}/generator.pt"
"$python" transfer/scripts/body_encoder.py "$new" "$nao_walk" --output "$nao_new"
scale="$("$python" -c "
import math
g, g1, nao = 9.81, 0.6545728842979989, 0.24801
v = math.sqrt(g * nao) / math.sqrt(g * g1)
w = math.sqrt(g1 / g) / math.sqrt(nao / g)
for name, (vx, vy, wz) in (('forward', (0.25, 0, 0)), ('fast', (0.6, 0, 0)),
                           ('turning', (0.3, 0.2, 0.3))):
    print(name, f'{vx * v:.4f}', f'{vy * v:.4f}', f'{wz * w:.4f}')
")"
export HUMANOID_CONTRACT="$root/$nao_new"
while read -r name vx vy wz; do
  for path in encoder truth; do
    extra=()
    if [ "$path" = truth ]; then extra=(--contract-from-truth); fi
    capture --reference physical "${extra[@]}" --command "$vx" "$vy" "$wz" \
      --output "nao/rollouts/track_new_${tag}_${name}_${path}.npz" \
      > "logs/contract/capture_nao_new_${tag}_${name}_${path}.log" 2>&1
    tail -1 "logs/contract/capture_nao_new_${tag}_${name}_${path}.log"
  done
  "$python" transfer/scripts/evaluate_contract.py \
    --run "teacher=g1/rollouts/flat_play_${name}.npz" \
    --run "encoder=nao/rollouts/track_new_${tag}_${name}_encoder.npz" \
    --run "truth=nao/rollouts/track_new_${tag}_${name}_truth.npz" \
    --output "transfer/results/contract_nao_new_task_${tag}_${name}.json"
done <<< "$scale"
