#!/usr/bin/env bash
# The blind task of the zero-shot study: Isaac Lab's published rough-terrain G1 policy, on a plane.
#
#   transfer/scripts/held_out_task.sh <g1 body contract> [stage...]
#
# Stages (all by default), run only once both trackers are frozen (skill_space.md §8.3):
#   export   download and script Isaac-Velocity-Rough-G1-v0 (logs/published/...)
#   capture  its rollouts on G1RoughOnFlat: DART with starts and stops for the generator, and
#            play rollouts under the three test commands (its command has no sideways part, so
#            the turning one is 0.3 0 0.3)
#   train    its contract generator, the published flat policy's recipe (horizon 20, commit 13),
#            on the G1 body's encoder: logs/contract/rough_h20
# Nothing of either body is trained here; zero_shot_ladder.sh with ZS_TASK=rough tests them.
set -euo pipefail

body_contract="$1"
shift
stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then stages=(export capture train); fi
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
want() { [[ " ${stages[*]} " == *" $1 "* ]]; }
published="logs/published/Isaac-Velocity-Rough-G1-v0/policy.pt"
capture() { "$python" -u g1/scripts/capture_rollouts.py --headless "$@"; }

if want export; then
  "$python" g1/scripts/export_published.py --task Isaac-Velocity-Rough-G1-v0
fi
if want capture; then
  capture --task-id G1RoughOnFlat-v0 --checkpoint "$published" --num-envs 128 \
    --steps 1000 --action-noise 0.1 --standing-fraction 0.3 --resample-seconds 2 6 \
    --output g1/rollouts/rough_dart.npz > logs/contract/capture_rough_dart.log 2>&1
  tail -1 logs/contract/capture_rough_dart.log
  while read -r name vx vy wz; do
    capture --task-id G1RoughOnFlat-Play-v0 --checkpoint "$published" --num-envs 32 \
      --steps 900 --settle-steps 100 --command "$vx" "$vy" "$wz" \
      --output "g1/rollouts/rough_play_${name}.npz" > "logs/contract/capture_rough_${name}.log" 2>&1
    tail -1 "logs/contract/capture_rough_${name}.log"
  done <<< "forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0 0.3"
fi
if want train; then
  "$python" -u transfer/scripts/train_contract.py --name rough_h20 \
    --data g1/rollouts/rough_dart.npz --gait g1/rollouts/rough_play_forward.npz \
    --encoder-from "$body_contract" --warp-loads --solver midpoint --flow-steps 8 \
    --horizon 20 --commit 13 --results transfer/results/contract_rough_h20.json \
    > logs/contract/train_rough_h20.log 2>&1
  tail -4 logs/contract/train_rough_h20.log
fi
