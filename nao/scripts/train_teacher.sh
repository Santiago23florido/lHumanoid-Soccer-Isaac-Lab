#!/usr/bin/env bash
# The teacher: the robot-trained tracker told each robot's parameters (NaoTrack-Sim2RealTeacher-v0).
#
#   bash nao/scripts/train_teacher.sh [iterations] [name] [from]
#        (defaults: 1000, s2r_teacher, logs/rsl_rl/nao_track/2026-11-14_15-54-50_s2r_r1/model_4997.pt)
#
# 1. widen  -- [from] given zero weights on robot_params (5 inputs appended to the actor and
#              to the critic, transfer/scripts/widen_inputs.py): at iteration 0 it acts
#              exactly as [from] does.
# 2. train  -- PPO on the teacher's task with the plans, contract reward, mirror loss and
#              robot model [from] was trained with, from its own learning rate.
# Progress: logs/contract/train_<name>.log. The student is fitted to the result by
# nao/scripts/train_adaptation.py.
set -euo pipefail

iterations="${1:-1000}"
name="${2:-s2r_teacher}"
from="${3:-logs/rsl_rl/nao_track/2026-11-14_15-54-50_s2r_r1/model_4997.pt}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

checkpoint="$(basename "$from" .pt)"
start_name="$(date +%Y-%m-%d_%H-%M-%S)_${name}_start"
"$python" transfer/scripts/widen_inputs.py "$from" --actor 5 --critic 5 \
  --output "logs/rsl_rl/nao_track/$start_name/${checkpoint}.pt"
lr="$("$python" -c "
import torch
saved = torch.load('$from', map_location='cpu', weights_only=False)
print(saved['optimizer_state_dict']['param_groups'][0]['lr'])")"
lr="${lr//$'\r'/}"
echo "start ${start_name} lr ${lr}"

export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_PROCEDURAL=0.7
export HUMANOID_SYMMETRY=loss
export HUMANOID_SIM2REAL_HISTORY="${HUMANOID_SIM2REAL_HISTORY:-3}"
"$python" -u scripts/train.py --task NaoTrack-Sim2RealTeacher-v0 --headless --num_envs 4096 \
  --max_iterations "$iterations" --seed 37 --run_name "$name" --resume \
  --load_run "$start_name" --checkpoint "${checkpoint}.pt" \
  "agent.algorithm.learning_rate=${lr}" > "logs/contract/train_${name}.log" 2>&1
echo "trained $(date +%H:%M)"
