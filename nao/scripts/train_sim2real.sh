#!/usr/bin/env bash
# The adopted NAO tracker, continued for the real robot (NaoTrack-Sim2Real-v0).
#
#   bash nao/scripts/train_sim2real.sh [iterations] [name]      (defaults: 1000, s2r_r1)
#
# 1. widen  -- the adopted tracker (zs_nao_r1, model_3498) given zero weights on the
#              appended history inputs (transfer/scripts/widen_inputs.py): at iteration 0
#              it acts exactly as the published tracker does.
# 2. train  -- PPO on the robot model, with the plans, contract reward and mirror loss the
#              tracker was trained with (70 % procedural plans, the walk's contract), from
#              the checkpoint's own learning rate (rsl-rl restarts its schedule at the
#              config's 1e-3 on resume).
# Progress: logs/contract/train_<name>.log.
set -euo pipefail

iterations="${1:-1000}"
name="${2:-s2r_r1}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

start_run="logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1"
checkpoint="model_3498"
history="${HUMANOID_SIM2REAL_HISTORY:-3}"
# Per step of history: gyroscope 3, gravity 3, and the 19 actuated joints' positions,
# velocities and last actions.
extra=$((history * 63))
start_name="$(date +%Y-%m-%d_%H-%M-%S)_${name}_start"
"$python" transfer/scripts/widen_inputs.py "$start_run/${checkpoint}.pt" \
  --actor "$extra" --critic 0 --output "logs/rsl_rl/nao_track/$start_name/${checkpoint}.pt"
lr="$("$python" -c "
import torch
saved = torch.load('$start_run/${checkpoint}.pt', map_location='cpu', weights_only=False)
print(saved['optimizer_state_dict']['param_groups'][0]['lr'])")"
lr="${lr//$'\r'/}"
echo "start ${start_name} lr ${lr}"

export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_PROCEDURAL=0.7
export HUMANOID_SYMMETRY=loss
export HUMANOID_SIM2REAL_HISTORY="$history"
"$python" -u scripts/train.py --task NaoTrack-Sim2Real-v0 --headless --num_envs 4096 \
  --max_iterations "$iterations" --seed 31 --run_name "$name" --resume \
  --load_run "$start_name" --checkpoint "${checkpoint}.pt" \
  "agent.algorithm.learning_rate=${lr}" > "logs/contract/train_${name}.log" 2>&1
echo "trained $(date +%H:%M)"
