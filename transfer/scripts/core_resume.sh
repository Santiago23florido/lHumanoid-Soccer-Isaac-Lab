#!/usr/bin/env bash
# Step 2 resumed from its gated distillation (transfer/docs/skill_space.md §9.1): the core's
# PPO, then its ladder, as core_and_nao.sh would have run them.
#
#   transfer/scripts/core_resume.sh <distilled run directory>
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/core_and_nao.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
distilled="$1"
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
note "core resumed from $distilled"
"$python" -u scripts/train.py --task G1TrackCore-v0 --headless --num_envs 4096 \
  --max_iterations "${WARM_ITERATIONS:-1000}" --seed 41 --run_name warm_core \
  --resume --load_run "$(basename "$distilled")" --checkpoint model_0.pt \
  "agent.algorithm.learning_rate=${WARM_LR:-5.0625e-05}" \
  > logs/contract/train_warm_core.log 2>&1 || { note "warm_core training FAILED"; exit 1; }
note "warm_core trained"
run="$(ls -d logs/rsl_rl/g1_track_core/*_warm_core | tail -1)"
checkpoint="$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID=G1TrackCore-Play-v0
for task in rough flat; do
  if [ "$task" = rough ]; then generator=rough_b4; stages=(map corners encoder walk l4); else
    generator=flat_dagger2; stages=(l4); fi
  export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
  bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "model_${checkpoint}" warm_core \
    "${stages[@]}" > "logs/contract/ladder_warm_core_${task}.log" 2>&1 \
    || note "warm_core $task ladder FAILED"
  note "evaluated warm_core $task"
done
note "core resumed done"
