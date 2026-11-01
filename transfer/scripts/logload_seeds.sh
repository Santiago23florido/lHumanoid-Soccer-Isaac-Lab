#!/usr/bin/env bash
# Seeds for the log-load decision (transfer/docs/skill_space.md §9.1), on the user's choice of
# 2026-10-25: the pre-registered rule adopts a tracker that masters more of the family's probes
# by over two standard errors, and one seed of each arm gave +1 probe (+3 unsymmetrized) with the
# duty six SE better. Seed 41 of both arms exists (warm_r3_r8, warm_r3_logload); this adds 42
# and 43 of each, each the incumbent continued exactly as there (its adapted rate, 1000
# iterations), then the map and the corners. compare_seeds.py decides.
#
#   bash transfer/scripts/logload_seeds.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/logload_seeds.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
incumbent=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
rate="$("$python" -c "
import torch
print(torch.load('$incumbent/model_6194.pt', map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])")"
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID=G1Track-Play-v0
export ZS_TASK=rough ZS_GENERATOR=logs/contract/rough_b4/generator.pt
note "log-load seeds: start (rate $rate)"
for seed in 42 43; do
  for space in linear log; do
    tag="warm_r3_s${seed}_${space}"
    HUMANOID_LOAD_SPACE="$space" "$python" -u scripts/train.py --task G1Track-v0 --headless \
      --num_envs 4096 --max_iterations 1000 --seed "$seed" --run_name "$tag" \
      --resume --load_run "$(basename "$incumbent")" --checkpoint model_6194.pt \
      "agent.algorithm.learning_rate=$rate" > "logs/contract/train_${tag}.log" 2>&1 \
      || { note "$tag training FAILED"; continue; }
    note "$tag trained"
    run="$(ls -d "logs/rsl_rl/g1_track/"*"_$tag" | tail -1)"
    checkpoint="model_$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
    bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "$checkpoint" "$tag" map corners \
      > "logs/contract/ladder_${tag}.log" 2>&1 || note "$tag ladder FAILED"
    note "evaluated $tag"
  done
done
note "log-load seeds: done"
