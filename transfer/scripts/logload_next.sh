#!/usr/bin/env bash
# After the loads scored in log space lifted the G1's limiting check (skill_space.md §9.1):
#
# 1. the NAO: its incumbent continued exactly as its control (warm_nao_r1: its adapted rate,
#    1000 iterations, seed 31) with only the load's space changed -- the NAO's rough stance is
#    the one shortfall of the deployed transfer;
# 2. the G1's adoption check: the log-load tracker deployed symmetrized, as the incumbent is,
#    against the incumbent's own map and corners remeasured under this code.
#
#   bash transfer/scripts/logload_next.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/logload_next.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
ladder() {  # <body> <run> <checkpoint> <tag> <rough stages...>
  local body="$1" run="$2" checkpoint="$3" tag="$4"
  shift 4
  for task in rough flat; do
    local generator stages
    if [ "$task" = rough ]; then generator=rough_b4; stages=("$@"); else generator=flat_dagger2; stages=(l4); fi
    ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" "${stages[@]}" \
      > "logs/contract/ladder_${tag}_${task}.log" 2>&1 || note "$tag $task ladder FAILED"
    note "evaluated $tag $task"
  done
}
export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic HUMANOID_LOAD_SPACE=log
export HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss
note "log-load next: start"

# 1. The NAO.
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" \
  HUMANOID_POSTURE_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" \
  "$python" -u scripts/train.py --task NaoTrack-v0 --headless --num_envs 4096 \
  --max_iterations 1000 --seed 31 --run_name warm_nao_r1_logload \
  --resume --load_run "$(basename "$nao_run")" --checkpoint model_3498.pt \
  "agent.algorithm.learning_rate=$(rate "$nao_run/model_3498.pt")" \
  > logs/contract/train_warm_nao_r1_logload.log 2>&1 || note "warm_nao_r1_logload training FAILED"
note "warm_nao_r1_logload trained"
run="$(ls -d logs/rsl_rl/nao_track/*_warm_nao_r1_logload | tail -1)"
checkpoint="model_$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
ZS_TASK_ID=NaoTrack-Play-v0 ladder nao "$run" "$checkpoint" warm_nao_r1_logload encoder walk l4

# 2. The G1, symmetrized, against the incumbent remeasured.
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
run="$(ls -d logs/rsl_rl/g1_track/*_warm_r3_logload | tail -1)"
checkpoint="model_$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
ZS_TASK_ID=G1Track-Play-v0 ZS_SYMMETRIZE=1 ladder g1 "$run" "$checkpoint" warm_r3_logload_sym \
  map corners encoder walk l4
ZS_TASK_ID=G1Track-Play-v0 ZS_SYMMETRIZE=1 ZS_TASK=rough \
  ZS_GENERATOR=logs/contract/rough_b4/generator.pt bash transfer/scripts/zero_shot_ladder.sh g1 \
  logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss model_6194 r3_6194_sym_v4 map corners \
  > logs/contract/ladder_r3_6194_sym_v4_map.log 2>&1 || note "incumbent map FAILED"
note "evaluated the incumbent's map (r3_6194_sym_v4)"
note "log-load next: done"
