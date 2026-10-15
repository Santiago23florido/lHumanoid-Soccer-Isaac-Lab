#!/usr/bin/env bash
# The blind test of the zero-shot study (transfer/docs/skill_space.md §8.4), end to end and once:
# Isaac Lab's published rough-terrain G1 policy, on a plane, reaching both frozen bodies.
#
#   transfer/scripts/blind_test.sh [stage...]
#
# Stages, all by default, in order (a stage list resumes a run that stopped, without
# repeating what it already wrote):
#   capture  the policy exported and its rollouts on the G1 (held_out_task.sh export capture)
#   train    its contract generator on the G1's frozen encoder -> logs/contract/rough_h20
#   dagger1  task-side DAgger round 1 on the G1, its frozen tracker as deployed -> rough_dagger1
#   dagger2  round 2 -> rough_dagger2
#   g1, nao  the amended L4 on each frozen body, deployed symmetrized
# The recipe is the published flat task's, fixed before any of this policy's data existed.
# Nothing of either body is trained. Progress: logs/contract/blind_test.status.
set -euo pipefail

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then stages=(capture train dagger1 dagger2 g1 nao); fi
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
want() { [[ " ${stages[*]} " == *" $1 "* ]]; }
status=logs/contract/blind_test.status
g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
g1_checkpoint=model_6194
g1_contract=logs/contract/walk_zs_g1_r3_6194_sym/generator.pt
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
nao_checkpoint=model_3498
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }

note "start: ${stages[*]}"
if want capture; then
  bash transfer/scripts/held_out_task.sh "$g1_contract" export capture >> "$status" 2>&1
  note "rollouts captured"
fi
if want train; then
  bash transfer/scripts/held_out_task.sh "$g1_contract" train >> "$status" 2>&1
  note "generator rough_h20 trained"
fi

export DAGGER_SYMMETRIZE=1
if want dagger1; then
  bash transfer/scripts/task_dagger.sh rough "$g1_run" "$g1_checkpoint" "$g1_contract" 1 \
    rough_dagger1 >> "$status" 2>&1
  note "task DAgger round 1"
fi
if want dagger2; then
  TASK_GENERATOR=logs/contract/rough_dagger1/generator.pt bash transfer/scripts/task_dagger.sh \
    rough "$g1_run" "$g1_checkpoint" "$g1_contract" 2 rough_dagger2 \
    g1/rollouts/rough_dagger_1.npz >> "$status" 2>&1
  note "task DAgger round 2"
fi
unset DAGGER_SYMMETRIZE

export ZS_TASK=rough ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1
export ZS_GENERATOR=logs/contract/rough_dagger2/generator.pt
if want g1; then
  bash transfer/scripts/zero_shot_ladder.sh g1 "$g1_run" "$g1_checkpoint" r3_6194_sym l4 \
    > logs/contract/blind_test_g1.log 2>&1
  note "L4 on the G1: logs/contract/blind_test_g1.log"
fi
if want nao; then
  bash transfer/scripts/zero_shot_ladder.sh nao "$nao_run" "$nao_checkpoint" r1_3498_sym l4 \
    > logs/contract/blind_test_nao.log 2>&1
  note "L4 on the NAO: logs/contract/blind_test_nao.log"
fi
note "done"
