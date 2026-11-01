#!/usr/bin/env bash
# The distillation's teacher, measured where it gives its labels (skill_space.md §9.1): the
# incumbent's terminations per 1000 steps, deterministic, in its own task and in the physics
# task, shown the residual as the first two distillations did or its own last action.
#
#   bash transfer/scripts/teacher_view_study.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
incumbent=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss
teacher=(--teacher "$incumbent/exported/model_6194_sym.pt" --teacher-checkpoint "$incumbent/model_6194.pt")
modes=(--modes teacher teacher_clamped teacher_saturated --steps 600 --num_envs 4096)
HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 "$python" -u transfer/scripts/teacher_in_physics.py --headless \
  --task G1Track-v0 "${teacher[@]}" "${modes[@]}" \
  --output transfer/results/teacher_in_plain.json > logs/contract/teacher_in_plain.log 2>&1
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 "$python" -u transfer/scripts/teacher_in_physics.py --headless \
  "${teacher[@]}" "${modes[@]}" --residual-view \
  --output transfer/results/teacher_in_physics_residual_view.json > logs/contract/teacher_in_physics_residual.log 2>&1
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 "$python" -u transfer/scripts/teacher_in_physics.py --headless \
  "${teacher[@]}" "${modes[@]}" \
  --output transfer/results/teacher_in_physics_own_view.json > logs/contract/teacher_in_physics_own.log 2>&1
echo done
