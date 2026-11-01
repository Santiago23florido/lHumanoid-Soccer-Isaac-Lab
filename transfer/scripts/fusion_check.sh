#!/usr/bin/env bash
# Does shrinking the plan towards the body's state shorten the physics arm's steps?
# (transfer/docs/skill_space.md §9.1). Round 8's trained physics arm, deployed on the walk as S0
# measures it (the walk's encoder, physical reference, kinematic estimator), with the prior's
# plan fused each way: shrink (as trained) and the two samples' mean. No retraining.
#
#   bash transfer/scripts/fusion_check.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
cd "$(cd "$(dirname "$0")/../.." && pwd)"
run="$(ls -d logs/rsl_rl/g1_track_physics/*_warm_v2t_r8 | tail -1)"
policy="$(ls "$run"/exported/model_*.pt | grep -v _sym | sort -V | tail -1)"
export HUMANOID_CONTRACT="$PWD/logs/contract/walk_zs_g1_warm_v2t_r8/generator.pt"
export HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 HUMANOID_FEEDFORWARD_OBS=1
for fusion in shrink mean; do
  out="g1/rollouts/fusion_${fusion}_walk_kin_encoder.npz"
  HUMANOID_PLAN_FUSION="$fusion" "$python" -u transfer/scripts/capture_contract.py --headless \
    --task-id G1TrackPhysics-Play-v0 --driver-policy "$policy" --steps 900 --settle-steps 100 \
    --num-envs 32 --reference physical --estimator kinematic --output "$out" \
    > "logs/contract/fusion_${fusion}_walk.log" 2>&1
  "$python" transfer/scripts/evaluate_contract.py --run teacher=g1/rollouts/closed_loop_teacher.npz \
    --run "encoder=$out" --output "transfer/results/fusion_check_${fusion}_walk.json" | grep -v "^  [^a]"
done
