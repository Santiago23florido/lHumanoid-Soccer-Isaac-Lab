#!/usr/bin/env bash
# The warm study's mirror-loss puzzle (transfer/docs/skill_space.md §9.1): the distilled physics
# trackers start PPO far from their mirror image (0.67 in round 6, 0.12 in round 3, 0.05 for
# the incumbent). mirror_error.py measures it on each, joint by joint and term by term.
#
#   bash transfer/scripts/mirror_study.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss
physics=logs/rsl_rl/g1_track_physics
HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 "$python" -u transfer/scripts/mirror_error.py --headless \
  --task G1Track-v0 --checkpoint logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss/model_6194.pt \
  --output transfer/results/mirror_error_incumbent.json > logs/contract/mirror_error_incumbent.log 2>&1
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 HUMANOID_FEEDFORWARD_OBS=1 "$python" -u transfer/scripts/mirror_error.py \
  --headless --task G1TrackPhysics-v0 --checkpoint "$physics/2026-10-22_21-29-34_distill_v2t/model_0.pt" \
  --output transfer/results/mirror_error_v2t_round6.json > logs/contract/mirror_error_round6.log 2>&1
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 HUMANOID_FEEDFORWARD_OBS=0 "$python" -u transfer/scripts/mirror_error.py \
  --headless --task G1TrackPhysics-v0 \
  --checkpoint "$physics/2026-10-17_22-13-25_distill_v2t_superseded/model_0.pt" \
  --output transfer/results/mirror_error_v2t_round3.json > logs/contract/mirror_error_round3.log 2>&1
for f in incumbent round6 round3; do echo "== $f"; grep -aE "^\{|^worst|^terms" "logs/contract/mirror_error_$f.log"; done
