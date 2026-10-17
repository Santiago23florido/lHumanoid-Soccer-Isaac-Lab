#!/usr/bin/env bash
# Decoder v2 at equal competence (transfer/docs/skill_space.md §9.1): does the physics prior help a
# tracker that already walks? Both arms start from the incumbent and get the same PPO budget.
#
#   warm_v2t  the incumbent distilled into the physics task (distill_tracker.py: DAgger on its
#             joint targets and values), then PPO;
#   warm_r3   the incumbent itself, PPO continued in the plain task.
#
# Same conditions as every tracker of the study (the walk's contract, 0.7 procedural, mirror loss,
# autonomous reference, seed 41); then the ladder for both, unsymmetrized alike, with the
# incumbent generators (rough_b4, flat_dagger2) and the kinematic estimator.
#
#   transfer/scripts/warm_study.sh          (from the repository root, under Start-Process)
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/warm_study.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
iterations="${WARM_ITERATIONS:-1000}"
incumbent=2026-10-14_15-38-39_zs_g1_r3_loss
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss

note "warm study start"
# 1. Distil.
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 "$python" -u transfer/scripts/distill_tracker.py --headless \
  --teacher "logs/rsl_rl/g1_track/${incumbent}/exported/model_6194_sym.pt" \
  --teacher-checkpoint "logs/rsl_rl/g1_track/${incumbent}/model_6194.pt" \
  --run_name distill_v2t > logs/contract/distill_v2t.log 2>&1
distilled="$(ls -d logs/rsl_rl/g1_track_physics/*_distill_v2t 2>/dev/null | tail -1)"
if [ -z "$distilled" ] || [ ! -f "$distilled/model_0.pt" ]; then note "distillation FAILED"; exit 1; fi
note "distilled: $(tail -3 logs/contract/distill_v2t.log | head -1)"

# 2. PPO from the distilled physics tracker.
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 "$python" -u scripts/train.py --task G1TrackPhysics-v0 \
  --headless --num_envs 4096 --max_iterations "$iterations" --seed 41 --run_name warm_v2t \
  --resume --load_run "$(basename "$distilled")" --checkpoint model_0.pt \
  > logs/contract/train_warm_v2t.log 2>&1 || note "warm_v2t training FAILED"
note "warm_v2t trained"

# 3. PPO continued on the incumbent itself.
HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 "$python" -u scripts/train.py --task G1Track-v0 \
  --headless --num_envs 4096 --max_iterations "$iterations" --seed 41 --run_name warm_r3 \
  --resume --load_run "$incumbent" --checkpoint model_6194.pt \
  > logs/contract/train_warm_r3.log 2>&1 || note "warm_r3 training FAILED"
note "warm_r3 trained"

# 4. The ladder for both.
export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic
for arm in warm_v2t warm_r3; do
  if [ "$arm" = warm_v2t ]; then
    run="$(ls -d logs/rsl_rl/g1_track_physics/*_warm_v2t | tail -1)"
    export ZS_TASK_ID=G1TrackPhysics-Play-v0 HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1
  else
    run="$(ls -d logs/rsl_rl/g1_track/*_warm_r3 | tail -1)"
    export ZS_TASK_ID=G1Track-Play-v0 HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  fi
  checkpoint="$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
  for task in rough flat; do
    if [ "$task" = rough ]; then generator=rough_b4; stages=(map corners encoder walk l4); else
      generator=flat_dagger2; stages=(l4); fi
    export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "model_${checkpoint}" "$arm" "${stages[@]}" \
      > "logs/contract/ladder_${arm}_${task}.log" 2>&1 || note "${arm} ${task} ladder FAILED"
    note "evaluate ${arm} ${task}"
  done
done
note "warm study done"
