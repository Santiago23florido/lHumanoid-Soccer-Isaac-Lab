#!/usr/bin/env bash
# Step 4 resumed after the asset server's outage of 2026-10-22 (transfer/docs/skill_space.md
# §9.1): the NAO's physics arm from its distillation (the gate, then PPO and the ladder), then
# its control, the incumbent's PPO continued in its plain task -- as core_and_nao.sh runs them.
#
#   transfer/scripts/nao_resume.sh <distilled run directory> [arms]   (default "nao_v2t nao_r1")
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/core_and_nao.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
distilled="$1"
arms="${2:-nao_v2t nao_r1}"
want() { [[ " $arms " == *" $1 "* ]]; }
iterations="${WARM_ITERATIONS:-1000}"
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
nao_teacher=(--teacher "$nao_run/exported/model_3498_sym.pt"
  --teacher-checkpoint "$nao_run/model_3498.pt" --teacher-scale 0.25)
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss

ladder() {  # <task id> <run> <tag>
  local task_id="$1" run="$2" tag="$3" checkpoint
  checkpoint="$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
  export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID="$task_id"
  for task in rough flat; do
    local generator stages
    if [ "$task" = rough ]; then generator=rough_b4; stages=(encoder walk l4); else
      generator=flat_dagger2; stages=(l4); fi
    export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh nao "$run" "model_${checkpoint}" "$tag" \
      "${stages[@]}" > "logs/contract/ladder_${tag}_${task}.log" 2>&1 \
      || note "$tag $task ladder FAILED"
    note "evaluated $tag $task"
  done
  unset ZS_TASK_ID
}

if want nao_v2t; then
  export HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1
  check=transfer/results/teacher_in_physics_student_nao_v2t.json
  "$python" -u transfer/scripts/teacher_in_physics.py --headless --task NaoTrackPhysics-v0 \
    "${nao_teacher[@]}" --checkpoint "$distilled/model_0.pt" --stochastic --steps 600 \
    --modes teacher_compensated student --output "$check" \
    > logs/contract/teacher_in_physics_student_nao_v2t.log 2>&1
  verdict="$("$python" - "$check" <<'PYEOF'
import json, math, sys
modes = json.load(open(sys.argv[1]))["modes"]
t, s = modes["teacher_compensated"]["terminated"], modes["student"]["terminated"]
se = math.sqrt(t + s)
print(f"{'pass' if s - t <= 2 * se else 'FAIL'} teacher {t} student {s} (2 SE {2 * se:.0f})")
PYEOF
)"
  note "nao_v2t: gate $verdict"
  if [[ "$verdict" == pass* ]]; then
    HUMANOID_TARGET_RATE=1 "$python" -u scripts/train.py --task NaoTrackPhysics-v0 --headless \
      --num_envs 4096 --max_iterations "$iterations" --seed 31 --run_name warm_nao_v2t \
      --resume --load_run "$(basename "$distilled")" --checkpoint model_0.pt \
      > logs/contract/train_warm_nao_v2t.log 2>&1 || note "warm_nao_v2t training FAILED"
    note "warm_nao_v2t trained"
    ladder NaoTrackPhysics-Play-v0 "$(ls -d logs/rsl_rl/nao_track_physics/*_warm_nao_v2t | tail -1)" warm_nao_v2t
  fi
fi

if want nao_r1; then
  export HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  "$python" -u scripts/train.py --task NaoTrack-v0 --headless --num_envs 4096 \
    --max_iterations "$iterations" --seed 31 --run_name warm_nao_r1 \
    --resume --load_run "$(basename "$nao_run")" --checkpoint model_3498.pt \
    > logs/contract/train_warm_nao_r1.log 2>&1 || { note "warm_nao_r1 training FAILED"; exit 1; }
  note "warm_nao_r1 trained"
  ladder NaoTrack-Play-v0 "$(ls -d logs/rsl_rl/nao_track/*_warm_nao_r1 | tail -1)" warm_nao_r1
fi
note "nao resumed done"
