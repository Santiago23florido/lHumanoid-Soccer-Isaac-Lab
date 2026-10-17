#!/usr/bin/env bash
# Decoder v2 at equal competence, continued (transfer/docs/skill_space.md §9.1):
#
#   step 2  the deep-learning core on the G1 (G1TrackCore-v0: the plan's temporal encoder, a
#           residual trunk, a proprioceptive history) against the warm study's control warm_r3;
#   step 4  the physics decoder on the NAO (NaoTrackPhysics-v0) against the NAO's own control,
#           the incumbent r1 with its PPO continued in its plain task.
#
# Every learner arm starts from its body's incumbent the same way (warm_gated.sh): the
# incumbent distilled into the arm's task (its own last action, labels cut where the drives
# saturate, the prior's feed-forward compensated, rollouts with PPO's noise), a gate -- the
# student may not end more episodes than the incumbent, both with PPO's noise, by more than two
# standard errors -- then 1000 PPO iterations under the incumbent's own conditions, then the
# ladder, unsymmetrized like the controls. On the NAO the incumbent is first measured in the
# physics task itself, deterministic (the check that found the distillation's fault on the G1).
# An arm that fails stops alone; the next goes on.
#
#   transfer/scripts/core_and_nao.sh [arms]     (default "core nao_check nao_v2t nao_r1")
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/core_and_nao.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
arms="${1:-core nao_check nao_v2t nao_r1}"
iterations="${WARM_ITERATIONS:-1000}"
want() { [[ " $arms " == *" $1 "* ]]; }

g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
g1_teacher=(--teacher "$g1_run/exported/model_6194_sym.pt"
  --teacher-checkpoint "$g1_run/model_6194.pt" --teacher-scale 0.5)
nao_teacher=(--teacher "$nao_run/exported/model_3498_sym.pt"
  --teacher-checkpoint "$nao_run/model_3498.pt" --teacher-scale 0.25)

g1_conditions() {
  export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
  unset HUMANOID_POSTURE_CONTRACT
  export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss
}
nao_conditions() {
  export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
  export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss
}

# distil_and_gate <task> <experiment> <name> <teacher args...>: prints the distilled run.
distil_and_gate() {
  local task="$1" experiment="$2" name="$3"
  shift 3
  "$python" -u transfer/scripts/distill_tracker.py --headless --task "$task" "$@" \
    --run_name "distill_$name" > "logs/contract/distill_$name.log" 2>&1
  local distilled
  distilled="$(ls -d "logs/rsl_rl/$experiment/"*"_distill_$name" 2>/dev/null | tail -1)"
  if [ -z "$distilled" ] || [ ! -f "$distilled/model_0.pt" ]; then
    note "$name: distillation FAILED" >&2
    return 1
  fi
  note "$name: distilled $distilled $(grep '^round' "logs/contract/distill_$name.log" | tail -1)" >&2
  local check="transfer/results/teacher_in_physics_student_$name.json"
  "$python" -u transfer/scripts/teacher_in_physics.py --headless --task "$task" "$@" \
    --checkpoint "$distilled/model_0.pt" --stochastic --steps 600 \
    --modes teacher_compensated student --output "$check" \
    > "logs/contract/teacher_in_physics_student_$name.log" 2>&1
  local verdict
  verdict="$("$python" - "$check" <<'PYEOF'
import json, math, sys
modes = json.load(open(sys.argv[1]))["modes"]
t, s = modes["teacher_compensated"]["terminated"], modes["student"]["terminated"]
se = math.sqrt(t + s)
print(f"{'pass' if s - t <= 2 * se else 'FAIL'} teacher {t} student {s} (2 SE {2 * se:.0f})")
PYEOF
)"
  note "$name: gate $verdict" >&2
  case "$verdict" in pass*) echo "$distilled" ;; *) return 1 ;; esac
}

# ladder <body> <task id> <run> <tag> <rough stages...>: rough, then flat's L4.
ladder() {
  local body="$1" task_id="$2" run="$3" tag="$4"
  shift 4
  local checkpoint
  checkpoint="$(ls "$run" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
  export ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID="$task_id"
  for task in rough flat; do
    local generator stages
    if [ "$task" = rough ]; then generator=rough_b4; stages=("$@"); else
      generator=flat_dagger2; stages=(l4); fi
    export ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "model_${checkpoint}" "$tag" \
      "${stages[@]}" > "logs/contract/ladder_${tag}_${task}.log" 2>&1 \
      || note "$tag $task ladder FAILED"
    note "evaluated $tag $task"
  done
  unset ZS_TASK_ID
}

note "core and nao start: $arms"

if want core; then
  g1_conditions
  export HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  if distilled="$(distil_and_gate G1TrackCore-v0 g1_track_core core "${g1_teacher[@]}")"; then
    "$python" -u scripts/train.py --task G1TrackCore-v0 --headless --num_envs 4096 \
      --max_iterations "$iterations" --seed 41 --run_name warm_core \
      --resume --load_run "$(basename "$distilled")" --checkpoint model_0.pt \
      > logs/contract/train_warm_core.log 2>&1 || note "warm_core training FAILED"
    note "warm_core trained"
    run="$(ls -d logs/rsl_rl/g1_track_core/*_warm_core | tail -1)"
    ladder g1 G1TrackCore-Play-v0 "$run" warm_core map corners encoder walk l4
  fi
fi

if want nao_check; then
  nao_conditions
  export HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1
  "$python" -u transfer/scripts/teacher_in_physics.py --headless --task NaoTrackPhysics-v0 \
    "${nao_teacher[@]}" --modes teacher teacher_compensated --steps 600 \
    --output transfer/results/teacher_in_physics_nao.json \
    > logs/contract/teacher_in_physics_nao.log 2>&1 || note "nao check FAILED"
  HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 "$python" -u transfer/scripts/teacher_in_physics.py \
    --headless --task NaoTrack-v0 "${nao_teacher[@]}" --modes teacher --steps 600 \
    --output transfer/results/teacher_in_plain_nao.json \
    > logs/contract/teacher_in_plain_nao.log 2>&1 || note "nao plain check FAILED"
  note "nao check: physics $(grep -aE '^teacher' logs/contract/teacher_in_physics_nao.log | tr '\n' ' ') plain $(grep -aE '^teacher' logs/contract/teacher_in_plain_nao.log | tr '\n' ' ')"
fi

if want nao_v2t; then
  nao_conditions
  export HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1
  if distilled="$(distil_and_gate NaoTrackPhysics-v0 nao_track_physics nao_v2t "${nao_teacher[@]}")"; then
    HUMANOID_TARGET_RATE=1 "$python" -u scripts/train.py --task NaoTrackPhysics-v0 --headless \
      --num_envs 4096 --max_iterations "$iterations" --seed 31 --run_name warm_nao_v2t \
      --resume --load_run "$(basename "$distilled")" --checkpoint model_0.pt \
      > logs/contract/train_warm_nao_v2t.log 2>&1 || note "warm_nao_v2t training FAILED"
    note "warm_nao_v2t trained"
    run="$(ls -d logs/rsl_rl/nao_track_physics/*_warm_nao_v2t | tail -1)"
    ladder nao NaoTrackPhysics-Play-v0 "$run" warm_nao_v2t encoder walk l4
  fi
fi

if want nao_r1; then
  nao_conditions
  export HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
  "$python" -u scripts/train.py --task NaoTrack-v0 --headless --num_envs 4096 \
    --max_iterations "$iterations" --seed 31 --run_name warm_nao_r1 \
    --resume --load_run "$(basename "$nao_run")" --checkpoint model_3498.pt \
    > logs/contract/train_warm_nao_r1.log 2>&1 || note "warm_nao_r1 training FAILED"
  note "warm_nao_r1 trained"
  run="$(ls -d logs/rsl_rl/nao_track/*_warm_nao_r1 | tail -1)"
  ladder nao NaoTrack-Play-v0 "$run" warm_nao_r1 encoder walk l4
fi

note "core and nao done"
