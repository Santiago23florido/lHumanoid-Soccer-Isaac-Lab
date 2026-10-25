#!/usr/bin/env bash
# The warm study's physics arm, gated (transfer/docs/skill_space.md §9.1): distil, then check
# that the student starts where the incumbent stands before spending PPO on it.
#
# 1. distil   -- distill_tracker.py (the teacher's own last action, labels cut where the drives
#                saturate, the prior's feed-forward compensated, rollouts with PPO's noise);
# 2. check    -- teacher_in_physics.py --stochastic: terminations per 1000 steps of the
#                incumbent (compensated) and of the student in the physics task, both with the
#                exploration noise PPO samples;
# 3. gate     -- the student may not end more episodes than the incumbent by more than two
#                standard errors of the difference (Poisson counts); if it does, stop;
# 4. PPO and the ladder -- warm_study.sh on that distillation (WARM_DISTILLED).
#
# The physics task's policy also sees the prior's feed-forward (HUMANOID_FEEDFORWARD_OBS=1, the
# default here, inherited by every stage): the labels are compensated for it, and a student
# that had to infer it ended 74 episodes with PPO's noise where the incumbent ended 45-48, while
# the same distillation into the plain task passed (57 against 53).
#
#   transfer/scripts/warm_gated.sh          (from the repository root, under Start-Process)
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/warm_study.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
incumbent=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss
export HUMANOID_FEEDFORWARD_OBS="${HUMANOID_FEEDFORWARD_OBS:-1}"
teacher=(--teacher "$incumbent/exported/model_6194_sym.pt"
  --teacher-checkpoint "$incumbent/model_6194.pt")

note "gated v2t start"
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 "$python" -u transfer/scripts/distill_tracker.py --headless \
  "${teacher[@]}" --run_name distill_v2t > logs/contract/distill_v2t.log 2>&1
distilled="$(ls -d logs/rsl_rl/g1_track_physics/*_distill_v2t 2>/dev/null | tail -1)"
if [ -z "$distilled" ] || [ ! -f "$distilled/model_0.pt" ]; then note "distillation FAILED"; exit 1; fi
note "distilled: $distilled $(grep '^round' logs/contract/distill_v2t.log | tail -1)"

check=transfer/results/teacher_in_physics_student_v2t.json
HUMANOID_HEADS=1 HUMANOID_PLAN_TRUST=1 "$python" -u transfer/scripts/teacher_in_physics.py --headless \
  "${teacher[@]}" --checkpoint "$distilled/model_0.pt" --stochastic --steps 600 \
  --modes teacher_compensated student --output "$check" > logs/contract/teacher_in_physics_student.log 2>&1
verdict="$("$python" - "$check" <<'PYEOF'
import json, math, sys
modes = json.load(open(sys.argv[1]))["modes"]
t, s = modes["teacher_compensated"]["terminated"], modes["student"]["terminated"]
se = math.sqrt(t + s)
print(f"{'pass' if s - t <= 2 * se else 'FAIL'} teacher {t} student {s} (2 SE {2 * se:.0f})")
PYEOF
)"
note "gate: $verdict"
case "$verdict" in pass*) ;; *) note "stopped: the student does not start where the incumbent stands"; exit 1 ;; esac

WARM_ARMS=v2t WARM_DISTILLED="$distilled" bash transfer/scripts/warm_study.sh
note "gated v2t done"
