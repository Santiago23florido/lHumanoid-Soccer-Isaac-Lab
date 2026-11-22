#!/usr/bin/env bash
# The held-out task on every body: W20's deployment (zero_shot_ladder.sh, L4: autonomous
# reference, kinematic estimator) driven once by the body's final teacher and once by a shared
# model in its place, through the same state estimate and the same generators.
#
#   PACT_ROOT=<repo> bash <snapshot copy> <model | -> <tag> ["<bodies>"]
#
# Per body (transfer/bodies/teachers.json, body_conditions.py; default every registered body):
# * the teacher (tag teacher_<body>, symmetrised as the registry says): its encoder stage first,
#   which refits the body's load estimate on the teacher's own states
#   (logs/contract/walk_zs_<body>_teacher_<body>), the estimate both drivers deploy through;
# * the model (tag <tag>_<body>, ZS_PACT_MODEL; the teacher stays the labeller), unless "-".
# Tasks, every one in the tracker's own units (ZS_OBSERVE=own, W14): the base tasks the shared
# model saw in training (flat at forward/fast/turning, jog at mid) and the held-out one
# (sidestep left / left_fast / right under mt18_no_sidestep and its _m variant, generators that
# never saw it). Commands are the G1's, Froude-scaled to the body by the ladder.
# Results: transfer/results/zero_shot_<body>_<tag>_<generator>_own_auto_kin_l4_<command>.json,
# read by icml_held_out_read.py. Status: logs/pact/held_out.status
# PACT_HELD_OUT_TEACHER: the teacher's tag stem (default "teacher"; a smoke run uses its own, so
# its tiny estimate is never deployed by a real run); PACT_HELD_OUT_QUICK=1: one command of one
# base task and of one held-out generator only (smoke). PACT_HELD_OUT_H2=1: H2's closed loop
# instead (prereg G-P, 2026-11-22 07:14): sidestep under mt18_no_sidestep_s0 as drawn and with
# its plans projected (HUMANOID_PLAN_CONTACT=stick, tags <tag>_stick), both drivers.
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
model="$1"
pact_tag="$2"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
bodies="${3:-$("$python" transfer/scripts/body_conditions.py --bodies)}"
teacher="${PACT_HELD_OUT_TEACHER:-teacher}"
quick="${PACT_HELD_OUT_QUICK:-0}"
h2="${PACT_HELD_OUT_H2:-0}"
status=logs/pact/held_out.status
mkdir -p logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

sidestep="left 0 0.25 0
left_fast 0 0.45 0
right 0 -0.35 0"

conditions() {  # <body>: W20's evaluation conditions and the body's own (teachers.json)
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7
  unset HUMANOID_TASKS HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS \
    HUMANOID_PROCEDURAL_DUTY
  eval "$("$python" transfer/scripts/body_conditions.py "$1" --prefix teacher_)"
}

teacher_args() {  # <body>: the ladder's run and checkpoint of the body's teacher
  ( eval "$("$python" transfer/scripts/body_conditions.py "$1" --prefix t_ --check)" \
      && echo "$t_run $t_checkpoint" )
}

# <body> <tag> <model or ""> <stage> <task> <generator> <commands> <last command> [plan contact]
ladder() {
  local body="$1" tag="$2" with="$3" stage="$4" task="$5" generator="$6" commands="$7" last="$8"
  local contact="${9:-}"
  local result="transfer/results/zero_shot_${body}_${tag}_${generator}_own_auto_kin_l4_${last}.json"
  [ "$stage" = l4 ] && [ -f "$result" ] && return 0
  [ "$stage" = encoder ] && [ -f "logs/contract/walk_zs_${body}_${teacher}_${body}/generator.pt" ] && return 0
  local args
  args="$(teacher_args "$body")" || { note "$body: no teacher ($args)"; return 1; }
  read -r run checkpoint <<< "$args"
  ( conditions "$body"
    export ZS_OBSERVE=own
    [ "$teacher_symmetrised" = 1 ] && export ZS_SYMMETRIZE=1
    [ -n "$with" ] && export ZS_PACT_MODEL="$with"
    [ -n "$contact" ] && export HUMANOID_PLAN_CONTACT="$contact"
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID="$teacher_play_task" \
      ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      ZS_COMMANDS="$commands" ZS_TEACHER_ROLLOUT=play ZS_ENCODER_TAG="${teacher}_${body}" \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" "$stage" ) \
    < /dev/null > "logs/pact/held_out_${tag}_${generator}_${task}_${stage}.log" 2>&1 \
    || { note "$tag $generator $task $stage FAILED"; return 1; }
  note "$tag $generator $task $stage done"
}

tasks() {  # <body> <tag> <model or "">: the base tasks and the held-out one
  if [ "$h2" = 1 ]; then
    ladder "$1" "$2" "$3" l4 base_sidestep mt18_no_sidestep_s0 "$sidestep" right
    ladder "$1" "${2}_stick" "$3" l4 base_sidestep mt18_no_sidestep_s0 "$sidestep" right stick
    return
  fi
  if [ "$quick" = 1 ]; then
    ladder "$1" "$2" "$3" l4 flat flat_dagger2 "forward 0.25 0 0" forward
    ladder "$1" "$2" "$3" l4 base_sidestep mt18_no_sidestep "left 0 0.25 0" left
    return
  fi
  ladder "$1" "$2" "$3" l4 flat flat_dagger2 "forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0.2 0.3" turning
  ladder "$1" "$2" "$3" l4 base_jog base_jog_b4 "mid 1.25 0 0" mid
  for generator in mt18_no_sidestep mt18_no_sidestep_m; do
    ladder "$1" "$2" "$3" l4 base_sidestep "$generator" "$sidestep" right
  done
}

note "start model=$model tag=$pact_tag bodies=$bodies"
for body in $bodies; do
  # The estimate both drivers deploy through, refitted once on the teacher's own states.
  ladder "$body" "${teacher}_${body}" "" encoder flat flat_dagger2 "" "" || continue
  tasks "$body" "${teacher}_${body}" ""
  [ "$model" = - ] || tasks "$body" "${pact_tag}_${body}" "$model"
done
note "done"
