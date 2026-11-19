#!/usr/bin/env bash
# The held-out task with PACT driving: W20's deployment (zero_shot_ladder.sh, L4), once with
# the tracker PACT imitated (G-D1's teacher, not symmetrised) and once with PACT in its place,
# through the same refitted state estimate (W20's, ZS_ENCODER_TAG) and the same generators.
#
#   PACT_ROOT=<repo> bash <snapshot copy> <pact model> <pact tag> ["g1 nao"]
#
# Tasks: the base tasks PACT saw in training (flat at its three commands, jog at "mid") and
# the held-out one (sidestep, left / left_fast / right, under mt18_no_sidestep and its _m
# variant, generators that never saw it). Results:
# transfer/results/zero_shot_<body>_<tag>_<generator>[_own]_auto_kin_l4_<command>.json, tags
# xs_<body>_s<seed>_plain (tracker) and <pact tag>_<body> (PACT). Status:
# logs/pact/held_out.status
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
model="$1"
pact_tag="$2"
bodies="${3:-g1 nao}"
status=logs/pact/held_out.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }

declare -A run=(
  [g1]=logs/rsl_rl/g1_track/2026-11-08_18-22-20_xs_g1_s41
  [nao]=logs/rsl_rl/nao_track/2026-11-08_21-02-51_xs_nao_s31
)
declare -A checkpoint=([g1]=model_7193 [nao]=model_4497)
declare -A encoder=([g1]=xs_g1_s41_sym [nao]=xs_nao_s31_sym)
declare -A play_id=([g1]=G1Track-Play-v0 [nao]=NaoTrack-Play-v0)
declare -A observe=([g1]=task [nao]=own)  # the base tasks' units, as W20
sidestep="left 0 0.25 0
left_fast 0 0.45 0
right 0 -0.35 0"

conditions() {  # <body>: W20's evaluation conditions
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7
  unset HUMANOID_TASKS HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}

# <body> <tag> <pact model or ""> <units> <task> <generator> <commands> <last command>
ladder() {
  local body="$1" tag="$2" with="$3" units="$4" task="$5" generator="$6" commands="$7" last="$8"
  local suffix=""
  [ "$units" = own ] && suffix=_own
  local result="transfer/results/zero_shot_${body}_${tag}_${generator}${suffix}_auto_kin_l4_${last}.json"
  [ -f "$result" ] && return 0
  ( conditions "$body"
    [ "$units" = own ] && export ZS_OBSERVE=own
    [ -n "$with" ] && export ZS_PACT_MODEL="$with"
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_TASK_ID="${play_id[$body]}" \
      ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      ZS_COMMANDS="$commands" ZS_TEACHER_ROLLOUT=play ZS_ENCODER_TAG="${encoder[$body]}" \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "${run[$body]}" "${checkpoint[$body]}" \
      "$tag" l4 ) < /dev/null > "logs/pact/held_out_${tag}_${generator}${suffix}_${task}.log" 2>&1 \
    || { note "$tag $generator $task FAILED"; return 1; }
  note "$tag $generator$suffix $task: $(ls transfer/results/zero_shot_${body}_${tag}_${generator}${suffix}_auto_kin_l4_*.json 2>/dev/null | wc -l) results"
}

note "start model=$model tag=$pact_tag bodies=$bodies"
for body in $bodies; do
  for driver in tracker pact; do
    if [ "$driver" = tracker ]; then
      tag="xs_${body}_s$([ "$body" = g1 ] && echo 41 || echo 31)_plain" with=""
    else
      tag="${pact_tag}_${body}" with="$model"
    fi
    ladder "$body" "$tag" "$with" "${observe[$body]}" flat flat_dagger2 \
      "forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0.2 0.3" turning
    ladder "$body" "$tag" "$with" "${observe[$body]}" base_jog base_jog_b4 "mid 1.25 0 0" mid
    for generator in mt18_no_sidestep mt18_no_sidestep_m; do
      ladder "$body" "$tag" "$with" own base_sidestep "$generator" "$sidestep" right
    done
  done
done
note "done"
