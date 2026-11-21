#!/usr/bin/env bash
# The teachers on the walking family: each body's tracker fine-tuned from its current
# checkpoint with its own training conditions (mirror loss, procedural share 0.5, the
# cross-fitted base tasks, its walk contract and load space), the procedural duty
# restricted to walks (HUMANOID_PROCEDURAL_DUTY, default 0.55,0.95), then exported and
# mirror-symmetrised for the audit (pact_teacher_audit.sh with variant "walk").
#
#   PACT_ROOT=<repo> bash <snapshot copy> ["g1 nao h1 cassie"] [iterations=1000]
#
# Runs: logs/rsl_rl/<body>_track/<stamp>_walkfam_<body>; policies:
# <run>/exported/model_<last>.pt and model_<last>_sym_pact.pt. Status: logs/pact/walk_teachers.status
# With HUMANOID_CURRICULUM=1 the procedural plans are drawn by the tracker's contact
# disagreement, with HUMANOID_CURRICULUM=bias by its signed contact error and with
# HUMANOID_CURRICULUM=gait by its largest relative duty or cadence error, half of them
# uniformly (latent/curriculum.py); each run's record goes to logs/pact/curriculum/<tag>_<body>.npz.
# HUMANOID_ENTROPY sets PPO's entropy coefficient (default 0.008; 0: no bonus, PPO's own gradient
# sets the action std).
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
bodies="${1:-g1 nao h1 cassie}"
iterations="${2:-1000}"
duty="${HUMANOID_PROCEDURAL_DUTY:-0.55,0.95}"
# PACT_TEACHER_TAG names the runs (<stamp>_<tag>_<body>); PACT_TEACHER_FROM, if set, starts
# from the latest <from>_<body> run instead of the trackers the co-executions used.
run_tag="${PACT_TEACHER_TAG:-walkfam}"
from_tag="${PACT_TEACHER_FROM:-}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/walk_teachers.status
mkdir -p logs/pact
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }
latest_run() { ls -d "$1"/*_"$2" 2>/dev/null | tail -1; }
latest_checkpoint() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}

conditions() {  # conditions <body>: the start run, task, contract and mirror of each tracker
  unset HUMANOID_POSTURE_CONTRACT HUMANOID_OBSERVE_UNITS HUMANOID_TASK_WEIGHTS \
    HUMANOID_PLAN_CONTACT HUMANOID_HEADS HUMANOID_PLAN_TRUST
  export HUMANOID_SYMMETRY=loss HUMANOID_PROCEDURAL=0.5 HUMANOID_PROCEDURAL_DUTY="$duty"
  export HUMANOID_TASKS="$root/transfer/results/base_tasks_xf.json"
  case "$1" in
    g1)
      start=logs/rsl_rl/g1_track/2026-11-08_18-22-20_xs_g1_s41 task=G1Track-v0 mirror=g1/body/mirror.json
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      export HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
      ;;
    nao)
      start=logs/rsl_rl/nao_track/2026-11-08_21-02-51_xs_nao_s31 task=NaoTrack-v0 mirror=nao/body/mirror.json
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
      export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT" HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
      ;;
    h1 | cassie | digit)
      start="$(latest_run "logs/rsl_rl/${1}_track" "pact_${1}_s41")"
      task="$(echo "${1:0:1}" | tr a-z A-Z)${1:1}Track-v0" mirror="transfer/bodies/$1/mirror.json"
      export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_LOAD_SPACE=log
      ;;
    *) return 1 ;;
  esac
}

note "start bodies=$bodies iterations=$iterations duty=$duty tag=$run_tag from=${from_tag:-trackers} contact=${HUMANOID_CONTACT_REWARD:-0} curriculum=${HUMANOID_CURRICULUM:-0} entropy=${HUMANOID_ENTROPY:-0.008}"
for body in $bodies; do
  ( conditions "$body" || exit 1
    name="${run_tag}_${body}"
    if [ "${HUMANOID_CURRICULUM:-0}" != 0 ]; then
      mkdir -p logs/pact/curriculum
      export HUMANOID_CURRICULUM_LOG="$root/logs/pact/curriculum/${name}.npz"
    fi
    if [ -n "$from_tag" ]; then
      start="$(latest_run "logs/rsl_rl/${body}_track" "${from_tag}_${body}")"
      [ -n "$start" ] || { note "$name: no ${from_tag}_${body} run to start from"; exit 1; }
    fi
    if ! ls -d logs/rsl_rl/"${body}"_track/*_"$name" > /dev/null 2>&1; then
      checkpoint="$(latest_checkpoint "$start")"
      note "$name: from $start/$checkpoint"
      "$python" -u scripts/train.py --task "$task" --headless --num_envs 4096 \
        --max_iterations "$iterations" --seed 41 --run_name "$name" --resume \
        --load_run "$(basename "$start")" --checkpoint "$checkpoint.pt" \
        "agent.algorithm.learning_rate=$(rate "$start/$checkpoint.pt")" \
        < /dev/null > "logs/pact/train_${name}.log" 2>&1 || { note "$name training FAILED"; exit 1; }
    fi
    run="$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" | tail -1)"
    last="$(latest_checkpoint "$run")"
    "$python" transfer/scripts/export_actor.py "$run/$last.pt" > /dev/null \
      || { note "$name export FAILED"; exit 1; }
    kind="$body"; [ "$body" = g1 ] && kind=g1
    "$python" transfer/scripts/symmetrize_actor.py "$run/exported/$last.pt" --mirror "$mirror" \
      --contract "$HUMANOID_CONTRACT" --body "$kind" --output "$run/exported/${last}_sym_pact.pt" \
      > "logs/pact/sym_${name}.log" 2>&1 || { note "$name symmetrise FAILED"; exit 1; }
    note "$name trained: $run/$last ($(tail -1 "logs/pact/sym_${name}.log" | cut -c1-80))"
  )
done
note "done"
