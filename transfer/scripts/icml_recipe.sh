#!/usr/bin/env bash
# One shared arm, one seed, through the whole recipe every arm gets (prereg, "Final data: fresh
# per arm, not relabelled", 2026-11-22):
#
#   PACT_ROOT=<repo> bash <snapshot copy> <arm> <seed> [stage...]
#
# arm: pact_d, pact_na, pact_nb, urma_d, hpt_d (pact_gate_d.py). Stages, in order, all by
# default; each one is skipped when its output exists:
#   offline   from scratch on the base data of every body -- its final teacher's audit captures,
#             the probe map (<body>_<variant>_probes.npz) and the training task
#             (<body>_<variant>_train.npz) -- walking probes only, 24000 steps
#   probes    four DAgger rounds on the probe map (training 8000 steps after rounds 1-3)
#   train     three rounds in the training task, on the walking family (training after each)
#   score     the final model on the probe map, read against each teacher's own map score
#             (G-D1 criterion, walking probes) -> transfer/results/icml_final_<name>_d1.json
#   held_out  the held-out task on every body against its teacher (pact_held_out.sh)
#             -> transfer/results/icml_held_out_icml_<name>.json
#   clean     the run's captures archived as summaries and removed (pact_archive_captures.py)
# name = <arm>_s<seed>. Work files: logs/icml/<name>/; final model: logs/icml/models/<name>.pt;
# status: logs/icml/<name>.status. Offline training stages are pure PyTorch; the others run
# Isaac (one process at a time).
#
# ICML_SMOKE=1 runs every stage at minimal size under the name smoke_<arm>_s<seed> (two bodies,
# one round of each kind, 50 training steps, short captures, one held-out command), so a code
# path that would fail does so in minutes; its results are written under logs/icml/ only.
# ICML_BODIES overrides the bodies (default: every body in transfer/bodies/teachers.json).
set -uo pipefail
root="${PACT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
arm="$1"
seed="$2"
shift 2
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(offline probes train score held_out clean)
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
want() { [[ " ${stages[*]} " == *" $1 "* ]]; }

smoke="${ICML_SMOKE:-0}"
if [ "$smoke" = 1 ]; then
  name="smoke_${arm}_s${seed}"
  bodies="${ICML_BODIES:-g1 digit}"
  offline_steps=50 dagger_steps=50 probe_rounds=1 train_rounds=1
  export PACT_DAGGER_ENVS=71 PACT_DAGGER_CAPTURE_STEPS=120 PACT_DAGGER_SETTLE=20
  export ZS_STEPS=120 ZS_SETTLE=20 PACT_HELD_OUT_QUICK=1 PACT_HELD_OUT_TEACHER=smoke_teacher
  results_final="logs/icml/smoke_results"
else
  name="${arm}_s${seed}"
  bodies="${ICML_BODIES:-$("$python" transfer/scripts/body_conditions.py --bodies)}"
  offline_steps=24000 dagger_steps=8000 probe_rounds=4 train_rounds=3
  results_final="transfer/results"
fi
dir="logs/icml/$name"
models="$dir/models"
final_model="logs/icml/models/${name}.pt"
status="logs/icml/${name}.status"
mkdir -p "$dir/results" "$models" logs/icml/models "$results_final"
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" >> "$status"; }
csv="$(echo $bodies | tr ' ' ',')"
# The base data, read through each body's final teacher (pact_gate_d.py fills {variant}).
base_map="logs/pact/teachers/{body}_{variant}_probes.npz"
base_train="logs/pact/teachers/{body}_{variant}_train.npz"
teachers_score="transfer/results/icml_teacher_{body}_{variant}_probes.json"

dagger() {  # <start model> <rounds> <tag: probes|train> [first round]
  PACT_DAGGER_ARM="$arm" PACT_DAGGER_SEED="$seed" PACT_DAGGER_STEPS="$dagger_steps" \
    PACT_DAGGER_CAPTURES="$base_map" PACT_DAGGER_MIN_DUTY=0.55 \
    PACT_DAGGER_GLOB="$base_train;$dir/dagger/*/r*/{body}.npz" \
    PACT_DAGGER_DIR="$dir/dagger" PACT_DAGGER_RESULTS="$dir/results" PACT_DAGGER_MODELS="$models" \
    PACT_DAGGER_ENV="$([ "$3" = train ] && echo train || echo probes)" \
    PACT_DAGGER_TRAIN_LAST="$([ "$3" = train ] && echo 1 || echo 0)" \
    PACT_DAGGER_FIRST="${4:-1}" \
    bash transfer/scripts/pact_dagger.sh "$1" "$2" "$bodies" "$3" < /dev/null
}

note "start arm=$arm seed=$seed bodies=$bodies stages=${stages[*]} smoke=$smoke"
for body in $bodies; do
  "$python" transfer/scripts/body_conditions.py "$body" --check > /dev/null \
    || { note "$body: no final teacher; stopped"; exit 1; }
done

if want offline && [ ! -f "$models/r0.pt" ]; then
  "$python" -u transfer/scripts/pact_gate_d.py --bodies "$csv" --captures "$base_map" \
    --dagger "$base_train" --min-duty 0.55 --steps "$offline_steps" --arms "hold,$arm" \
    --seed "$seed" --save "$models/r0.pt" --output "$results_final/icml_final_${name}_offline.json" \
    > "$dir/offline.log" 2>&1 || { note "offline FAILED, see $dir/offline.log"; exit 1; }
  note "offline: R2 (torque) $("$python" -c "
import json, sys
d = json.load(open(sys.argv[1]))['bodies']
print({b: round(e[sys.argv[2]]['r2'], 3) for b, e in d.items() if sys.argv[2] in e})
" "$results_final/icml_final_${name}_offline.json" "$arm")"
fi

last_probe="$models/d1_probes_r$((probe_rounds - 1)).pt"
[ "$probe_rounds" = 1 ] && last_probe="$models/r0.pt"
if want probes && [ ! -f "$dir/results/icml_gate_d1_probes_r${probe_rounds}_${bodies##* }.json" ]; then
  dagger "$models/r0.pt" "$probe_rounds" probes || { note "probe rounds FAILED"; exit 1; }
  note "probe rounds done"
fi

trained="$models/d1_train_r${train_rounds}.pt"
if want train && [ ! -f "$trained" ]; then
  [ -f "$last_probe" ] || { note "no $last_probe; stopped"; exit 1; }
  dagger "$last_probe" "$train_rounds" train || { note "training-task rounds FAILED"; exit 1; }
  [ -f "$trained" ] || { note "no $trained after the training-task rounds; stopped"; exit 1; }
  cp "$trained" "$final_model"
  note "training-task rounds done: $final_model"
fi

score_round=$((probe_rounds + 1))
if want score && [ ! -f "$results_final/icml_final_${name}_d1.json" ]; then
  [ -f "$final_model" ] || { note "no $final_model; stopped"; exit 1; }
  dagger "$final_model" 1 probes "$score_round" || { note "score round FAILED"; exit 1; }
  # Only the score round is read: earlier rounds were driven by intermediate models.
  mkdir -p "$dir/score"
  cp "$dir/results/icml_gate_d1_probes_r${score_round}_"*.json "$dir/score/"
  "$python" transfer/scripts/pact_gate_d1_read.py --tag probes --results "$dir/score" \
    --min-duty 0.55 --teachers "$teachers_score" --output "$results_final/icml_final_${name}_d1.json" \
    > "$dir/score.log" 2>&1 || { note "score read FAILED, see $dir/score.log"; exit 1; }
  note "score: $(tail -3 "$dir/score.log" | tr '\n' ' ' | cut -c1-240)"
fi

if want held_out; then
  [ -f "$final_model" ] || { note "no $final_model; stopped"; exit 1; }
  bash transfer/scripts/pact_held_out.sh "$final_model" "icml_${name}" "$bodies" < /dev/null
  "$python" transfer/scripts/icml_held_out_read.py --tag "icml_${name}" --bodies "$csv" \
    --teacher-tag "${PACT_HELD_OUT_TEACHER:-teacher}" \
    --output "$results_final/icml_held_out_icml_${name}.json" > "$dir/held_out.log" 2>&1 \
    || { note "held-out read FAILED, see $dir/held_out.log"; exit 1; }
  note "held out: $(tail -1 "$dir/held_out.log" | cut -c1-200)"
fi

if want clean; then
  # Kept: the final model, every result JSON and the archive's summaries; removed: captures.
  patterns=("$dir/dagger/*/r*/*.npz")
  for body in $bodies; do
    rollouts="$("$python" transfer/scripts/body_conditions.py "$body" --prefix b_ | sed -n 's/^b_rollouts=//p')"
    patterns+=("$rollouts/zs_icml_${name}_${body}_*.npz")
  done
  "$python" transfer/scripts/pact_archive_captures.py "${patterns[@]}" \
    --output "$results_final/icml_capture_archive.json" --delete > "$dir/clean.log" 2>&1 \
    && note "clean: $(tail -1 "$dir/clean.log")" || note "clean FAILED, see $dir/clean.log"
fi
note "done"
