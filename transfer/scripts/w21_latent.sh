#!/usr/bin/env bash
# W21 of skill_space.md §10.22: the multi-body generator with a learned latent, from free to
# collapsed through a bottleneck on its history summary, deployed on the best decoders so far.
#
#   0. correction  mt17_l (W17's learned latent, refused by a leakage gate since corrected) on
#                  rough's L4 on W13's trackers, beside mt17_m's existing rows
#   1. generators  W18's recipe without sidestep (rough is never in any): for each beta, the
#                  experts first (mt21_<arm>_c), then the bodies' histories (mt21_<arm>), with
#                  --summary-beta: b1 0.001, b2 0.01, z 1.0
#   2. leakage     w21_leakage.py on every arm and on c and m -> transfer/results/w21_leakage.json;
#                  an arm that fails is not deployed
#   3. decoders    w21_compare.py --decoders: W20's xs or W19's xf, per body
#   4. L4          on the chosen decoders: sidestep under each new arm that passes; rough under
#                  c, m and each new arm that passes
#   5. compare     w21_compare.py -> transfer/results/w21_latent.json
#
# One Isaac process at a time; every stage is skipped when its output exists. It starts when W20
# is done and always ends with "w21 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w21_latent.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

declare -A w13=(
  [g1_41]=logs/rsl_rl/g1_track/2026-11-06_23-26-29_mt_g1_s41
  [g1_42]=logs/rsl_rl/g1_track/2026-11-07_01-30-35_mt_g1_s42
  [g1_43]=logs/rsl_rl/g1_track/2026-11-07_03-30-28_mt_g1_s43
  [nao_31]=logs/rsl_rl/nao_track/2026-11-07_05-30-28_mt_nao_s31
  [nao_32]=logs/rsl_rl/nao_track/2026-11-07_08-28-04_mt_nao_s32
  [nao_33]=logs/rsl_rl/nao_track/2026-11-07_11-24-19_mt_nao_s33
)
declare -A seeds=([g1]="41 42 43" [nao]="31 32 33")
declare -A walk=([g1]=logs/contract/walk_dagger3e/generator.pt [nao]=logs/contract/walk_dagger2e/generator.pt)
declare -A expert=(
  [walk]=g1/rollouts/teacher_dart.npz
  [flat]=g1/rollouts/flat_dart.npz
  [backward]=g1/rollouts/base_backward_dart.npz
  [jog]=g1/rollouts/base_jog_dart.npz
)
declare -A beta=([b1]=0.001 [b2]=0.01 [z]=1.0)
recipe="--encoder-from logs/contract/walk_zs_g1_r3_6194_sym/generator.pt --sigma-from-body \
--warp-loads --solver midpoint --flow-steps 8 --horizon 20 --commit 13 --noise-conditional \
--gait g1/rollouts/teacher.npz --relabel-span 50"
sidestep="left 0 0.25 0
left_fast 0 0.45 0
right 0 -0.35 0"

latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
isaac_busy() {  # no brackets or backslashes in the pattern (bash -> PowerShell quoting)
  local busy
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [ "$busy" != 0 ]
}
w20_done() {  # in the current run only
  awk '/^w20 start/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' \
    logs/contract/w20_from_s.status 2>/dev/null | grep -q "^w20 done"
}
conditions() {  # <body>: the evaluation conditions of W16-W20
  export HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0 HUMANOID_PROCEDURAL=0.7
  unset HUMANOID_TASKS HUMANOID_TASK_WEIGHTS HUMANOID_PLAN_CONTACT HUMANOID_OBSERVE_UNITS
  if [ "$1" = g1 ]; then
    export HUMANOID_CONTRACT="$root/${walk[g1]}" HUMANOID_LOAD_SPACE=log
    unset HUMANOID_POSTURE_CONTRACT
  else
    export HUMANOID_CONTRACT="$root/${walk[nao]}" HUMANOID_LOAD_SPACE=linear
    export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
  fi
}

# <name> <extra arguments...>: every base expert but sidestep's, W18's recipe
train() {
  local name="$1" data="" task
  shift
  [ -f "logs/contract/${name}/generator.pt" ] && return 0
  for task in walk flat backward jog; do data="$data --data ${expert[$task]}"; done
  note "$name training start"
  # shellcheck disable=SC2086
  "$python" -u transfer/scripts/train_contract.py --name "$name" $data $recipe --epochs 200 "$@" \
    --results "$results/contract_${name}.json" > "logs/contract/train_${name}.log" 2>&1 \
    || { note "$name training FAILED"; return 1; }
  note "$name: $(grep -E '^\[leakage\]|^gate|chain fidelity' "logs/contract/train_${name}.log" | tr '\n' ' ')"
}

# <body> <run> <checkpoint> <tag> <generator> <zs task> <commands> <last command>
ladder() {
  local body="$1" run="$2" checkpoint="$3" tag="$4" name="$5" task="$6" commands="$7" last="$8"
  [ -f "$results/zero_shot_${body}_${tag}_${name}_own_auto_kin_l4_${last}.json" ] && return 0
  while isaac_busy; do sleep 60; done
  ( conditions "$body"
    ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK="$task" \
    ZS_COMMANDS="$commands" ZS_GENERATOR="logs/contract/${name}/generator.pt" ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_${name}_${task}.log" 2>&1 \
    || { note "$tag $name $task FAILED"; return 1; }
  note "$task $tag $name"
}

until w20_done; do sleep 120; done
note "w21 start"

# 0. The correction: W17's learned latent on rough, on W13's trackers.
for body in g1 nao; do
  checkpoint=model_7193
  [ "$body" = nao ] && checkpoint=model_4497
  for seed in ${seeds[$body]}; do
    ladder "$body" "${w13[${body}_${seed}]}" "$checkpoint" "mt_${body}_s${seed}_sym" mt17_l rough "" turning
  done
done

# 1. The generators.
bodies=""
for f in g1/rollouts/w16_mt_g1_s41_sym_*.npz nao/rollouts/w16_mt_nao_s31_sym_*.npz; do
  [ -f "$f" ] && bodies="$bodies --body-data $f"
done
[ -n "$bodies" ] || { note "no body capture FAILED"; note "w21 done"; exit 1; }
for arm in b1 b2 z; do
  while isaac_busy; do sleep 60; done
  # shellcheck disable=SC2086
  train "mt21_${arm}_c" --summary-beta "${beta[$arm]}" $bodies --body-epochs 0 || continue
  # shellcheck disable=SC2086
  train "mt21_${arm}" --summary-beta "${beta[$arm]}" --init-from "logs/contract/mt21_${arm}_c/generator.pt" \
    $bodies --body-targets teacher --pair-command target
done

# 2. Leakage beyond the z* history, every arm on one set of pairs (read once: a resumed run keeps
# the first reading).
if [ ! -f "$results/w21_leakage.json" ]; then
  "$python" transfer/scripts/w21_leakage.py --device cuda --generators mt18_no_sidestep \
    mt18_no_sidestep_m mt21_b1 mt21_b2 mt21_z mt21_b1_c mt21_b2_c mt21_z_c \
    --output "$results/w21_leakage.json" > logs/contract/w21_leakage.out 2>&1 \
    || note "w21 leakage FAILED"
  note "leakage: $(grep -E 'passes|FAILS' logs/contract/w21_leakage.out | awk '{print $1, $NF}' | tr '\n' ' ')"
fi
deployed=""
for arm in b1 b2 z; do
  grep -qE "^mt21_${arm} .*passes$" logs/contract/w21_leakage.out && deployed="$deployed mt21_${arm}"
done
note "deployed:${deployed:- none}"

# 3. The decoders.
"$python" transfer/scripts/w21_compare.py --decoders > logs/contract/w21_decoders.out 2>&1 \
  || { note "decoders FAILED"; note "w21 done"; exit 1; }
note "decoders: $(tr '\n' ' ' < logs/contract/w21_decoders.out)"

# 4. L4 on the chosen decoders (read first: nothing in the loop may consume the file). Python
# writes CRLF on Windows, and a stray carriage return would end up in the run's name.
mapfile -t chosen < <(tr -d '\r' < logs/contract/w21_decoders.out)
for line in "${chosen[@]}"; do
  read -r body arm <<< "$line"
  for seed in ${seeds[$body]}; do
    run="$(ls -d logs/rsl_rl/"${body}"_track/*_"${arm}_${body}_s${seed}" 2>/dev/null | tail -1)"
    [ -n "$run" ] || { note "${arm}_${body}_s${seed}: no run FAILED"; continue; }
    tag="${arm}_${body}_s${seed}_sym"
    for name in $deployed; do
      ladder "$body" "$run" "$(latest "$run")" "$tag" "$name" base_sidestep "$sidestep" right
    done
    for name in mt18_no_sidestep mt18_no_sidestep_m $deployed; do
      ladder "$body" "$run" "$(latest "$run")" "$tag" "$name" rough "" turning
    done
  done
done

# 5. Compare.
"$python" transfer/scripts/w21_compare.py --output "$results/w21_latent.json" \
  > logs/contract/w21_compare.out 2>&1 || note "w21 compare FAILED"
note "w21 reading: $(tail -1 logs/contract/w21_compare.out)"
note "w21 done"
