#!/usr/bin/env bash
# W17: the command a future achieved (latent/hindsight.py), with rough
# held out of everything, on both bodies' frozen W13 decoders.
#
#   1. mt17_c      the G1 experts' five base tasks, every command relabelled (span 50)
#   2. pre-check   mt17_c chained on its own plans from rough's histories (held_out_chains.py):
#                  speed ratio 0.9-1.1 at every command, mean largest below mt16_c's; a gate
#                  before any Isaac run
#   3. mt17_m/l    from mt17_c: + the bodies' histories (W16's step-4 captures) with the
#                  teacher's phase-matched futures and their achieved commands; and the same
#                  with the history summary held to carry no body by an MMD constraint
#   4. rough       L4 on the six W13 trackers, pinned units, the leakage gate first for mt17_l;
#                  W16's mt16_c and rough_b4 captures are reused
#   5. chains      every arm chained offline on rough's histories, reported
#   6. compare     w17_compare.py -> transfer/results/w17_relabel.json
#
# One Isaac process at a time; every stage is skipped when its output exists. It always ends
# with "w17 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w17_relabel.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

declare -A runs=(
  [mt_g1_s41_sym]=logs/rsl_rl/g1_track/2026-11-06_23-26-29_mt_g1_s41
  [mt_g1_s42_sym]=logs/rsl_rl/g1_track/2026-11-07_01-30-35_mt_g1_s42
  [mt_g1_s43_sym]=logs/rsl_rl/g1_track/2026-11-07_03-30-28_mt_g1_s43
  [mt_nao_s31_sym]=logs/rsl_rl/nao_track/2026-11-07_05-30-28_mt_nao_s31
  [mt_nao_s32_sym]=logs/rsl_rl/nao_track/2026-11-07_08-28-04_mt_nao_s32
  [mt_nao_s33_sym]=logs/rsl_rl/nao_track/2026-11-07_11-24-19_mt_nao_s33
)
declare -A checkpoint=([g1]=model_7193 [nao]=model_4497)
declare -A walk=([g1]=logs/contract/walk_dagger3e/generator.pt [nao]=logs/contract/walk_dagger2e/generator.pt)
experts="g1/rollouts/teacher_dart.npz g1/rollouts/flat_dart.npz g1/rollouts/base_backward_dart.npz \
g1/rollouts/base_sidestep_dart.npz g1/rollouts/base_jog_dart.npz"
recipe="--encoder-from logs/contract/walk_zs_g1_r3_6194_sym/generator.pt --sigma-from-body \
--warp-loads --solver midpoint --flow-steps 8 --horizon 20 --commit 13 --noise-conditional \
--gait g1/rollouts/teacher.npz --relabel-span 50"

body_of() { case "$1" in *g1*) echo g1 ;; *) echo nao ;; esac; }
conditions() {  # <body>: each body's evaluation conditions, as W16's
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
isaac_busy() {  # no brackets or backslashes in the pattern (bash -> PowerShell quoting)
  local busy
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [ "$busy" != 0 ]
}

# <arm> <extra arguments...>
train() {
  local arm="$1" data="" path
  shift
  [ -f "logs/contract/mt17_${arm}/generator.pt" ] && return 0
  for path in $experts; do data="$data --data $path"; done
  note "mt17_${arm} training start"
  # shellcheck disable=SC2086
  "$python" -u transfer/scripts/train_contract.py --name "mt17_${arm}" $data $recipe "$@" \
    --results "$results/contract_mt17_${arm}.json" > "logs/contract/train_mt17_${arm}.log" 2>&1 \
    || { note "mt17_${arm} training FAILED"; return 1; }
  note "mt17_${arm}: $(grep -E '^\[leakage\]|^gate|chain fidelity' "logs/contract/train_mt17_${arm}.log" | tr '\n' ' ')"
}

# <tag> <generator directory name>. Rough's L4 on the deployed path, pinned units.
rough() {
  local tag="$1" name="$2" body
  body="$(body_of "$tag")"
  [ -f "$results/zero_shot_${body}_${tag}_${name}_own_auto_kin_l4_turning.json" ] && return 0
  while isaac_busy; do sleep 60; done
  ( conditions "$body"
    ZS_OBSERVE=own ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK=rough \
    ZS_GENERATOR="logs/contract/${name}/generator.pt" ZS_TEACHER_ROLLOUT=play \
    bash transfer/scripts/zero_shot_ladder.sh "$body" "${runs[$tag]}" "${checkpoint[$body]}" "$tag" l4 ) \
    < /dev/null > "logs/contract/ladder_${tag}_${name}_rough.log" 2>&1 \
    || { note "$tag $name rough FAILED"; return 1; }
  note "rough $tag $name"
}

note "w17 start"
bodies=""
for f in g1/rollouts/w16_mt_g1_s41_sym_*.npz nao/rollouts/w16_mt_nao_s31_sym_*.npz; do
  [ -f "$f" ] && bodies="$bodies --body-data $f"
done
[ -n "$bodies" ] || { note "no body capture FAILED"; note "w17 done"; exit 1; }
note "body captures: $(echo $bodies | grep -o 'body-data' | wc -l)"

# shellcheck disable=SC2086
train c --epochs 200 $bodies --body-epochs 0 || { note "w17 done"; exit 1; }

pre="$results/held_out_chains_w17_pre.json"
[ -f "$pre" ] || "$python" transfer/scripts/held_out_chains.py --generators mt17_c --output "$pre" \
  > logs/contract/w17_precheck.log 2>&1 || { note "pre-check FAILED"; note "w17 done"; exit 1; }
if "$python" transfer/scripts/w17_compare.py --precheck "$pre" > logs/contract/w17_precheck.out 2>&1; then
  note "pre-check: $(tail -1 logs/contract/w17_precheck.out)"
else
  note "pre-check: $(tail -1 logs/contract/w17_precheck.out); no L4"
  note "w17 done"
  exit 0
fi

# shellcheck disable=SC2086
train m --epochs 200 --init-from logs/contract/mt17_c/generator.pt $bodies --body-targets teacher \
  --pair-command target
# shellcheck disable=SC2086
train l --epochs 200 --init-from logs/contract/mt17_c/generator.pt $bodies --body-targets teacher \
  --pair-command target --invariance-kind mmd --invariance 1.0

tags="mt_g1_s41_sym mt_g1_s42_sym mt_g1_s43_sym mt_nao_s31_sym mt_nao_s32_sym mt_nao_s33_sym"
for name in mt17_c mt17_m; do
  [ -f "logs/contract/${name}/generator.pt" ] || continue
  for tag in $tags; do rough "$tag" "$name"; done
done
if "$python" transfer/scripts/w17_compare.py --leakage-gate > logs/contract/w17_gate.out 2>&1; then
  note "mt17_l leakage gate: $(tail -1 logs/contract/w17_gate.out)"
  for tag in $tags; do rough "$tag" mt17_l; done
else
  note "mt17_l leakage gate: $(tail -1 logs/contract/w17_gate.out); not evaluated"
fi

arms=""
for name in mt17_c mt17_m mt17_l; do [ -f "logs/contract/${name}/generator.pt" ] && arms="$arms $name"; done
# shellcheck disable=SC2086
"$python" transfer/scripts/held_out_chains.py --generators $arms \
  --output "$results/held_out_chains_w17.json" > logs/contract/w17_chains.log 2>&1 \
  || note "chains FAILED"
"$python" transfer/scripts/w17_compare.py --output "$results/w17_relabel.json" \
  > logs/contract/w17_compare.out 2>&1 || note "w17 compare FAILED"
note "w17 reading: $(tail -1 logs/contract/w17_compare.out)"
note "w17 done"
