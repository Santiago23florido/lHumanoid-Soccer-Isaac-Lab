#!/usr/bin/env bash
# W16 of skill_space.md §10.17: steps 4-6 of method_revision.md §6. The multi-task generator,
# with and without the bodies' executions, and rough -- held out of everything -- on both
# bodies' frozen W13 decoders.
#
#   4. bodies     each body's W13 decoder (G1 s41, NAO s31) executing walk, flat and jog at
#                 three commands each, on the deployed path, observing in its own units
#   5. generators one multi-task generator on the G1 experts' DART captures of the five base
#                 tasks (mt16_c), then from it: + the bodies' histories with the teacher's
#                 phase-matched futures (mt16_m), + the bodies' own futures (mt16_d), and as
#                 mt16_m with the history summary trained to carry no body (mt16_l)
#   6. rough      L4 on the six W13 trackers, pinned units, with each generator, the leakage
#                 gate first for mt16_l; and rough_b4 (a generator that saw rough) as reference
#   7. compare    w16_compare.py -> transfer/results/w16_multibody.json
#
# Nothing of rough enters steps 4-5. Waits for W15's "done". One Isaac process at a time;
# every stage is skipped when its output exists. It always ends with "w16 done".
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w16_multibody.status
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
declare -A generator=([walk]="" [flat]=flat_dagger2 [jog]=base_jog_b4)
# Each task's commands in the G1's units, inside its expert's DART range (walk: 0-0.25 m/s);
# the NAO's are scaled to its own, as the ladder does.
declare -A commands=(
  [walk]="slow 0.1 0 0;mid 0.18 0 0;forward 0.25 0 0"
  [flat]="forward 0.25 0 0;fast 0.6 0 0;turning 0.3 0.2 0.3"
  [jog]="slow 1.0 0 0;mid 1.25 0 0;fast 1.5 0 0"
)
experts="g1/rollouts/teacher_dart.npz g1/rollouts/flat_dart.npz g1/rollouts/base_backward_dart.npz \
g1/rollouts/base_sidestep_dart.npz g1/rollouts/base_jog_dart.npz"
recipe="--encoder-from logs/contract/walk_zs_g1_r3_6194_sym/generator.pt --sigma-from-body \
--warp-loads --solver midpoint --flow-steps 8 --horizon 20 --commit 13 --noise-conditional \
--gait g1/rollouts/teacher.npz"

body_of() { case "$1" in *g1*) echo g1 ;; *) echo nao ;; esac; }
conditions() {  # <body>: each body's evaluation conditions, as W13's ladder() and W14 give them
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
scale() {  # <body>: speed and turn scales from the G1's units to the body's
  if [ "$1" = g1 ]; then echo "1 1"; else "$python" -c "
import math
g1, nao = 0.6545728842979989, 0.24801
print(math.sqrt(nao / g1), math.sqrt(g1 / nao))"; fi
}

# Step 4: <tag> <task>. The body's tracker on the task's generator moved to its own encoder.
execute() {
  local tag="$1" task="$2" body speed turn line name vx vy wz contract out log attempt
  body="$(body_of "$tag")"
  read -r speed turn <<< "$(scale "$body")"
  contract="logs/contract/walk_zs_${body}_${tag}/generator.pt"
  [ -n "${generator[$task]}" ] && contract="logs/contract/${generator[$task]}_zs_${body}_${tag}/generator.pt"
  [ -f "$contract" ] || { note "$tag $task: no $contract FAILED"; return 1; }
  IFS=';' read -ra lines <<< "${commands[$task]}"
  for line in "${lines[@]}"; do
    read -r name vx vy wz <<< "$line"
    out="$body/rollouts/w16_${tag}_${task}_${name}.npz"
    [ -f "$out" ] && continue
    vx=$("$python" -c "print($vx * $speed)"); vy=$("$python" -c "print($vy * $speed)")
    wz=$("$python" -c "print($wz * $turn)")
    log="logs/contract/capture_w16_${tag}_${task}_${name}.log"
    while isaac_busy; do sleep 60; done
    for attempt in 1 2 3; do
      ( conditions "$body"
        export HUMANOID_CONTRACT="$root/$contract" HUMANOID_OBSERVE_UNITS="$root/${walk[$body]}"
        "$python" -u transfer/scripts/capture_contract.py --headless \
          --task-id "$( [ "$body" = g1 ] && echo G1Track-Play-v0 || echo NaoTrack-Play-v0 )" \
          --driver-policy "${runs[$tag]}/exported/${checkpoint[$body]}_sym.pt" --steps 900 \
          --settle-steps 100 --num-envs 64 --reference autonomous --estimator kinematic \
          --command "$vx" "$vy" "$wz" --output "$out" ) < /dev/null > "$log" 2>&1
      grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" \
        "$log" || break
      sleep 120
    done
    [ -f "$out" ] || { note "$tag $task $name capture FAILED"; return 1; }
  done
  note "executed $tag $task"
}

# Step 5: <arm> <extra arguments...>
train() {
  local arm="$1" data="" path
  shift
  [ -f "logs/contract/mt16_${arm}/generator.pt" ] && return 0
  for path in $experts; do data="$data --data $path"; done
  note "mt16_${arm} training start"
  # shellcheck disable=SC2086
  "$python" -u transfer/scripts/train_contract.py --name "mt16_${arm}" $data $recipe "$@" \
    --results "$results/contract_mt16_${arm}.json" > "logs/contract/train_mt16_${arm}.log" 2>&1 \
    || { note "mt16_${arm} training FAILED"; return 1; }
  note "mt16_${arm}: $(grep -E '^\[leakage\]|^gate|chain fidelity' "logs/contract/train_mt16_${arm}.log" | tr '\n' ' ')"
}

# Step 6: <tag> <generator directory name>. Rough's L4 on the deployed path, pinned units.
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

until grep -q "^w15 done" logs/contract/w15_family.status 2>/dev/null; do sleep 120; done
note "w16 start"

for tag in mt_g1_s41_sym mt_nao_s31_sym; do
  for task in walk flat jog; do execute "$tag" "$task"; done
done
bodies=""
for f in g1/rollouts/w16_mt_g1_s41_sym_*.npz nao/rollouts/w16_mt_nao_s31_sym_*.npz; do
  [ -f "$f" ] && bodies="$bodies --body-data $f"
done
[ -n "$bodies" ] || { note "no body capture FAILED"; note "w16 done"; exit 1; }
note "body captures: $(echo $bodies | grep -o 'body-data' | wc -l)"

# One base model on the experts; the arms differ only in what follows it. The epochs give it
# the per-task generators' number of steps on five times their data (1000 x 1 capture).
# shellcheck disable=SC2086
train c --epochs 200 $bodies --body-epochs 0 || { note "w16 done"; exit 1; }
# shellcheck disable=SC2086
train m --epochs 200 --init-from logs/contract/mt16_c/generator.pt $bodies --body-targets teacher
# shellcheck disable=SC2086
train l --epochs 200 --init-from logs/contract/mt16_c/generator.pt $bodies --body-targets teacher \
  --invariance 1.0
# shellcheck disable=SC2086
train d --epochs 200 --init-from logs/contract/mt16_c/generator.pt $bodies --body-targets own

tags="mt_g1_s41_sym mt_g1_s42_sym mt_g1_s43_sym mt_nao_s31_sym mt_nao_s32_sym mt_nao_s33_sym"
for name in mt16_c mt16_m; do
  [ -f "logs/contract/${name}/generator.pt" ] || continue
  for tag in $tags; do rough "$tag" "$name"; done
done
if "$python" transfer/scripts/w16_compare.py --leakage-gate > logs/contract/w16_gate.out 2>&1; then
  note "mt16_l leakage gate: $(tail -1 logs/contract/w16_gate.out)"
  for tag in $tags; do rough "$tag" mt16_l; done
else
  note "mt16_l leakage gate: $(tail -1 logs/contract/w16_gate.out); not evaluated"
fi
for name in mt16_d rough_b4; do
  [ -f "logs/contract/${name}/generator.pt" ] || continue
  for tag in $tags; do rough "$tag" "$name"; done
done
"$python" transfer/scripts/w16_compare.py --output "$results/w16_multibody.json" \
  > logs/contract/w16_compare.out 2>&1 || note "w16 compare FAILED"
note "w16 reading: $(tail -1 logs/contract/w16_compare.out)"
note "w16 done"
