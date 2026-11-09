#!/usr/bin/env bash
# W5b (skill_space.md §10.7): W5's perturbations on flat_b4, the flat generator trained as
# rough_b4 was (history noise across levels, read as an input), on both incumbents.
# W5 found the flat skill's incumbent generator, flat_dagger2, falling almost every episode
# with the estimate read at twice its precision, where rough_b4 did not fall at four times.
# Instrumentation: it decides nothing. Runs after W8; W6 is launched after it. Always ends
# with "w5b done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w5b_flat_b4.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

room() {
  local free gpu busy
  free=$(powershell -NoProfile -Command \
    "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  # No brackets or backslashes in the pattern (bash -> PowerShell quoting).
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [[ "$free" =~ ^[0-9]+$ && "$gpu" =~ ^[0-9]+$ && "$busy" =~ ^[0-9]+$ ]] || return 1
  [ "$free" -ge 10 ] && [ "$gpu" -le 2500 ] && [ "$busy" -eq 0 ]
}
wait_for_room() {
  local waited=0
  until room; do
    [ $waited -eq 0 ] && note "waiting for room before $1"
    waited=1
    sleep 120
  done
}
assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}

g1_run=logs/rsl_rl/g1_track/2026-10-24_17-31-00_warm_r3_logload
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1

# <body> <run dir> <checkpoint> <tag>
robust() {
  local body="$1" run="$2" checkpoint="$3" tag="$4" task_id=G1Track-Play-v0 rollouts=g1/rollouts
  local vx=0.25 policy="$run/exported/${checkpoint}_sym.pt"
  local contract="logs/contract/flat_b4_zs_${body}_${tag}/generator.pt"
  if [ "$body" = nao ]; then
    task_id=NaoTrack-Play-v0
    rollouts=nao/rollouts
    vx=$("$python" -c "import math; print(0.25 * math.sqrt(0.24801 / 0.6545728842979989))")
    export HUMANOID_POSTURE_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
  else
    unset HUMANOID_POSTURE_CONTRACT
  fi
  [ -f "$policy" ] || { note "$tag: no $policy FAILED"; return 1; }
  [ -f "$contract" ] || { note "$tag: no $contract FAILED"; return 1; }
  for condition in push10 push20 noise2 noise4; do
    local out="$results/zero_shot_${body}_${tag}_flat_b4_auto_kin_robust_${condition}_forward.json"
    [ -f "$out" ] && continue
    local option=()
    case "$condition" in
      push10) option=(--push 0.1 20) ;;
      push20) option=(--push 0.2 20) ;;
      noise2) option=(--estimate-noise 2) ;;
      noise4) option=(--estimate-noise 4) ;;
    esac
    wait_for_room "$tag flat_b4 $condition"
    until assets_online; do sleep 60; done
    local capture="$rollouts/zs_${tag}_flat_b4_auto_kin_robust_${condition}_forward.npz"
    HUMANOID_CONTRACT="$root/$contract" \
      "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task_id" \
      --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
      --reference autonomous --estimator kinematic --command "$vx" 0 0 "${option[@]}" \
      --output "$capture" > "logs/contract/capture_zs_${tag}_robust_flat_b4_${condition}.log" 2>&1 \
      || { note "$tag flat_b4 $condition capture FAILED"; continue; }
    "$python" transfer/scripts/evaluate_contract.py \
      --run "teacher=g1/rollouts/flat_play_forward.npz" --run "encoder=$capture" \
      --output "$out" > /dev/null || { note "$tag flat_b4 $condition evaluate FAILED"; continue; }
    note "$tag flat_b4 $condition done"
  done
}

until grep -q "^w8 done" logs/contract/w8_plan_contact.status 2>/dev/null; do sleep 120; done
note "w5b start"
# The G1 incumbent has no flat_b4 L4 yet: its nominal capture, which also moves its encoder
# into flat_b4's contract.
if [ ! -f "$results/zero_shot_g1_warm_r3_logload_sym_flat_b4_auto_kin_l4_turning.json" ]; then
  wait_for_room "g1 flat_b4 l4"
  until assets_online; do sleep 60; done
  ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID=G1Track-Play-v0 \
    ZS_TASK=flat ZS_GENERATOR=logs/contract/flat_b4/generator.pt \
    bash transfer/scripts/zero_shot_ladder.sh g1 "$g1_run" model_7193 warm_r3_logload_sym l4 \
    > logs/contract/ladder_warm_r3_logload_sym_flat_b4.log 2>&1 \
    && note "g1 flat_b4 l4 done" || note "g1 flat_b4 l4 FAILED"
fi
robust g1 "$g1_run" model_7193 warm_r3_logload_sym
robust nao "$nao_run" model_3498 r1_3498_sym_v3
"$python" transfer/scripts/w5_table.py --output "$results/w5_robust.json" \
  > logs/contract/w5_table.out 2>&1 || note "w5 table FAILED"
note "w5b done"
