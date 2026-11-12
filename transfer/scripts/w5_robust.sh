#!/usr/bin/env bash
# W5: robustness, reported beside the metrics.
#
#   bash transfer/scripts/w5_robust.sh
#
# Each tracker, deployed as L4 deploys it (autonomous reference, kinematic estimator,
# symmetrized, its own encoder in the task's contract), at L4's forward command of both
# skills, under four conditions:
#   push10, push20   the base pushed by up to 0.1 / 0.2 V each horizontal way, every 20 T
#                    on average (on the G1, 0.2 V is about Isaac Lab's training push, 0.5 m/s)
#   noise2, noise4   the deployed estimate read with 2 / 4 times the estimator's precision
# The trackers: both incumbents, the NAO incumbent under the generator-space interface (W1),
# and W3's first seeds with their controls, where they exist. Waits for the W3/W4 queue.
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w5_robust.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results

room() {
  local free gpu busy
  free=$(powershell -NoProfile -Command \
    "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [[ "$free" =~ ^[0-9]+$ && "$gpu" =~ ^[0-9]+$ && "$busy" =~ ^[0-9]+$ ]] || return 1
  [ "$free" -ge 10 ] && [ "$gpu" -le 2500 ] && [ "$busy" -eq 0 ]
}
assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}
latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}

# <body> <run dir> <tag> <space>
robust() {
  local body="$1" run="$2" tag="$3" space="$4" task_id=G1Track-Play-v0 rollouts=g1/rollouts
  local vx=0.25 policy
  if [ "$body" = nao ]; then
    task_id=NaoTrack-Play-v0
    rollouts=nao/rollouts
    vx=$("$python" -c "import math; print(0.25 * math.sqrt(0.24801 / 0.6545728842979989))")
    export HUMANOID_POSTURE_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt"
  else
    unset HUMANOID_POSTURE_CONTRACT
  fi
  policy="$run/exported/$(latest "$run")_sym.pt"
  [ -f "$policy" ] || { note "$tag: no $policy FAILED"; return 1; }
  for generator in rough_b4 flat_dagger2; do
    local task=${generator%%_*}
    local contract="logs/contract/${generator}_zs_${body}_${tag}/generator.pt"
    [ -f "$contract" ] || { note "$tag: no $contract, skipped"; continue; }
    for condition in push10 push20 noise2 noise4; do
      local out="$results/zero_shot_${body}_${tag}_${generator}_auto_kin_robust_${condition}_forward.json"
      [ -f "$out" ] && continue
      local option=()
      case "$condition" in
        push10) option=(--push 0.1 20) ;;
        push20) option=(--push 0.2 20) ;;
        noise2) option=(--estimate-noise 2) ;;
        noise4) option=(--estimate-noise 4) ;;
      esac
      until room; do sleep 120; done
      until assets_online; do sleep 60; done
      local capture="$rollouts/zs_${tag}_${generator}_auto_kin_robust_${condition}_forward.npz"
      HUMANOID_RESAMPLE="$space" HUMANOID_CONTRACT="$root/$contract" \
        "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task_id" \
        --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
        --reference autonomous --estimator kinematic --command "$vx" 0 0 "${option[@]}" \
        --output "$capture" > "logs/contract/capture_zs_${tag}_robust_${generator}_${condition}.log" 2>&1 \
        || { note "$tag $generator $condition capture FAILED"; continue; }
      "$python" transfer/scripts/evaluate_contract.py \
        --run "teacher=g1/rollouts/${task}_play_forward.npz" --run "encoder=$capture" \
        --output "$out" > /dev/null || { note "$tag $generator $condition evaluate FAILED"; continue; }
      note "$tag $generator $condition done"
    done
  done
}

until grep -q "^w3/w4 done" logs/contract/w3_w4.status 2>/dev/null; do sleep 120; done
note "w5 start"
robust g1 logs/rsl_rl/g1_track/2026-10-24_17-31-00_warm_r3_logload warm_r3_logload_sym linear
robust nao logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1 r1_3498_sym_v3 linear
[ -d logs/contract/rough_b4_zs_nao_r1_3498_sym_v3_gen ] && \
  robust nao logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1 r1_3498_sym_v3_gen generator
for pair in "g1 warm_r3_shape_s41" "nao warm_nao_shape_s31"; do
  read -r body name <<< "$pair"
  run="$(ls -d logs/rsl_rl/"${body}"_track/*_"$name" 2>/dev/null | tail -1)"
  [ -n "$run" ] && robust "$body" "$run" "${name}_sym" linear
done
# W3's NAO control at seed 31 (S's warm_nao_r1); the G1's seed 41 control is the incumbent.
[ -d logs/rsl_rl/nao_track/2026-10-24_02-47-22_warm_nao_r1 ] && [ -n "$(ls -d logs/rsl_rl/nao_track/*_warm_nao_shape_s31 2>/dev/null)" ] && \
  robust nao logs/rsl_rl/nao_track/2026-10-24_02-47-22_warm_nao_r1 warm_nao_r1_s31_sym linear
note "w5 done"
