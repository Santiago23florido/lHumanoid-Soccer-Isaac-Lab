#!/usr/bin/env bash
# W9 of skill_space.md §10.11: frozen trackers of both bodies with each foot their plans load
# grounded and held still (HUMANOID_PLAN_CONTACT=stick), L4 on both skills and S0's walk. The
# G1's are paired with their W8 grounded captures, the NAO's with the plans as drawn
# (w9_compare.py). Nothing is trained.
#
#   bash transfer/scripts/w9_stick.sh          # the G1's three seeds, then the NAO's
#   W9_STAGES="g1" bash transfer/scripts/w9_stick.sh
#
# Each tracker keeps the encoder and task contracts its ladder used, copied under the new tag,
# so the plans' feet are the only difference. Every stage waits for room and is skipped when
# its last result exists. It always ends with "w9 done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w9_stick.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
stages="${W9_STAGES:-g1 nao}"
want() { [[ " $stages " == *" $1 "* ]]; }

room() {
  local free gpu busy
  free=$(powershell -NoProfile -Command \
    "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  # No brackets or backslashes in the pattern (bash -> PowerShell quoting).
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|scripts.train.py|capture_velocity' }).Count")
  [[ "$free" =~ ^[0-9]+$ && "$gpu" =~ ^[0-9]+$ && "$busy" =~ ^[0-9]+$ ]] || return 1
  # Thresholds: free RAM in GB and GPU memory in use in MiB. W9_ROOM_GB=0 runs alongside
  # whatever else is open (the choice for W9, 2026-11-06); one Isaac process at a time
  # is always kept.
  [ "$free" -ge "${W9_ROOM_GB:-10}" ] && [ "$gpu" -le "${W9_ROOM_GPU_MB:-2500}" ] \
    && [ "$busy" -eq 0 ]
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

# <body> <run dir> <checkpoint> <old tag> <new tag>
evaluate_tracker() {
  local body="$1" run="$2" checkpoint="$3" old="$4" new="$5" task generator task_id rollouts
  task_id=G1Track-Play-v0 rollouts=g1/rollouts
  local walk=logs/contract/walk_dagger3e/generator.pt
  if [ "$body" = nao ]; then
    task_id=NaoTrack-Play-v0 rollouts=nao/rollouts walk=logs/contract/walk_dagger2e/generator.pt
  fi
  for contract in walk_zs flat_dagger2_zs rough_b4_zs; do
    local from="logs/contract/${contract}_${body}_${old}/generator.pt"
    local to="logs/contract/${contract}_${body}_${new}/generator.pt"
    [ -f "$from" ] || { note "$new: missing $from FAILED"; return 1; }
    mkdir -p "$(dirname "$to")"
    [ -f "$to" ] || cp "$from" "$to"
  done
  for task in rough flat; do
    generator=flat_dagger2
    [ "$task" = rough ] && generator=rough_b4
    if [ -f "transfer/results/zero_shot_${body}_${new}_${generator}_auto_kin_l4_turning.json" ]; then
      note "$new $task: already done"
      continue
    fi
    wait_for_room "$new $task"
    until assets_online; do sleep 60; done
    note "$new $task l4 start"
    HUMANOID_PLAN_CONTACT=stick ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 \
      ZS_TASK_ID="$task_id" ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$checkpoint" "$new" l4 \
      > "logs/contract/ladder_${new}_${task}.log" 2>&1 \
      || { note "$new $task l4 FAILED"; continue; }
    note "$new $task l4 done"
  done
  local walk_result="transfer/results/zero_shot_${body}_${new}_walk_auto_kin.json"
  if [ ! -f "$walk_result" ]; then
    wait_for_room "$new walk"
    until assets_online; do sleep 60; done
    note "$new walk start"
    local policy="$run/exported/${checkpoint}_sym.pt"
    [ -f "$policy" ] || { note "$new walk: no $policy FAILED"; return 1; }
    local posture=()
    [ "$body" = nao ] && posture=(HUMANOID_POSTURE_CONTRACT="$root/$walk")
    env HUMANOID_PLAN_CONTACT=stick "${posture[@]}" \
      HUMANOID_CONTRACT="$root/logs/contract/walk_zs_${body}_${new}/generator.pt" \
      "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task_id" \
      --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
      --reference autonomous --estimator kinematic \
      --output "$rollouts/zs_${new}_walk_auto_kin_encoder.npz" \
      > "logs/contract/capture_zs_${new}_walk_auto_kin_encoder.log" 2>&1 \
      || { note "$new walk capture FAILED"; return 1; }
    "$python" transfer/scripts/evaluate_contract.py \
      --run teacher=g1/rollouts/closed_loop_teacher.npz \
      --run "encoder=$rollouts/zs_${new}_walk_auto_kin_encoder.npz" \
      --output "$walk_result" > /dev/null || { note "$new walk evaluate FAILED"; return 1; }
    note "$new walk done"
  fi
}

note "w9 start: $stages"
if want g1; then
  evaluate_tracker g1 logs/rsl_rl/g1_track/2026-10-24_17-31-00_warm_r3_logload model_7193 \
    warm_r3_logload_sym warm_r3_logload_sym_stick
  for seed in 42 43; do
    run="$(ls -d logs/rsl_rl/g1_track/*_warm_r3_s${seed}_log | tail -1)"
    evaluate_tracker g1 "$run" model_7193 "warm_r3_s${seed}_log_sym" "warm_r3_s${seed}_log_sym_stick"
  done
fi
if want nao; then
  incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
  evaluate_tracker nao "$incumbent" model_3498 r1_3498_sym_v3 r1_3498_sym_v3_stick
  for seed in 32 33; do
    run="$(ls -d logs/rsl_rl/nao_track/*_warm_nao_r1_s${seed} | tail -1)"
    evaluate_tracker nao "$run" model_4497 "warm_nao_r1_s${seed}_sym" "warm_nao_r1_s${seed}_sym_stick"
  done
fi
"$python" transfer/scripts/w9_compare.py --output transfer/results/w9_stick.json \
  > logs/contract/w9_compare.out 2>&1 || note "w9 compare FAILED"
reading=$("$python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('g1', d.get('g1', {}).get('passes'), 'nao', d.get('nao', {}).get('passes'))" \
  transfer/results/w9_stick.json 2>/dev/null || echo "missing")
note "w9 decision: $reading"
note "w9 done"
