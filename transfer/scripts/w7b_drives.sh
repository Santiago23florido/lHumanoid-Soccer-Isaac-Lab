#!/usr/bin/env bash
# W7b: W7's criteria at leg gain 2 (K x 2, D x sqrt 2) over the same
# three frozen NAO trackers. W7 at gain 4 read "both" but failed its falls criterion (rough_b4
# at the fast command); the incumbent at gain 2, W7's dose, did not fall. This adds what the
# dose lacked: the incumbent's walk, and S's seeds 32 and 33 through L4 of both skills and the
# walk. Nothing is trained.
#
#   bash transfer/scripts/w7b_drives.sh
#
# Runs after W8 (w8_plan_contact.sh); W5b is launched after it. Every stage waits for room and
# is skipped when its result exists, so the queue can be resumed. It always ends with
# "w7b done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w7b_drives.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }

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

walk=logs/contract/walk_dagger2e/generator.pt
gain=2

# <run dir> <checkpoint> <old tag> <new tag>: as w7_drives.sh's evaluate_tracker, at gain 2,
# always with the walk.
evaluate_tracker() {
  local run="$1" checkpoint="$2" old="$3" new="$4" task generator
  for contract in walk_zs flat_dagger2_zs rough_b4_zs; do
    local from="logs/contract/${contract}_nao_${old}/generator.pt"
    local to="logs/contract/${contract}_nao_${new}/generator.pt"
    [ -f "$from" ] || { note "$new: missing $from FAILED"; return 1; }
    mkdir -p "$(dirname "$to")"
    [ -f "$to" ] || cp "$from" "$to"
  done
  for task in flat rough; do
    generator=flat_dagger2
    [ "$task" = rough ] && generator=rough_b4
    if [ -f "transfer/results/zero_shot_nao_${new}_${generator}_auto_kin_l4_turning.json" ]; then
      note "$new $task: already done"
      continue
    fi
    wait_for_room "$new $task"
    until assets_online; do sleep 60; done
    note "$new $task l4 start"
    HUMANOID_NAO_LEG_GAIN="$gain" HUMANOID_RESAMPLE=linear ZS_REFERENCE=autonomous \
      ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID=NaoTrack-Play-v0 ZS_TASK="$task" \
      ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$checkpoint" "$new" l4 \
      > "logs/contract/ladder_${new}_${task}.log" 2>&1 \
      || { note "$new $task l4 FAILED"; continue; }
    note "$new $task l4 done"
  done
  local walk_result="transfer/results/zero_shot_nao_${new}_walk_auto_kin.json"
  if [ ! -f "$walk_result" ]; then
    wait_for_room "$new walk"
    until assets_online; do sleep 60; done
    note "$new walk start"
    local policy="$run/exported/${checkpoint}_sym.pt"
    [ -f "$policy" ] || { note "$new walk: no $policy FAILED"; return 1; }
    HUMANOID_NAO_LEG_GAIN="$gain" HUMANOID_RESAMPLE=linear \
      HUMANOID_CONTRACT="$root/logs/contract/walk_zs_nao_${new}/generator.pt" \
      HUMANOID_POSTURE_CONTRACT="$root/$walk" \
      "$python" -u transfer/scripts/capture_contract.py --headless --task-id NaoTrack-Play-v0 \
      --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
      --reference autonomous --estimator kinematic \
      --output "nao/rollouts/zs_${new}_walk_auto_kin_encoder.npz" \
      > "logs/contract/capture_zs_${new}_walk_auto_kin_encoder.log" 2>&1 \
      || { note "$new walk capture FAILED"; return 1; }
    "$python" transfer/scripts/evaluate_contract.py \
      --run teacher=g1/rollouts/closed_loop_teacher.npz \
      --run "encoder=nao/rollouts/zs_${new}_walk_auto_kin_encoder.npz" \
      --output "$walk_result" > /dev/null || { note "$new walk evaluate FAILED"; return 1; }
    note "$new walk done"
  fi
}

until grep -q "^w8 done" logs/contract/w8_plan_contact.status 2>/dev/null; do sleep 120; done
note "w7b start"
incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1
evaluate_tracker "$incumbent" model_3498 r1_3498_sym_v3 r1_3498_sym_v3_g2
for seed in 32 33; do
  run="$(ls -d logs/rsl_rl/nao_track/*_warm_nao_r1_s${seed} | tail -1)"
  evaluate_tracker "$run" model_4497 "warm_nao_r1_s${seed}_sym" "warm_nao_r1_s${seed}_sym_g2"
done
"$python" transfer/scripts/w7_compare.py --gain 2 --output transfer/results/w7b_drives.json \
  > logs/contract/w7b_compare.out 2>&1 || note "w7b compare FAILED"
reading=$("$python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('reading', 'missing'), 'rl-arm' if d.get('opens_rl_arm') else 'no-arm')" \
  transfer/results/w7b_drives.json 2>/dev/null || echo "missing no-arm")
note "w7b decision: $reading"
note "w7b done"
