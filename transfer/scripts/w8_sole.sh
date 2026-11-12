#!/usr/bin/env bash
# W8n: W8's grounded plans on W6's NAO trackers, whose sole point
# rests at the contact (HUMANOID_SOLE=contact). W8 missed on the NAO, whose loaded feet read
# 0.012-0.013 L through the URDF sole point; this asks whether the miss was that offset.
# L4 of both skills and S0's walk with HUMANOID_PLAN_CONTACT=ground, paired with W6's own
# captures of the plans as drawn. Nothing is trained. Runs after W6 (w6_sole.sh), on every
# W6 tracker that exists (seed 31; 32 and 33 only if W6's screen passed). Always ends with
# "w8n done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w8n_sole.status
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

# W6's ladder ran under its training conditions; the same here, so grounding is the only
# difference (w6_sole.sh).
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_SOLE=contact

latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
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
    HUMANOID_PLAN_CONTACT=ground ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 \
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
    env HUMANOID_PLAN_CONTACT=ground "${posture[@]}" \
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

until grep -q "^w6 done" logs/contract/w6_sole.status 2>/dev/null; do sleep 120; done
note "w8n start"
tags=()
for seed in 31 32 33; do
  run="$(ls -d logs/rsl_rl/nao_track/*_warm_nao_sole_s${seed} 2>/dev/null | tail -1)"
  [ -n "$run" ] || continue
  tag="warm_nao_sole_s${seed}_sym"
  [ -f "transfer/results/zero_shot_nao_${tag}_rough_b4_auto_kin_l4_turning.json" ] || {
    note "$tag: W6's own L4 missing, skipped"
    continue
  }
  evaluate_tracker nao "$run" "$(latest "$run")" "$tag" "${tag}_ground" && tags+=("$tag")
done
if [ ${#tags[@]} -gt 0 ]; then
  "$python" transfer/scripts/w8_compare.py --bodies nao --nao-trackers "${tags[@]}" \
    --output transfer/results/w8n_sole.json > logs/contract/w8n_compare.out 2>&1 \
    || note "w8n compare FAILED"
  note "w8n trackers: ${tags[*]}"
else
  note "w8n: no W6 tracker FAILED"
fi
note "w8n done"
