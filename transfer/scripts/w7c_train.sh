#!/usr/bin/env bash
# W7c: the RL arm W7b opened. r1_3498 continued 1000 iterations as S's
# NAO seeds were, with the leg drives at gain 2 (HUMANOID_NAO_LEG_GAIN=2: K x 2, D x sqrt 2)
# in training and evaluation; three seeds, laddered as S's were.
#
#   bash transfer/scripts/w7c_train.sh
#
# 1. smoke    3 iterations at gain 2; the saved env.yaml must carry the scaled leg gains.
# 2. control  S's seed 31 deployed frozen at gain 2, as W7b did seeds 32 and 33.
# 3. seeds    31, 32, 33: train, then ladder (rough: encoder, walk, l4; flat: l4).
# 4. compare  w7c_compare.py -> transfer/results/w7c_trained.json.
# Runs after W8n (w8_sole.sh). Each stage skips what already exists, so the queue resumes.
# Always ends with "w7c done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w7c_trained.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
results=transfer/results
incumbent=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1

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
rate() {
  "$python" -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['optimizer_state_dict']['param_groups'][0]['lr'])" "$1"
}
latest() {
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
# S's NAO training conditions (w6_sole.sh, w3_w4_queue.sh conditions nao), plus the drives.
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_NAO_LEG_GAIN=2
# <run name> <seed> <iterations> [num envs]
train() {
  local name="$1" seed="$2" iterations="$3" envs="${4:-4096}"
  wait_for_room "train $name"
  until assets_online; do sleep 60; done
  note "$name training start"
  "$python" -u scripts/train.py --task NaoTrack-v0 --headless --num_envs "$envs" \
    --max_iterations "$iterations" --seed "$seed" --run_name "$name" --resume \
    --load_run "$(basename "$incumbent")" --checkpoint model_3498.pt \
    "agent.algorithm.learning_rate=$(rate "$incumbent/model_3498.pt")" \
    > "logs/contract/train_${name}.log" 2>&1 || { note "$name training FAILED"; return 1; }
  note "$name trained"
}
# <run dir> <tag>: laddered as S's NAO seeds were
ladder() {
  local run="$1" tag="$2" checkpoint
  checkpoint="$(latest "$run")"
  for task in rough flat; do
    local generator=flat_dagger2 stages=(l4)
    [ "$task" = rough ] && { generator=rough_b4; stages=(encoder walk l4); }
    [ -f "$results/zero_shot_nao_${tag}_${generator}_auto_kin_l4_turning.json" ] && continue
    wait_for_room "l4 $tag $task"
    until assets_online; do sleep 60; done
    ZS_REFERENCE=autonomous ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID=NaoTrack-Play-v0 \
      ZS_TASK="$task" ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$checkpoint" "$tag" "${stages[@]}" \
      > "logs/contract/ladder_${tag}_${task}.log" 2>&1 || note "$tag $task ladder FAILED"
  done
  note "evaluated $tag"
}
seed_run() {  # <seed>: train if absent, then ladder
  local seed="$1" name="warm_nao_g2_s$1" run
  ls -d logs/rsl_rl/nao_track/*_"$name" > /dev/null 2>&1 || train "$name" "$seed" 1000 || return 1
  run="$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)"
  ladder "$run" "${name}_sym"
}
# S's seed 31 frozen at gain 2, as w7b_drives.sh's evaluate_tracker (L4 both skills, walk).
frozen_s31() {
  local run=logs/rsl_rl/nao_track/2026-10-24_02-47-22_warm_nao_r1 checkpoint=model_4497
  local old=warm_nao_r1_s31_sym new=warm_nao_r1_s31_sym_g2 task generator
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
    [ -f "$results/zero_shot_nao_${new}_${generator}_auto_kin_l4_turning.json" ] && continue
    wait_for_room "$new $task"
    until assets_online; do sleep 60; done
    # As W7b: the trackers' own capture conditions, not the training exports.
    env -u HUMANOID_PROCEDURAL -u HUMANOID_SYMMETRY -u HUMANOID_HEADS -u HUMANOID_PLAN_TRUST \
      -u HUMANOID_LOAD_SPACE -u HUMANOID_CONTRACT -u HUMANOID_POSTURE_CONTRACT \
      HUMANOID_RESAMPLE=linear ZS_REFERENCE=autonomous \
      ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1 ZS_TASK_ID=NaoTrack-Play-v0 ZS_TASK="$task" \
      ZS_GENERATOR="logs/contract/${generator}/generator.pt" \
      bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$checkpoint" "$new" l4 \
      > "logs/contract/ladder_${new}_${task}.log" 2>&1 || { note "$new $task l4 FAILED"; continue; }
    note "$new $task l4 done"
  done
  local walk_result="$results/zero_shot_nao_${new}_walk_auto_kin.json"
  [ -f "$walk_result" ] && return 0
  wait_for_room "$new walk"
  until assets_online; do sleep 60; done
  env -u HUMANOID_PROCEDURAL -u HUMANOID_SYMMETRY -u HUMANOID_HEADS -u HUMANOID_PLAN_TRUST \
    -u HUMANOID_LOAD_SPACE HUMANOID_RESAMPLE=linear \
    HUMANOID_CONTRACT="$root/logs/contract/walk_zs_nao_${new}/generator.pt" \
    HUMANOID_POSTURE_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" \
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id NaoTrack-Play-v0 \
    --driver-policy "$run/exported/${checkpoint}_sym.pt" --steps 900 --settle-steps 100 \
    --num-envs 32 --reference autonomous --estimator kinematic \
    --output "nao/rollouts/zs_${new}_walk_auto_kin_encoder.npz" \
    > "logs/contract/capture_zs_${new}_walk_auto_kin_encoder.log" 2>&1 \
    || { note "$new walk capture FAILED"; return 1; }
  "$python" transfer/scripts/evaluate_contract.py \
    --run teacher=g1/rollouts/closed_loop_teacher.npz \
    --run "encoder=nao/rollouts/zs_${new}_walk_auto_kin_encoder.npz" \
    --output "$walk_result" > /dev/null || { note "$new walk evaluate FAILED"; return 1; }
  note "$new walk done"
}

until grep -q "^w8n done" logs/contract/w8n_sole.status 2>/dev/null; do sleep 120; done
note "w7c start"

if ! ls -d logs/rsl_rl/nao_track/*_w7csmoke_nao > /dev/null 2>&1; then
  train w7csmoke_nao 1 3 256 || { note "w7c stopped"; note "w7c done"; exit 1; }
fi
smoke="$(ls -d logs/rsl_rl/nao_track/*_w7csmoke_nao | tail -1)"
if "$python" -c "
import math, sys, yaml
env = yaml.unsafe_load(open(sys.argv[1], encoding='utf-8'))
legs = env['scene']['robot']['actuators']['legs']
ok = abs(legs['stiffness']['LHipPitch'] - 22.0) < 1e-9 and \
    abs(legs['damping']['LHipPitch'] - 1.6 * math.sqrt(2)) < 1e-9
sys.exit(0 if ok else 1)" "$smoke/params/env.yaml"; then
  note "smoke: env.yaml carries the gain-2 leg drives"
else
  note "smoke: env.yaml lacks the gain-2 leg drives FAILED"
  note "w7c stopped"
  note "w7c done"
  exit 1
fi

frozen_s31 || note "frozen s31 at gain 2 FAILED"
for seed in 31 32 33; do
  seed_run "$seed" || note "seed $seed FAILED"
done
"$python" transfer/scripts/w7c_compare.py --output "$results/w7c_trained.json" \
  > logs/contract/w7c_compare.out 2>&1 || note "w7c compare FAILED"
verdict=$("$python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('passes' if d.get('passes') else 'fails', 'target' if d.get('target_met') else 'no-target')" \
  "$results/w7c_trained.json" 2>/dev/null || echo "missing")
note "w7c decision: $verdict"
note "w7c done"
