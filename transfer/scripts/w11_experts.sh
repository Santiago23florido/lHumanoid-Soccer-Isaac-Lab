#!/usr/bin/env bash
# W11 of skill_space.md §10.12: three new G1 experts for the generator's base tasks
# (g1/tasks/g1_base: backward, sidestep, jog), each Isaac Lab's flat G1 recipe with only its
# command range changed, trained as G1Walk-Teacher-v0 was. For each:
#   train    scripts/train.py, G1FlatPPORunnerCfg, 1500 iterations, 4096 envs, seed 42
#   export   transfer/scripts/export_actor.py (no simulator)
#   capture  DART rollouts for the generator (as held_out_task.sh), then play rollouts at the
#            three fixed commands of PLAY_COMMANDS
# then w11_accept.py -> transfer/results/w11_experts.json. A 3-iteration smoke run checks the
# saved env.yaml's command range first. Nothing of either body is trained.
#
#   bash transfer/scripts/w11_experts.sh
#
# Runs after W9 (w9_stick.sh). Room, per the project's rule: no free-RAM threshold; one Isaac
# process at a time; it waits while the GPU has under about 2 GB free. Every stage is skipped
# when its output exists. It always ends with "w11 done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w11_experts.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }

room() {
  local free gpu busy
  free=$(powershell -NoProfile -Command \
    "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)")
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  # No brackets or backslashes in the pattern (bash -> PowerShell quoting).
  busy=$(powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'capture_contract|capture_rollouts|scripts.train.py|capture_velocity' }).Count")
  [[ "$free" =~ ^[0-9]+$ && "$gpu" =~ ^[0-9]+$ && "$busy" =~ ^[0-9]+$ ]] || return 1
  [ "$free" -ge "${W11_ROOM_GB:-0}" ] && [ "$gpu" -le "${W11_ROOM_GPU_MB:-6100}" ] \
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
run_of() { ls -d logs/rsl_rl/g1_flat/*_"$1" 2>/dev/null | tail -1; }

# <Task> <run name> <iterations> <envs>
train() {
  local task="$1" name="$2" iterations="$3" envs="$4"
  [ -n "$(run_of "$name")" ] && [ -f "$(run_of "$name")/model_$((iterations - 1)).pt" ] \
    && { note "$name: already trained"; return 0; }
  wait_for_room "train $name"
  until assets_online; do sleep 60; done
  note "$name training start"
  "$python" -u scripts/train.py --task "G1Base-${task}-v0" --headless --num_envs "$envs" \
    --max_iterations "$iterations" --seed 42 --run_name "$name" \
    > "logs/contract/train_${name}.log" 2>&1 || { note "$name training FAILED"; return 1; }
  note "$name trained"
}

# <Task> <lowercase task>
expert() {
  local task="$1" lower="$2" name="base_$2" run policy
  train "$task" "$name" 1500 4096 || return 1
  run="$(run_of "$name")"
  policy="$run/exported/model_1499.pt"
  [ -f "$policy" ] || "$python" transfer/scripts/export_actor.py "$run/model_1499.pt" > /dev/null \
    || { note "$name export FAILED"; return 1; }
  if [ ! -f "g1/rollouts/base_${lower}_dart.npz" ]; then
    wait_for_room "$name dart"
    until assets_online; do sleep 60; done
    "$python" -u g1/scripts/capture_rollouts.py --headless --task-id "G1Base-${task}-v0" \
      --checkpoint "$policy" --num-envs 128 --steps 1000 --action-noise 0.1 \
      --standing-fraction 0.3 --resample-seconds 2 6 \
      --output "g1/rollouts/base_${lower}_dart.npz" > "logs/contract/capture_base_${lower}_dart.log" 2>&1 \
      || { note "$name dart capture FAILED"; return 1; }
    note "$name dart captured"
  fi
  while read -r cname vx vy wz; do
    local out="g1/rollouts/base_${lower}_play_${cname}.npz"
    [ -f "$out" ] && continue
    wait_for_room "$name play $cname"
    until assets_online; do sleep 60; done
    "$python" -u g1/scripts/capture_rollouts.py --headless --task-id "G1Base-${task}-Play-v0" \
      --checkpoint "$policy" --num-envs 32 --steps 900 --settle-steps 100 \
      --command "$vx" "$vy" "$wz" --output "$out" \
      > "logs/contract/capture_base_${lower}_${cname}.log" 2>&1 \
      || { note "$name play $cname capture FAILED"; continue; }
  done < <("$python" -c "
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding='utf-8').read())
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'PLAY_COMMANDS':
        for name, (vx, vy, wz) in ast.literal_eval(node.value)[sys.argv[2]].items():
            print(name, vx, vy, wz)" \
    source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_base/g1_base_env_cfg.py "$lower")
  note "$name played"
}

until grep -q "^w9 done" logs/contract/w9_stick.status 2>/dev/null; do sleep 120; done
note "w11 start"

if [ -z "$(run_of w11smoke)" ]; then
  wait_for_room "smoke"
  until assets_online; do sleep 60; done
  "$python" -u scripts/train.py --task G1Base-Backward-v0 --headless --num_envs 256 \
    --max_iterations 3 --seed 42 --run_name w11smoke > logs/contract/train_w11smoke.log 2>&1 \
    || { note "smoke training FAILED"; note "w11 stopped"; note "w11 done"; exit 1; }
fi
if "$python" -c "
import sys, yaml
env = yaml.unsafe_load(open(sys.argv[1], encoding='utf-8'))
r = env['commands']['base_velocity']['ranges']
ok = list(r['lin_vel_x']) == [-0.5, 0.0] and list(r['lin_vel_y']) == [0.0, 0.0] \
    and list(r['ang_vel_z']) == [0.0, 0.0]
sys.exit(0 if ok else 1)" "$(run_of w11smoke)/params/env.yaml"; then
  note "smoke: env.yaml carries the backward command range"
else
  note "smoke: env.yaml lacks the backward command range FAILED"
  note "w11 stopped"
  note "w11 done"
  exit 1
fi

expert Backward backward || note "backward FAILED"
expert Sidestep sidestep || note "sidestep FAILED"
expert Jog jog || note "jog FAILED"
"$python" transfer/scripts/w11_accept.py --output transfer/results/w11_experts.json \
  > logs/contract/w11_accept.out 2>&1 || note "w11 accept FAILED"
accepted=$("$python" -c "
import json, sys
print(' '.join(json.load(open(sys.argv[1]))['accepted']) or 'none')" \
  transfer/results/w11_experts.json 2>/dev/null || echo "missing")
note "w11 decision: accepted $accepted"
note "w11 done"
