#!/usr/bin/env bash
# W6 (skill_space.md §10.8): the NAO's sole point at its resting contact, after W5.
#
#   bash transfer/scripts/w6_sole.sh
#
# 1. smoke   3 iterations with HUMANOID_SOLE=contact; the saved env.yaml must carry the
#            lowered sole points (z below -0.047 m in the ankle frame, against -0.04511).
# 2. screen  seed 31: r1_3498 continued 1000 iterations as S's NAO seeds, with the switch;
#            laddered as S's were (rough: encoder, walk, l4; flat: l4), the switch exported.
# 3. seeds   32 and 33 likewise, only if the screen passes (w6_compare.py).
# Each stage skips what already exists, so the queue resumes.
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w6_sole.status
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
# S's NAO training conditions (w3_w4_queue.sh conditions nao), plus the switch.
export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_HEADS=0 HUMANOID_PLAN_TRUST=0
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger2e/generator.pt" HUMANOID_LOAD_SPACE=linear
export HUMANOID_POSTURE_CONTRACT="$HUMANOID_CONTRACT"
export HUMANOID_SOLE=contact
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
  local seed="$1" name="warm_nao_sole_s$1" run
  ls -d logs/rsl_rl/nao_track/*_"$name" > /dev/null 2>&1 || train "$name" "$seed" 1000 || return 1
  run="$(ls -d logs/rsl_rl/nao_track/*_"$name" | tail -1)"
  ladder "$run" "${name}_sym"
}

until grep -q "^w5 done" logs/contract/w5_robust.status 2>/dev/null; do sleep 120; done
note "w6 start"

if ! ls -d logs/rsl_rl/nao_track/*_w6smoke_nao > /dev/null 2>&1; then
  train w6smoke_nao 1 3 256 || { note "w6 stopped"; exit 1; }
fi
smoke="$(ls -d logs/rsl_rl/nao_track/*_w6smoke_nao | tail -1)"
if "$python" -c "
import sys, yaml
env = yaml.unsafe_load(open(sys.argv[1], encoding='utf-8'))
found = []
def walk(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'sole_points':
                found.append(v)
            walk(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            walk(v)
walk(env)
ok = found and all(all(p[2] < -0.047 for p in points) for points in found)
sys.exit(0 if ok else 1)" "$smoke/params/env.yaml"; then
  note "smoke: env.yaml carries the contact sole points"
else
  note "smoke: env.yaml lacks the contact sole points FAILED"
  note "w6 stopped"
  exit 1
fi

seed_run 31
"$python" transfer/scripts/w6_compare.py --output "$results/w6_sole.json" \
  > logs/contract/w6_compare.out 2>&1 || note "w6 compare FAILED"
screen=$("$python" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('yes' if d.get('screen', {}).get('passes') else 'no')" "$results/w6_sole.json" 2>/dev/null || echo no)
note "w6 screen: $screen"
if [ "$screen" = yes ]; then
  seed_run 32
  seed_run 33
  "$python" transfer/scripts/w6_compare.py --output "$results/w6_sole.json" \
    > logs/contract/w6_compare.out 2>&1 || note "w6 compare FAILED"
fi
note "w6 done"
