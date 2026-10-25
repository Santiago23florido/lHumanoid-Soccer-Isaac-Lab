#!/usr/bin/env bash
# The paper's baselines that need tasks of their own (transfer/paper/README.md), after
# paper_queue.sh. Every stage skips itself when its last result file exists.
#
#   check   capture_velocity.py held to capture_rollouts.py: the G1's published rough policy,
#           forward, captured by both and compared (transfer/results/capture_velocity_check_g1.json)
#   b2      the oracle: NaoVelocityFlat-v0 and NaoVelocityRough-v0 trained by Isaac Lab's G1
#           recipe in the NAO's units, then each skill's three L4 commands on the plane
#           (transfer/results/oracle_nao_<skill>_l4_<command>.json)
#
#   bash transfer/scripts/paper_baselines.sh [stages]     (default: "check b2")
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/paper_baselines.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
stages="${1:-check b2}"
want() { [[ " $stages " == *" $1 "* ]]; }
results=transfer/results
seed=42

assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}
capture() {  # <log name> [capture_velocity.py arguments]: retried when an asset could not be opened
  local log="logs/contract/capture_$1.log" attempt
  shift
  for attempt in 1 2 3; do
    until assets_online; do sleep 60; done
    "$python" -u transfer/scripts/capture_velocity.py --headless "$@" > "$log" 2>&1
    grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || break
    note "capture $log: an asset could not be opened (attempt $attempt); retrying"
  done
  tail -1 "$log"
  grep -q "written" "$log"
}
latest() {  # <run directory>: its last checkpoint's name
  echo "model_$(ls "$1" | grep -E '^model_[0-9]+\.pt$' | sed 's/model_//; s/\.pt//' | sort -n | tail -1)"
}
# The G1's test commands in the NAO's units, as zero_shot_ladder.sh scales them.
read -r speed_scale turn_scale <<< "$("$python" -c "
import math
g1, nao = 0.6545728842979989, 0.24801
print(math.sqrt(nao / g1), math.sqrt(g1 / nao))")"
commands() {  # <skill>: the L4 commands, as zero_shot_ladder.sh lists them
  if [ "$1" = rough ]; then
    printf 'forward 0.25 0 0\nfast 0.6 0 0\nturning 0.3 0 0.3\n'
  else
    printf 'forward 0.25 0 0\nfast 0.6 0 0\nturning 0.3 0.2 0.3\n'
  fi
}

note "paper baselines start: $stages"

if want check && [ ! -f "$results/capture_velocity_check_g1.json" ]; then
  if capture velocity_check_g1 --body g1 --task-id G1RoughOnFlat-Play-v0 \
    --driver-policy logs/published/Isaac-Velocity-Rough-G1-v0/policy.pt --command 0.25 0 0 \
    --output g1/rollouts/check_velocity_rough_forward.npz; then
    "$python" transfer/scripts/evaluate_contract.py \
      --run teacher=g1/rollouts/rough_play_forward.npz \
      --run check=g1/rollouts/check_velocity_rough_forward.npz \
      --output "$results/capture_velocity_check_g1.json" | grep -v "^  "
    note "capture check done"
  else
    note "capture check FAILED"
  fi
fi

if want b2; then
  for skill in flat rough; do
    [ -f "$results/oracle_nao_${skill}_l4_turning.json" ] && { note "oracle $skill: already done"; continue; }
    if [ "$skill" = flat ]; then task=NaoVelocityFlat-v0 play=NaoVelocityFlat-Play-v0; else
      task=NaoVelocityRough-v0 play=NaoVelocityRoughOnFlat-Play-v0; fi
    name="oracle_${skill}"
    if ! ls -d logs/rsl_rl/nao_velocity_${skill}/*_"$name" > /dev/null 2>&1; then
      "$python" -u scripts/train.py --task "$task" --headless --num_envs 4096 --seed "$seed" \
        --run_name "$name" > "logs/contract/train_${name}.log" 2>&1 \
        || { note "$name training FAILED"; continue; }
      note "$name trained"
    fi
    run="$(ls -d logs/rsl_rl/nao_velocity_${skill}/*_"$name" | tail -1)"
    checkpoint="$(latest "$run")"
    "$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt" | tail -1
    while read -r command vx vy wz; do
      vx=$("$python" -c "print($vx * $speed_scale)")
      vy=$("$python" -c "print($vy * $speed_scale)")
      wz=$("$python" -c "print($wz * $turn_scale)")
      capture "${name}_${command}" --body nao --task-id "$play" \
        --driver-policy "$run/exported/$checkpoint.pt" --command "$vx" "$vy" "$wz" \
        --output "nao/rollouts/${name}_${command}.npz" || { note "$name $command capture FAILED"; continue; }
      "$python" transfer/scripts/evaluate_contract.py \
        --run "teacher=g1/rollouts/${skill}_play_${command}.npz" \
        --run "oracle=nao/rollouts/${name}_${command}.npz" \
        --output "$results/oracle_nao_${skill}_l4_${command}.json" | grep -v "^  "
    done < <(commands "$skill")
    note "evaluated $name"
  done
fi

note "paper baselines done: $stages"
