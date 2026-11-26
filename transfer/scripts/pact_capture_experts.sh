#!/usr/bin/env bash
# Capture the published velocity experts of the registered bodies, with joint states,
# at the evaluation's commands Froude-scaled from the G1's (v ~ sqrt(L), w ~ 1/sqrt(L)).
#
# G1 commands: forward 0.25 m/s; fast 0.6 m/s; turning 0.3 m/s with 0.3 rad/s.
# Isaac Lab trained H1 on vx in [0, 1] and Cassie on vx in [0.7, 1.0] (yaw by heading);
# the NAO's oracle is the paper's B2 (flat):
# a command outside that range is recorded but flagged out_of_range in the status file.
#
# Usage: bash transfer/scripts/pact_capture_experts.sh [skill=flat] [bodies=h1,cassie]
# Output: logs/pact/rollouts/<body>/<skill>_<command>.npz; status logs/pact/capture_experts.status
set -u
cd "$(dirname "$0")/../.."
skill="${1:-flat}"
bodies="${2:-h1 cassie}"
bodies="${bodies//,/ }"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
status=logs/pact/capture_experts.status
mkdir -p logs/pact
echo "start $(date -Iseconds) skill=$skill bodies=$bodies" >> "$status"

g1_leg=0.65457
for body in $bodies; do
  leg=$("$python" -c "import json;r=json.load(open('transfer/bodies/bodies.json'))['$body'];c=json.load(open(r['card']));print(r.get('leg_length_unit', c['leg_length']))" | tr -d '\r')
  Skill=$(echo "$skill" | sed 's/./\U&/')
  case "$body" in
    h1) Body=H1; lo=0.0; hi=1.0 ;;
    cassie) Body=Cassie; lo=0.7; hi=1.0 ;;
    g1) Body=G1; lo=0.0; hi=1.0 ;;
    nao) Body=NAO; lo=-9; hi=9 ;;
    *) echo "no published expert for $body" >> "$status"; continue ;;
  esac
  task="Isaac-Velocity-${Skill}-${Body}-Play-v0"
  policy="logs/published/Isaac-Velocity-${Skill}-${Body}-v0/policy.pt"
  if [ "$body" = nao ]; then
    # The NAO has no published expert: its oracle is Isaac Lab's G1 recipe in the NAO's
    # units (the paper's B2), seed 42, flat only.
    [ "$skill" = flat ] || { echo "no NAO oracle for $skill" >> "$status"; continue; }
    task=NaoVelocityFlat-Play-v0
    policy=logs/rsl_rl/nao_velocity_flat/2026-10-27_09-07-45_oracle_flat/exported/model_1499.pt
  fi
  out="logs/pact/rollouts/$body"
  mkdir -p "$out"
  for spec in "forward 0.25 0" "fast 0.6 0" "turning 0.3 0.3"; do
    set -- $spec
    name=$1
    read -r vx wz < <("$python" -c "import math;s=math.sqrt($leg/$g1_leg);print(f'{$2*s:.4f} {$3/s:.4f}')" | tr -d '\r')
    flag=$("$python" -c "print('in_range' if $lo <= $vx <= $hi else 'out_of_range')" | tr -d '\r')
    log="$out/${skill}_${name}.log"
    "$python" -u transfer/scripts/capture_velocity.py --headless --body "$body" \
      --task-id "$task" --driver-policy "$policy" --command "$vx" 0 "$wz" --joints \
      --output "$out/${skill}_${name}.npz" > "$log" 2>&1
    code=$?
    summary=$(grep -E "falls" "$log" | tail -1 | tr -d '\r')
    if [ $code -eq 0 ] && [ -f "$out/${skill}_${name}.npz" ]; then
      echo "ok $body $skill $name vx=$vx wz=$wz $flag | $summary" >> "$status"
    else
      echo "FAILED $body $skill $name (exit $code), see $log" >> "$status"
    fi
  done
done
echo "done $(date -Iseconds)" >> "$status"
