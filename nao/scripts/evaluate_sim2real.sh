#!/usr/bin/env bash
# A tracker trained for the real robot, on the settings that decide transfer.
#
#   bash nao/scripts/evaluate_sim2real.sh <policy> <tag> [task]
#
# The captures of nao/scripts/sensitivity.sh that matter (the nominal simulator, latency,
# torque ceilings, reflected inertia, friction, encoder offsets, the robot's own pipeline, and
# everything at once at the values expected of a real robot), forward at 0.154 m/s, plus the
# paper's three commands on the nominal simulator. A tracker with history inputs plays in
# NaoTrack-Sim2Real-Play-v0 (the plain play scene with those inputs); the default task.
# nao/scripts/sensitivity.py reads the captures into nao/results/sensitivity_<tag>.json.
set -euo pipefail

policy="$1"
tag="$2"
task="${3:-NaoTrack-Sim2Real-Play-v0}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

walk="logs/contract/walk_dagger2e/generator.pt"
export HUMANOID_POSTURE_CONTRACT="$root/$walk"
export HUMANOID_OBSERVE_UNITS="$root/$walk"
export HUMANOID_CONTRACT="$root/logs/contract/flat_dagger2_zs_nao_r1_3498_sym_v3/generator.pt"
[ -f "$policy" ] || { echo "missing $policy"; exit 1; }
speed=0.15388
turn_scale=1.6246

robot='"delay_ms": [30, 40], "encoder_offset_deg": 1.0, "imu_tilt_deg": 1.0, "gyro_bias": 0.01, "fsr_scale": 0.1, "fsr_offset": 0.02, "armature_scale": [0.5, 1.5], "effort_scale": [0.8, 1.0], "sensors": true, "estimator": true, "odometry": true, "loads": "fsr"'
settings="nominal|{}|$speed 0 0
fast|{}|$(awk "BEGIN{print 0.6*0.61552}") 0 0
turning|{}|$(awk "BEGIN{print 0.3*0.61552}") $(awk "BEGIN{print 0.2*0.61552}") $(awk "BEGIN{print 0.3*$turn_scale}")
delay_40|{\"delay_ms\": [40, 40]}|$speed 0 0
delay_60|{\"delay_ms\": [60, 60]}|$speed 0 0
effort_0.6|{\"effort_scale\": [0.6, 0.6]}|$speed 0 0
armature_2|{\"armature_scale\": [2.0, 2.0]}|$speed 0 0
friction_0.4|{\"friction\": [0.4, 0.4]}|$speed 0 0
encoder_3deg|{\"sensors\": true, \"encoder_offset_deg\": 3.0}|$speed 0 0
robot_pipeline|{\"sensors\": true, \"estimator\": true, \"odometry\": true, \"loads\": \"fsr\"}|$speed 0 0
robot_expected|{$robot}|$speed 0 0
robot_expected_turning|{$robot}|$(awk "BEGIN{print 0.3*0.61552}") $(awk "BEGIN{print 0.2*0.61552}") $(awk "BEGIN{print 0.3*$turn_scale}")"

captures=()
while IFS='|' read -r name model command; do
  out="nao/rollouts/sens_${tag}__${name}.npz"
  log="logs/contract/sens_${tag}__${name}.log"
  if [ ! -f "$out" ]; then
    # shellcheck disable=SC2086
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task" \
      --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
      --reference autonomous --estimator kinematic --command $command --joint-load \
      --robot-model "$model" --output "$out" > "$log" 2>&1 || true
  fi
  tail -1 "$log"
  [ -f "$out" ] && captures+=("$out")
done <<< "$settings"

"$python" nao/scripts/sensitivity.py "${captures[@]}" --output "nao/results/sensitivity_${tag}.json"
# The leg motors' load on the nominal commands (nao/scripts/joint_load.py).
loads=()
for name in nominal fast turning; do
  [ -f "nao/rollouts/sens_${tag}__${name}.npz" ] && loads+=("nao/rollouts/sens_${tag}__${name}.npz")
done
"$python" nao/scripts/joint_load.py "${loads[@]}" --output "nao/results/joint_load_${tag}.json"
