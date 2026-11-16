#!/usr/bin/env bash
# A tracker on robots whose parameters are known: what telling (or estimating) them is worth.
#
#   bash nao/scripts/evaluate_adaptation.sh <policy> <tag> [task]
#
# Each setting is one robot: the robot's pipeline (its sensors, estimator and odometry), its
# drives drawn as one factor with the joints +-10 % around it (as the teacher trains), and one
# parameter of robot_params moved to an end of its range -- or several, or the values expected
# of a real robot -- forward at 0.154 m/s (and turning for the expected robot). The task gives
# the tracker its inputs: NaoTrack-Sim2Real-Play-v0 for the robot-trained tracker,
# NaoTrack-Sim2RealTeacher-Play-v0 for the teacher (told the true parameters),
# NaoTrack-Sim2RealStudent-Play-v0 for the student (its own estimate).
# nao/scripts/sensitivity.py reads the captures into nao/results/adaptation_<tag>.json.
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
forward="$speed 0 0"
turning="$(awk "BEGIN{print 0.3*0.61552}") $(awk "BEGIN{print 0.2*0.61552}") $(awk "BEGIN{print 0.3*$turn_scale}")"

pipe='"sensors": true, "estimator": true, "odometry": true, "loads": "fsr", "joint_spread": 0.1'
unit='"stiffness_scale": [1, 1], "damping_scale": [1, 1], "effort_scale": [1, 1], "armature_scale": [1, 1]'
expected='"delay_ms": [30, 40], "encoder_offset_deg": 1.0, "imu_tilt_deg": 1.0, "gyro_bias": 0.01, "fsr_scale": 0.1, "fsr_offset": 0.02, "stiffness_scale": [0.85, 1.15], "damping_scale": [0.85, 1.15], "armature_scale": [0.5, 1.5], "effort_scale": [0.8, 1.0]'
settings="middle|{$pipe, $unit, \"delay_ms\": [35, 35]}|$forward
delay_10|{$pipe, $unit, \"delay_ms\": [10, 10]}|$forward
delay_60|{$pipe, $unit, \"delay_ms\": [60, 60]}|$forward
effort_0.7|{$pipe, $unit, \"delay_ms\": [35, 35], \"effort_scale\": [0.7, 0.7]}|$forward
effort_1.1|{$pipe, $unit, \"delay_ms\": [35, 35], \"effort_scale\": [1.1, 1.1]}|$forward
stiffness_0.7|{$pipe, $unit, \"delay_ms\": [35, 35], \"stiffness_scale\": [0.7, 0.7]}|$forward
stiffness_1.3|{$pipe, $unit, \"delay_ms\": [35, 35], \"stiffness_scale\": [1.3, 1.3]}|$forward
armature_0.5|{$pipe, $unit, \"delay_ms\": [35, 35], \"armature_scale\": [0.5, 0.5]}|$forward
armature_1.5|{$pipe, $unit, \"delay_ms\": [35, 35], \"armature_scale\": [1.5, 1.5]}|$forward
weak_late|{$pipe, \"delay_ms\": [60, 60], \"stiffness_scale\": [0.7, 0.7], \"damping_scale\": [0.7, 0.7], \"effort_scale\": [0.7, 0.7], \"armature_scale\": [1.5, 1.5]}|$forward
strong_early|{$pipe, \"delay_ms\": [10, 10], \"stiffness_scale\": [1.3, 1.3], \"damping_scale\": [1.3, 1.3], \"effort_scale\": [1.1, 1.1], \"armature_scale\": [0.5, 0.5]}|$forward
expected|{$pipe, $expected}|$forward
expected_turning|{$pipe, $expected}|$turning"

captures=()
while IFS='|' read -r name model command; do
  out="nao/rollouts/adapt_${tag}__${name}.npz"
  log="logs/contract/adapt_${tag}__${name}.log"
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

"$python" nao/scripts/sensitivity.py "${captures[@]}" --nominal middle \
  --output "nao/results/adaptation_${tag}.json"
