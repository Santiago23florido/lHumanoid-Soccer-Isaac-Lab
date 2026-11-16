#!/usr/bin/env bash
# How much each of a real NAO's imperfections costs a tracker that never saw it.
#
#   bash nao/scripts/sensitivity.sh [run dir] [checkpoint] [tag] [policy file]
#
# The paper's NAO path (autonomous reference, kinematic estimator, the flat skill's
# generator, the tracker observing in its walk contract's units) walking forward at the
# Froude-matched 0.154 m/s, once per setting, one imperfection at a time
# (humanoid_transfer.nao.sim2real.robot_model), then all of them together at the values
# expected of a real robot. nao/scripts/sensitivity.py reads the captures.
# Defaults: the adopted NAO tracker (transfer/results/incumbents.json).
set -euo pipefail

run="${1:-logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1}"
checkpoint="${2:-model_3498}"
tag="${3:-r1_3498_sym_v3}"
policy="${4:-$run/exported/${checkpoint}_sym.pt}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

walk="logs/contract/walk_dagger2e/generator.pt"
export HUMANOID_POSTURE_CONTRACT="$root/$walk"
export HUMANOID_OBSERVE_UNITS="$root/$walk"
export HUMANOID_CONTRACT="$root/logs/contract/flat_dagger2_zs_nao_r1_3498_sym_v3/generator.pt"
[ -f "$policy" ] || { echo "missing $policy"; exit 1; }
# 0.25 m/s on the G1, in the NAO's units: sqrt(0.24801 / 0.65457) of it.
speed=0.15388

robot='"delay_ms": [30, 40], "encoder_offset_deg": 1.0, "imu_tilt_deg": 1.0, "gyro_bias": 0.01, "fsr_scale": 0.1, "fsr_offset": 0.02, "armature_scale": [0.5, 1.5], "effort_scale": [0.8, 1.0], "sensors": true, "estimator": true, "odometry": true, "loads": "fsr"'
settings="nominal|{}
delay_10|{\"delay_ms\": [10, 10]}
delay_20|{\"delay_ms\": [20, 20]}
delay_30|{\"delay_ms\": [30, 30]}
delay_40|{\"delay_ms\": [40, 40]}
delay_60|{\"delay_ms\": [60, 60]}
stiffness_0.5|{\"stiffness_scale\": [0.5, 0.5]}
stiffness_0.75|{\"stiffness_scale\": [0.75, 0.75]}
stiffness_1.5|{\"stiffness_scale\": [1.5, 1.5]}
effort_0.6|{\"effort_scale\": [0.6, 0.6]}
effort_0.8|{\"effort_scale\": [0.8, 0.8]}
armature_0.5|{\"armature_scale\": [0.5, 0.5]}
armature_1|{\"armature_scale\": [1.0, 1.0]}
armature_2|{\"armature_scale\": [2.0, 2.0]}
friction_0.4|{\"friction\": [0.4, 0.4]}
friction_0.6|{\"friction\": [0.6, 0.6]}
friction_1.2|{\"friction\": [1.2, 1.2]}
mass_0.85|{\"mass_scale\": [0.85, 0.85]}
mass_1.15|{\"mass_scale\": [1.15, 1.15]}
added_mass_0.2|{\"added_mass_kg\": [0.2, 0.2]}
base_com_1.5cm|{\"base_com_m\": 0.015}
sensors|{\"sensors\": true}
encoder_1deg|{\"sensors\": true, \"encoder_offset_deg\": 1.0}
encoder_3deg|{\"sensors\": true, \"encoder_offset_deg\": 3.0}
imu_2deg|{\"sensors\": true, \"imu_tilt_deg\": 2.0}
leg_odometry_velocity|{\"base_velocity\": false}
robot_estimator_contact|{\"estimator\": true, \"loads\": \"contact\"}
robot_estimator_fsr|{\"estimator\": true, \"loads\": \"fsr\", \"fsr_scale\": 0.1, \"fsr_offset\": 0.02}
robot_estimator_encoder_no_vel|{\"estimator\": true, \"loads\": \"encoder_no_vel\"}
odometry_path|{\"odometry\": true}
robot_pipeline|{\"sensors\": true, \"estimator\": true, \"odometry\": true, \"loads\": \"fsr\"}
robot_expected|{$robot}"

captures=()
while IFS='|' read -r name model; do
  out="nao/rollouts/sens_${tag}__${name}.npz"
  log="logs/contract/sens_${tag}__${name}.log"
  if [ ! -f "$out" ]; then
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id NaoTrack-Play-v0 \
      --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
      --reference autonomous --estimator kinematic --command "$speed" 0 0 \
      --robot-model "$model" --output "$out" > "$log" 2>&1 || true
  fi
  tail -1 "$log"
  [ -f "$out" ] && captures+=("$out")
done <<< "$settings"

"$python" nao/scripts/sensitivity.py "${captures[@]}" --output "nao/results/sensitivity_${tag}.json"
