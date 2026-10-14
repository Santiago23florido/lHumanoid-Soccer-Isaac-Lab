#!/usr/bin/env bash
# The NAO's round of the zero-shot study (transfer/docs/skill_space.md §8.4): the NAO's body side
# trained once on the procedural family, then the G1-side task, frozen, deployed on it. Nothing
# of the G1 is trained, and no data of the published tasks reaches the NAO.
#
#   transfer/scripts/zero_shot_nao.sh [iterations] [name]      (defaults: 2000, zs_nao_r1)
#
# 1. mirror  -- the NAO's sagittal mirror from its own kinematics, once: nao/body/mirror.json,
#               the right hip yaw-pitch recorded as the left's mimic (one motor drives both).
#               Stops if any commanded joint's sign is ambiguous.
# 2. train   -- its walk tracker (crouch 2, model_1499) resumed with 70 % procedural plans, the
#               octave likelihood and the mirror loss: the G1's recipe as round 3 ended it.
# 3. ladder  -- deployed symmetrized, every level, on the published generator the G1 passed
#               L4 with (flat_dagger2), the amended L4 (autonomous reference).
#
# About 4.5 h of training at ~8 s per iteration, then about an hour of ladder. Progress:
# logs/contract/<name>.status and logs/contract/train_<name>.log.
set -euo pipefail

iterations="${1:-2000}"
name="${2:-zs_nao_r1}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status="logs/contract/${name}.status"
walk="logs/contract/walk_dagger2e/generator.pt"
start_run=2026-10-03_03-06-21_track_walk_nao_crouch2
start_checkpoint=1499

export HUMANOID_CONTRACT="$root/$walk"
export HUMANOID_POSTURE_CONTRACT="$root/$walk"
echo "start $(date +%H:%M) iterations $iterations" > "$status"

if [ ! -f nao/body/mirror.json ]; then
  "$python" -u transfer/scripts/measure_mirror.py --headless --task-id NaoTrack-v0 \
    --prefix L R --prefix l_ r_ --mimic RHipYawPitch LHipYawPitch 1.0 \
    --max-joints 48 \
    --output nao/body/mirror.json > logs/contract/measure_mirror_nao.log 2>&1
fi
"$python" - <<'PYEOF' | tee -a "$status"
import json
import sys

import numpy as np

sys.path.insert(0, "source/humanoid_transfer")
from humanoid_transfer.nao.assets.nao_kinematics import ACTUATED_JOINTS
from humanoid_transfer.transfer.symmetry import joint_mirror, joint_offsets

spec = json.load(open("nao/body/mirror.json", encoding="utf-8"))
rows = {r["name"]: r for r in spec["joints"]}
bad = [n for n in ACTUATED_JOINTS if not rows[n]["residual"] < 0.05 < 1.0 < rows[n]["other_residual"]]
p = joint_mirror(ACTUATED_JOINTS, spec)
offset = joint_offsets(p, np.array([spec["default_joint_pos"][n] for n in ACTUATED_JOINTS]), 1.0)[0]
print(f"mirror: default pose residual {spec['default_pose_residual']:.2e}, "
      f"unpaired links {spec['unpaired_links']}, |P q0 - q0| max {np.abs(offset).max():.3f}")
if bad or not np.allclose(p @ p, np.eye(len(ACTUATED_JOINTS))):
    print(f"mirror ambiguous for {bad}: stopping before training")
    sys.exit(1)
PYEOF

export HUMANOID_PROCEDURAL=0.7
export HUMANOID_SYMMETRY=loss
"$python" -u scripts/train.py --task NaoTrack-v0 --headless --num_envs 4096 \
  --max_iterations "$iterations" --seed 31 --run_name "$name" --resume \
  --load_run "$start_run" --checkpoint "model_${start_checkpoint}.pt" \
  > "logs/contract/train_${name}.log" 2>&1
echo "trained $(date +%H:%M)" >> "$status"
unset HUMANOID_PROCEDURAL HUMANOID_SYMMETRY

run="$(ls -d logs/rsl_rl/nao_track/*_"${name}" | tail -1)"
checkpoint="model_$((start_checkpoint + iterations - 1))"
ZS_SYMMETRIZE=1 ZS_REFERENCE=autonomous ZS_GENERATOR=logs/contract/flat_dagger2/generator.pt \
  bash transfer/scripts/zero_shot_ladder.sh nao "$run" "$checkpoint" \
  "${name#zs_nao_}_${checkpoint#model_}_sym" \
  > "logs/contract/ladder_${name}.log" 2>&1
echo "ladder $(date +%H:%M): logs/contract/ladder_${name}.log" >> "$status"
