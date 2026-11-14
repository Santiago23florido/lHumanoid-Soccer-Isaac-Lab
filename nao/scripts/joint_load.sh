#!/usr/bin/env bash
# The NAO's leg loads on the paper's deployed path, against its motors' datasheet.
#
#   bash nao/scripts/joint_load.sh [run dir] [checkpoint] [tag] [task]
#
# Repeats the L4 captures of transfer/scripts/zero_shot_ladder.sh for the NAO (autonomous
# reference, kinematic estimator, the tracker observing in its walk contract's units, the
# tracker symmetrised), with --joint-load, and reads them with nao/scripts/joint_load.py.
# Defaults: the adopted NAO tracker (transfer/results/incumbents.json) on the flat skill.
set -euo pipefail

run="${1:-logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1}"
checkpoint="${2:-model_3498}"
tag="${3:-r1_3498_sym_v3}"
task="${4:-flat}"
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

walk="logs/contract/walk_dagger2e/generator.pt"
generator="${task}_dagger2"
if [ "$task" = rough ]; then generator=rough_b4; fi
export HUMANOID_POSTURE_CONTRACT="$root/$walk"
export HUMANOID_OBSERVE_UNITS="$root/$walk"
export HUMANOID_CONTRACT="$root/logs/contract/${generator}_zs_nao_${tag}/generator.pt"
policy="$run/exported/${checkpoint}_sym.pt"
[ -f "$policy" ] || { echo "missing $policy (zero_shot_ladder.sh with ZS_SYMMETRIZE=1)"; exit 1; }

read -r speed_scale turn_scale <<< "$("$python" -c "
import math
g1, nao = 0.6545728842979989, 0.24801
print(math.sqrt(nao / g1), math.sqrt(g1 / nao))")"
if [ "$task" = rough ]; then
  commands="forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0 0.3"
else
  commands="forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0.2 0.3"
fi

captures=()
while read -r name vx vy wz; do
  vx=$("$python" -c "print($vx * $speed_scale)")
  vy=$("$python" -c "print($vy * $speed_scale)")
  wz=$("$python" -c "print($wz * $turn_scale)")
  out="nao/rollouts/load_${tag}_${generator}_${name}.npz"
  "$python" -u transfer/scripts/capture_contract.py --headless --task-id NaoTrack-Play-v0 \
    --driver-policy "$policy" --steps 900 --settle-steps 100 --num-envs 32 \
    --reference autonomous --estimator kinematic --command "$vx" "$vy" "$wz" --joint-load \
    --output "$out" > "logs/contract/capture_load_${tag}_${generator}_${name}.log" 2>&1
  tail -1 "logs/contract/capture_load_${tag}_${generator}_${name}.log"
  captures+=("$out")
done <<< "$commands"

"$python" nao/scripts/joint_load.py "${captures[@]}" \
  --output "nao/results/joint_load_${tag}_${generator}.json"
