#!/usr/bin/env bash
# Why W11's backward expert fails its play rollouts but not its DART capture . Same exported policy, same command (mid, -0.35 m/s), 32 environments,
# 900 steps; a 2 x 2 of the two things the captures differ in:
#   environment   the play variant (no observation noise, no pushes) or the training one
#   action noise  0 (the play rollouts) or 0.1 (the DART capture)
# The play variant without noise is W11's own capture and is not repeated. Nothing is
# trained. Runs after W11 (one Isaac process), then reads every cell with w11_accept.check.
#
#   bash transfer/scripts/w11_play_check.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w11_play_check.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
policy="$(ls -d logs/rsl_rl/g1_flat/*_base_backward | tail -1)/exported/model_1499.pt"

until grep -q "^w11 done" logs/contract/w11_experts.status 2>/dev/null; do sleep 120; done
note "check start"
for cell in "play 0.1 G1Base-Backward-Play-v0" "train 0 G1Base-Backward-v0" "train 0.1 G1Base-Backward-v0"; do
  read -r env noise task <<< "$cell"
  out="g1/rollouts/check_base_backward_${env}_noise${noise}.npz"
  [ -f "$out" ] && continue
  "$python" -u g1/scripts/capture_rollouts.py --headless --task-id "$task" --checkpoint "$policy" \
    --num-envs 32 --steps 900 --settle-steps 100 --action-noise "$noise" --command -0.35 0 0 \
    --output "$out" > "logs/contract/capture_check_backward_${env}_${noise}.log" 2>&1 \
    || note "$env $noise capture FAILED"
done
"$python" - <<'EOF' | tee -a "$status"
import sys
from pathlib import Path
sys.path.insert(0, "transfer/scripts")
from w11_accept import check
cells = {
    "play, noise 0 (W11)": "g1/rollouts/base_backward_play_mid.npz",
    "play, noise 0.1": "g1/rollouts/check_base_backward_play_noise0.1.npz",
    "train, noise 0": "g1/rollouts/check_base_backward_train_noise0.npz",
    "train, noise 0.1": "g1/rollouts/check_base_backward_train_noise0.1.npz",
}
for name, path in cells.items():
    if not Path(path).exists():
        print(f"{name:20s} missing")
        continue
    r = check(Path(path), (-0.35, 0.0, 0.0), 0)
    print(f"{name:20s} speed {r['achieved_speed']:+.3f} (ratio {r['speed_ratio']:.2f})"
          f"  duty {max(r['duty']):.2f}  falls {r['falls']}")
EOF
note "check done"
