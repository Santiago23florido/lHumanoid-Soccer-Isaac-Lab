#!/usr/bin/env bash
# Why does PPO break a distilled tracker in its first update, and not the incumbent it was
# distilled from? (transfer/docs/skill_space.md §9.1). Six PPO iterations from one distilled
# checkpoint, one change at a time; the collapse shows by the second (base contact ends ~0.98
# of the episodes):
#
#   base       as the warm study runs it;
#   fresh      the optimizer's moments cleared (the distillation's are handed over otherwise);
#   no_mirror  without the mirror loss;
#   slow       PPO's learning rate starting at the incumbent's adapted one (5.06e-5), not 1e-3.
#
#   bash transfer/scripts/ppo_start_ablation.sh <distilled run directory> [variants]
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
distilled="$1"
variants="${2:-base fresh no_mirror slow}"
experiment="$(basename "$(dirname "$distilled")")"
task=G1TrackPhysics-v0
case "$experiment" in g1_track_core) task=G1TrackCore-v0 ;; nao_track_physics) task=NaoTrackPhysics-v0 ;; esac
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"
export HUMANOID_PROCEDURAL=0.7 HUMANOID_HEADS="${HUMANOID_HEADS:-1}" HUMANOID_PLAN_TRUST="${HUMANOID_PLAN_TRUST:-1}"
export HUMANOID_TARGET_RATE=1 HUMANOID_FEEDFORWARD_OBS="${HUMANOID_FEEDFORWARD_OBS:-1}"
out=transfer/results/ppo_start_ablation_$(basename "$distilled").json
echo "{}" > "$out"
for variant in $variants; do
  run="$distilled" symmetry=loss extra=()
  case "$variant" in
    fresh)
      run="${distilled}_fresh_adam"
      mkdir -p "$run"
      "$python" - "$distilled/model_0.pt" "$run/model_0.pt" <<'PYEOF'
import sys, torch
saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
saved["optimizer_state_dict"]["state"] = {}
torch.save(saved, sys.argv[2])
PYEOF
      ;;
    no_mirror) symmetry="" ;;
    slow) extra=(agent.algorithm.learning_rate=5.0625e-05) ;;
  esac
  HUMANOID_SYMMETRY="$symmetry" "$python" -u scripts/train.py --task "$task" --headless \
    --num_envs 4096 --max_iterations 6 --seed 41 --run_name "ablation_$variant" \
    --resume --load_run "$(basename "$run")" --checkpoint model_0.pt "${extra[@]}" \
    > "logs/contract/ppo_start_ablation_$variant.log" 2>&1
  "$python" - "logs/contract/ppo_start_ablation_$variant.log" "$variant" "$out" <<'PYEOF'
import json, re, sys
rows, it = {}, None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"Learning iteration (\d+)/", line)
    if m: it = int(m.group(1)); rows.setdefault(it, {}); continue
    m = re.match(r"\s*(Mean episode length|Mean reward|Mean value loss|Mean symmetry loss|Episode_Termination/base_contact)[^:]*:\s*(-?[\d.e+-]+)", line)
    if m and it is not None: rows[it][m.group(1)] = float(m.group(2))
report = json.load(open(sys.argv[3]))
report[sys.argv[2]] = {str(k): v for k, v in sorted(rows.items())}
json.dump(report, open(sys.argv[3], "w"), indent=2)
last = rows[max(rows)] if rows else {}
print(sys.argv[2], {k: round(v, 3) for k, v in last.items()})
PYEOF
done
