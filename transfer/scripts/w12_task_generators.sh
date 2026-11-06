#!/usr/bin/env bash
# W12 of skill_space.md §10.13: one generator per new base task, step 2 of method_revision.md §6.
# For each task W11 accepted (transfer/results/w11_experts.json, its calibrated reading: the
# deviation declared in skill_space.md §10.12), its contract generator is
# trained on its G1 expert's DART rollouts with rough_b4's and flat_b4's recipe:
#   horizon 20, commit 13, midpoint solver with 8 flow steps, log-warped loads;
#   the G1 body's encoder (walk_zs_g1_r3_6194_sym), trained under its noise and
#   noise-conditional.
# There is no DAgger data: a DAgger round needs a G1 decoder that executes the task, which
# exists only after step 3. The gate is train_contract.py's own, fixed before training: the
# instant each foot's contact next changes is calibrated to within 0.05 on held-out data.
# Nothing of rough enters, and nothing of either body is trained.
#
#   bash transfer/scripts/w12_task_generators.sh
#
# Runs after W11 (w11_experts.sh). Every generator is skipped when it exists. It always ends
# with "w12 done".
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w12_generators.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
body=logs/contract/walk_zs_g1_r3_6194_sym/generator.pt

# The play rollout each generator's chain fidelity is read on: the middle command of each task.
declare -A gait=([backward]=mid [sidestep]=left [jog]=mid)

until grep -q "^w11 done" logs/contract/w11_experts.status 2>/dev/null; do sleep 120; done
note "w12 start"
accepted=$("$python" -c "
import json, sys
print(' '.join(json.load(open(sys.argv[1]))['accepted_calibrated']))" transfer/results/w11_experts.json \
  2>/dev/null)
[ -n "$accepted" ] || { note "no accepted task"; note "w12 done"; exit 0; }
note "accepted: $accepted"

for task in $accepted; do
  name="base_${task}_b4"
  if [ -f "logs/contract/${name}/generator.pt" ]; then note "$name: exists"; continue; fi
  note "$name training start"
  "$python" -u transfer/scripts/train_contract.py --name "$name" \
    --data "g1/rollouts/base_${task}_dart.npz" \
    --gait "g1/rollouts/base_${task}_play_${gait[$task]}.npz" \
    --encoder-from "$body" --warp-loads --solver midpoint --flow-steps 8 --horizon 20 \
    --commit 13 --sigma-from-body --noise-conditional \
    --results "transfer/results/contract_${name}.json" \
    > "logs/contract/train_${name}.log" 2>&1 || { note "$name training FAILED"; continue; }
  note "$name: $(grep -E '^gate|chain fidelity' "logs/contract/train_${name}.log" | tr '\n' ' ')"
done
note "w12 done"
