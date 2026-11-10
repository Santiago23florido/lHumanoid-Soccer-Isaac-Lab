#!/usr/bin/env bash
# W11's second declared deviation (skill_space.md §10.12), the reading the 2 x 2 check
# pre-registered: backward passes at -0.35 m/s in its training variant without action noise
# (0 falls) and falls in the play variant (121). Its three fixed-command rollouts are captured
# again in the training variant (G1Base-Backward-v0: observation noise and pushes on), with
# action noise 0 and the play protocol otherwise (32 environments, 900 steps, settle 100):
# g1/rollouts/base_backward_train_<name>.npz. They are the calibrated acceptance reading and
# backward's L4 references; the play rollouts stay as the registered reading. Then W11's
# acceptance is read again. One Isaac process; runs after the check. Ends with "done".
#
#   bash transfer/scripts/w11_backward_train.sh
set -uo pipefail
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/w11_backward_train.status
note() { echo "$1 $(date +%Y-%m-%d_%H:%M)" | tee -a "$status"; }
policy="$(ls -d logs/rsl_rl/g1_flat/*_base_backward | tail -1)/exported/model_1499.pt"

until grep -q "^check done" logs/contract/w11_play_check.status 2>/dev/null; do sleep 60; done
note "start"
while read -r cname vx vy wz; do
  out="g1/rollouts/base_backward_train_${cname}.npz"
  [ -f "$out" ] && continue
  "$python" -u g1/scripts/capture_rollouts.py --headless --task-id G1Base-Backward-v0 \
    --checkpoint "$policy" --num-envs 32 --steps 900 --settle-steps 100 --action-noise 0 \
    --command "$vx" "$vy" "$wz" --output "$out" \
    > "logs/contract/capture_base_backward_train_${cname}.log" 2>&1 || note "$cname capture FAILED"
  note "captured $cname"
done < <("$python" -c "
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding='utf-8').read())
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'PLAY_COMMANDS':
        for name, (vx, vy, wz) in ast.literal_eval(node.value)['backward'].items():
            print(name, vx, vy, wz)" \
  source/humanoid_transfer/humanoid_transfer/g1/tasks/g1_base/g1_base_env_cfg.py)
"$python" transfer/scripts/w11_accept.py --output transfer/results/w11_experts.json \
  > logs/contract/w11_accept.out 2>&1 || note "accept FAILED"
note "decision: $(grep '^accepted' logs/contract/w11_accept.out | tr '\n' ' ')"
note "done"
