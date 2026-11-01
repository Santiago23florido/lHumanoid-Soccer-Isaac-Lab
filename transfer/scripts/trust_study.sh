#!/usr/bin/env bash
# Decoder v2, round 2 (transfer/docs/skill_space.md §9.1): the prior trusts each plan by its
# measured sampling noise and aims the trunk at its slow tilt; the biped's plans carry that
# noise and the gait's trunk sway. One simulator at a time:
#
#   1. smoke      the prior alone (zero residual) on the rough generator's plans, through the
#                 deployed path, without and with trust; on procedural plans with sway and
#                 noise; five PPO iterations of v2t. Stops the queue if any fails.
#   2. train      v2t from scratch, under v2h's conditions (decoder_study.sh train v2t).
#   3. evaluate   its ladder (decoder_study.sh evaluate v2t), against v2h and v1.
#   4. rebaseline the incumbent trackers under this code: the G1's corners, and L4 with rough_b3,
#                 rough_b4, flat_dagger2 and flat_b4 on both bodies -- F1's deciding comparison.
#
#   transfer/scripts/trust_study.sh          (from the repository root, under Start-Process)
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/trust_study.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
keeper() {
  while true; do
    powershell.exe -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | \
      ForEach-Object { try { \$_.PriorityClass = 'High' } catch {} }" > /dev/null 2>&1
    sleep 120
  done
}
keeper &
keeper_pid=$!
trap 'kill $keeper_pid 2>/dev/null' EXIT
g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1

note "trust study start"
# 1. Smoke.
zero() {
  local name="$1"
  shift
  "$python" -u transfer/scripts/capture_contract.py --headless --task-id G1TrackPhysics-Play-v0 \
    --driver-policy logs/contract/zero_policy_g1.pt --num-envs 16 --steps 300 --settle-steps 0 \
    "$@" --output "g1/rollouts/trust_zero_${name}.npz" > "logs/contract/trust_zero_${name}.log" 2>&1
  note "zero residual ${name}: $(tail -1 "logs/contract/trust_zero_${name}.log")"
}
for trust in 0 1; do
  HUMANOID_PLAN_TRUST=$trust HUMANOID_CONTRACT="$root/logs/contract/rough_b3_zs_g1_r3_6194_sym/generator.pt" \
    zero "rough_trust${trust}" --reference autonomous --estimator kinematic --command 0.25 0 0
done
HUMANOID_PLAN_TRUST=1 HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" \
  zero procedural_trust1 --reference autonomous --contract-from-truth --procedural-fraction 1
if ! grep -q "zero residual procedural_trust1: G1TrackPhysics" "$status"; then
  note "smoke FAILED: stopping"
  exit 1
fi
HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt" HUMANOID_HEADS=1 \
  HUMANOID_PLAN_TRUST=1 HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss \
  "$python" -u scripts/train.py --task G1TrackPhysics-v0 --headless --num_envs 256 \
  --max_iterations 5 --seed 41 --run_name trust_smoke > logs/contract/trust_smoke.log 2>&1
if ! grep -a -q "Iteration time" logs/contract/trust_smoke.log; then
  note "smoke: PPO did not iterate; stopping"
  exit 1
fi
note "smoke done"

# 2-3. Train and evaluate v2t.
bash transfer/scripts/decoder_study.sh train v2t || note "v2t training FAILED"
note "v2t trained"
bash transfer/scripts/decoder_study.sh evaluate v2t || note "v2t evaluation FAILED"
note "v2t evaluated"

# 4. The incumbents under this code, under their own tag (their encoders copied to it).
export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1 ZS_ESTIMATOR=kinematic
for body in g1 nao; do
  if [ "$body" = g1 ]; then run=$g1_run ckpt=model_6194 old=r3_6194_sym; else
    run=$nao_run ckpt=model_3498 old=r1_3498_sym; fi
  new="${old}_v3"
  for generator in walk rough_b3 flat_dagger2; do
    source_dir="logs/contract/${generator}_zs_${body}_${old}"
    [ -d "$source_dir" ] && cp -r "$source_dir" "logs/contract/${generator}_zs_${body}_${new}"
  done
  if [ "$body" = g1 ]; then
    bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "$ckpt" "$new" corners \
      > logs/contract/ladder_rebaseline_g1_corners.log 2>&1 || note "rebaseline g1 corners FAILED"
  fi
  for generator in rough_b3 rough_b4 flat_dagger2 flat_b4; do
    export ZS_TASK="${generator%%_*}" ZS_GENERATOR="logs/contract/${generator}/generator.pt"
    bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$ckpt" "$new" l4 \
      > "logs/contract/ladder_rebaseline_${body}_${generator}.log" 2>&1 \
      || note "rebaseline ${body} ${generator} FAILED"
  done
  note "rebaseline ${body} done"
done
note "trust study done"
