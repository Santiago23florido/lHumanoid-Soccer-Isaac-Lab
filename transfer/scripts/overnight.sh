#!/usr/bin/env bash
# The night's queue for transfer/docs/skill_space.md §9, one simulator at a time:
#
#   1. decoder smoke   the reconstructed balance against Newton on both bodies, the prior alone,
#                      five PPO iterations with the physics heads; stops the queue if any fails;
#   2. F5              both incumbents follow the probes and the corners, recording balance and
#                      gait phase; then the free latent's training and its cross-body test;
#   3. decoders        v2h (the full decoder v2) and v1 (the plain control) from scratch, then
#                      their ladders (decoder_study.sh);
#   4. re-baseline     the incumbent trackers under this code (the kinematic estimator with the
#                      sole corners): the G1's corners, and L4 with rough_b3, rough_b4,
#                      flat_dagger2 and flat_b4 on both bodies -- F1's deciding comparison;
#   5. ablation        v2 (the prior alone), trained and evaluated.
#
#   transfer/scripts/overnight.sh            (from the repository root, under Start-Process)
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/overnight.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
g1_run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss
nao_run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1

# Windows does not pass a High priority class on to child processes: raise the queue's Python
# processes every two minutes (the RL ran 8.1 -> 6.8 s/it at High next to the user's browser).
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

note "overnight start"
if ! bash transfer/scripts/decoder_study.sh smoke; then note "smoke FAILED: stopping"; exit 1; fi
if ! grep -q "ppo smoke:.*Iteration time" logs/contract/decoder_study.status; then
  note "smoke: PPO did not iterate; stopping"
  exit 1
fi
note "smoke done"

# 2. F5.
export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1
bash transfer/scripts/zero_shot_ladder.sh g1 "$g1_run" model_6194 r3_6194_sym_f5 map corners \
  > logs/contract/ladder_f5_g1.log 2>&1 || note "f5 g1 captures FAILED"
bash transfer/scripts/zero_shot_ladder.sh nao "$nao_run" model_3498 r1_3498_sym_f5 map corners \
  > logs/contract/ladder_f5_nao.log 2>&1 || note "f5 nao captures FAILED"
unset ZS_SYMMETRIZE
note "f5 captures done"
"$python" -u transfer/scripts/train_free_latent.py \
  --g1 g1/rollouts/zs_r3_6194_sym_f5_map.npz --g1 g1/rollouts/zs_r3_6194_sym_f5_corners.npz \
  --nao nao/rollouts/zs_r1_3498_sym_f5_map.npz --nao nao/rollouts/zs_r1_3498_sym_f5_corners.npz \
  --output logs/free_latent/f5_r1/model.pt --results transfer/results/free_latent_f5_r1.json \
  > logs/contract/train_free_latent_f5_r1.log 2>&1 || note "f5 training FAILED"
note "f5: $(grep -E 'NAO held-out|body classifier' logs/contract/train_free_latent_f5_r1.log | tr '\n' ' ')"

# 3. The full decoder and its control.
bash transfer/scripts/decoder_study.sh train v2h v1 || note "decoder training FAILED"
note "decoders v2h v1 trained"
bash transfer/scripts/decoder_study.sh evaluate v2h v1 || note "decoder evaluation FAILED"
note "decoders v2h v1 evaluated"

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
unset ZS_SYMMETRIZE ZS_TASK ZS_GENERATOR

# 5. The prior alone.
bash transfer/scripts/decoder_study.sh train v2 || note "decoder v2 training FAILED"
bash transfer/scripts/decoder_study.sh evaluate v2 || note "decoder v2 evaluation FAILED"
note "overnight done"
