#!/usr/bin/env bash
# The night's queue for transfer/docs/skill_space.md §9, one simulator at a time:
#
#   1. decoder smoke   the reconstructed balance against Newton on both bodies, the prior alone,
#                      five PPO iterations with the physics heads; stops the queue if any fails;
#   2. F5 captures     both incumbents follow the probes and the corners, recording balance and
#                      gait phase; then the free latent's training and its cross-body test;
#   3. decoder train   v2h, v1, v2 from scratch (decoder_study.sh train);
#   4. decoder eval    the ladder for each (decoder_study.sh evaluate).
#
#   transfer/scripts/overnight.sh            (from the repository root, under Start-Process)
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/overnight.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }

note "overnight start"
if ! bash transfer/scripts/decoder_study.sh smoke; then note "smoke FAILED: stopping"; exit 1; fi
if ! grep -q "ppo smoke:.*Iteration time" logs/contract/decoder_study.status; then
  note "smoke: PPO did not iterate; stopping"
  exit 1
fi
note "smoke done"

export ZS_REFERENCE=autonomous ZS_SYMMETRIZE=1
bash transfer/scripts/zero_shot_ladder.sh g1 logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss \
  model_6194 r3_6194_sym_f5 map corners > logs/contract/ladder_f5_g1.log 2>&1 \
  || note "f5 g1 captures FAILED"
bash transfer/scripts/zero_shot_ladder.sh nao logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1 \
  model_3498 r1_3498_sym_f5 map corners > logs/contract/ladder_f5_nao.log 2>&1 \
  || note "f5 nao captures FAILED"
unset ZS_SYMMETRIZE
note "f5 captures done"
"$python" -u transfer/scripts/train_free_latent.py \
  --g1 g1/rollouts/zs_r3_6194_sym_f5_map.npz --g1 g1/rollouts/zs_r3_6194_sym_f5_corners.npz \
  --nao nao/rollouts/zs_r1_3498_sym_f5_map.npz --nao nao/rollouts/zs_r1_3498_sym_f5_corners.npz \
  --output logs/free_latent/f5_r1/model.pt --results transfer/results/free_latent_f5_r1.json \
  > logs/contract/train_free_latent_f5_r1.log 2>&1 || note "f5 training FAILED"
note "f5: $(grep -E 'NAO held-out|body classifier' logs/contract/train_free_latent_f5_r1.log | tr '\n' ' ')"

bash transfer/scripts/decoder_study.sh train v2h v1 v2 || note "decoder training FAILED"
note "decoder trained"
bash transfer/scripts/decoder_study.sh evaluate v2h v1 v2 || note "decoder evaluation FAILED"
note "overnight done"
