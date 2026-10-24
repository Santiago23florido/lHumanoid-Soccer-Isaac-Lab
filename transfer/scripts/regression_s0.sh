#!/usr/bin/env bash
# The regression suite S0 of transfer/docs/skill_space.md §9, for the incumbents under this code.
#
#   transfer/scripts/regression_s0.sh
#
# 1. Chains: each generator's chain fidelity, recomputed at the body encoder's precision, must
#    equal byte for byte the one recorded when it was measured (chain_study_b4.sh), on the GPU.
# 2. L1b: the walk on every path, the deployed one through the kinematic estimator, for the
#    incumbent trackers of both bodies (their truth and autonomous captures carried over from
#    their own tag: those paths do not read the estimator). Since 2026-10-24 S0's check is the
#    deployed path under the autonomous reference -- the one every tracker trains in and L4
#    deploys with; the physical reference, which no tracker trains in and a body can steer
#    towards a slower gait, is kept as instrumentation (skill_space.md §9).
# (L4 of flat_dagger2 on both bodies, S0's third part, ran in the re-baseline.)
set -uo pipefail

python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/regression_s0.status
note() { echo "$1 $(date +%H:%M)" | tee -a "$status"; }
scratch="$(mktemp -d)"

note "s0 start"
for name in rough_b3 rough_b4 flat_b4; do
  task="${name%%_*}"
  "$python" transfer/scripts/chain_fidelity.py "logs/contract/${name}/generator.pt" \
    --gait "g1/rollouts/${task}_play_forward.npz" --noise-scale 1 \
    --output "$scratch/${name}.json" > /dev/null 2>&1
  if cmp -s <(grep -v '"contract"' "$scratch/${name}.json") \
            <(grep -v '"contract"' "transfer/results/chain_fidelity_${name}_x1.json"); then
    note "chain ${name}: identical"
  else
    note "chain ${name}: DIFFERS"
  fi
done

export ZS_ESTIMATOR=kinematic ZS_SYMMETRIZE=1
for body in g1 nao; do
  if [ "$body" = g1 ]; then
    run=logs/rsl_rl/g1_track/2026-10-14_15-38-39_zs_g1_r3_loss ckpt=model_6194 old=r3_6194_sym
    rollouts=g1/rollouts
  else
    run=logs/rsl_rl/nao_track/2026-10-14_19-59-50_zs_nao_r1 ckpt=model_3498 old=r1_3498_sym
    rollouts=nao/rollouts
  fi
  for path in walk_truth walk_auto; do
    [ -f "$rollouts/zs_${old}_v3_${path}.npz" ] || cp "$rollouts/zs_${old}_${path}.npz" \
      "$rollouts/zs_${old}_v3_${path}.npz"
  done
  bash transfer/scripts/zero_shot_ladder.sh "$body" "$run" "$ckpt" "${old}_v3" walk \
    > "logs/contract/regression_s0_walk_${body}.log" 2>&1 || note "walk ${body} FAILED"
  log="logs/contract/regression_s0_walk_${body}.log"
  note "walk ${body} (S0, autonomous): $(grep -E '^encoder' "$log" | sed -n 2p)"
  note "walk ${body} (physical, instrumentation): $(grep -E '^encoder' "$log" | head -1)"
done
rm -rf "$scratch"
note "s0 done"
