#!/usr/bin/env bash
# F4 of transfer/docs/skill_space.md §9: the G1 tracker whose action is a residual on its body's
# physics (G1TrackPhysics-v0), against the plain tracker (G1Track-v0), both from scratch under
# the incumbent's conditions (the walk's contract, 0.7 procedural, mirror loss, autonomous).
#
#   transfer/scripts/decoder_study.sh <smoke|train|evaluate> [variant...]
#
# smoke     the prior alone (a zero residual) on the walk, true state, against the plain
#           tracker's zero action; then 5 PPO iterations of the physics task on 256 envs.
# train     2000 iterations from scratch per variant: v2 (G1TrackPhysics-v0, HUMANOID_PRIOR=full)
#           and v1 (G1Track-v0), same seed.
# evaluate  the ladder's competence stages (l1, map) and encoder, walk, l4 per variant.
set -euo pipefail

stage="${1:-smoke}"
shift || true
variants=("$@")
if [ ${#variants[@]} -eq 0 ]; then variants=(v2 v1); fi
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
status=logs/contract/decoder_study.status
iterations="${DECODER_ITERATIONS:-2000}"
export HUMANOID_CONTRACT="$root/logs/contract/walk_dagger3e/generator.pt"

task_of() { if [ "$1" = v1 ]; then echo G1Track-v0; else echo G1TrackPhysics-v0; fi; }

if [ "$stage" = smoke ]; then
  echo "smoke start $(date +%H:%M)" >> "$status"
  "$python" - <<'PYEOF'
import torch


class Zero(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 37, device=obs.device)


torch.jit.script(Zero()).save("logs/contract/zero_policy_g1.pt")
PYEOF
  for task in G1Track-Play-v0 G1TrackPhysics-Play-v0; do
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task" \
      --driver-policy logs/contract/zero_policy_g1.pt --num-envs 32 --steps 600 --settle-steps 0 \
      --contract-from-truth --reference autonomous \
      --output "g1/rollouts/decoder_zero_${task%%-*}.npz" > "logs/contract/decoder_zero_${task%%-*}.log" 2>&1
    echo "zero action on ${task}: $(tail -1 "logs/contract/decoder_zero_${task%%-*}.log")" >> "$status"
  done
  "$python" -u scripts/train.py --task G1TrackPhysics-v0 --headless --num_envs 256 \
    --max_iterations 5 --seed 41 --run_name decoder_smoke > logs/contract/decoder_smoke.log 2>&1
  echo "ppo smoke: $(grep -a -E 'Mean reward|Iteration time' logs/contract/decoder_smoke.log | tail -2 | tr '\n' ' ')" >> "$status"
  echo "smoke done $(date +%H:%M)" >> "$status"
fi

if [ "$stage" = train ]; then
  export HUMANOID_PROCEDURAL=0.7 HUMANOID_SYMMETRY=loss HUMANOID_PRIOR=full
  for variant in "${variants[@]}"; do
    echo "train ${variant} start $(date +%H:%M)" >> "$status"
    "$python" -u scripts/train.py --task "$(task_of "$variant")" --headless --num_envs 4096 \
      --max_iterations "$iterations" --seed 41 --run_name "decoder_${variant}" \
      > "logs/contract/train_decoder_${variant}.log" 2>&1
    echo "train ${variant} done $(date +%H:%M)" >> "$status"
  done
fi

if [ "$stage" = evaluate ]; then
  export ZS_REFERENCE=autonomous
  for variant in "${variants[@]}"; do
    if [ "$variant" = v1 ]; then experiment=g1_track; else experiment=g1_track_physics; fi
    run="$(ls -d logs/rsl_rl/${experiment}/*_decoder_${variant} | tail -1)"
    checkpoint="model_$((iterations - 1))"
    export ZS_TASK_ID="$(task_of "$variant" | sed 's/-v0/-Play-v0/')"
    for task in rough flat; do
      export ZS_TASK="$task"
      if [ "$task" = rough ]; then generator=rough_b3; else generator=flat_dagger2; fi
      export ZS_GENERATOR="logs/contract/${generator}/generator.pt"
      stages=(l4)
      if [ "$task" = rough ]; then stages=(l1 map encoder walk l4); fi
      bash transfer/scripts/zero_shot_ladder.sh g1 "$run" "$checkpoint" "dec_${variant}" \
        "${stages[@]}" > "logs/contract/ladder_decoder_${variant}_${task}.log" 2>&1
      echo "evaluate ${variant} ${task} $(date +%H:%M)" >> "$status"
    done
  done
fi
