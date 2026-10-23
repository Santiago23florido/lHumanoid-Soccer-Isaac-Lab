#!/usr/bin/env bash
# The pre-registered evaluation ladder of the zero-shot study (transfer/docs/skill_space.md §8.3).
#
#   [ZS_TASK=flat|rough] transfer/scripts/zero_shot_ladder.sh <g1|nao> <run dir> <checkpoint> <tag> [stage...]
#
# Stages, all of them by default:
#   l1       random procedural schedules, as in training (true state):  competence_<body>_<tag>_random
#   map      the pre-registered probes (true state):                    competence_<body>_<tag>
#   corners  the family's corners, some beyond its ranges (true state):  competence_<body>_<tag>_corners
#   encoder  the body's encoder refitted on its own new states -- walk (body's history,
#            autonomous, starts) and procedural from the spawn pose: logs/contract/walk_zs_<body>_<tag>
#   walk     L1b, the walk on every path (deployed through the refitted encoder)
#   l3       the task's generator on its own chain, started inside its gait (true state)
#   l4       the task deployed: the body's own history, true state and encoder, at three
#            commands (the G1's test commands, in the body's units)
# The task side is never retrained here. ZS_TASK picks the task: flat (Isaac Lab's published
# Isaac-Velocity-Flat-G1-v0, semi-blind; generator logs/contract/flat_h20) or rough (the blind
# Isaac-Velocity-Rough-G1-v0 on a plane; generator logs/contract/rough_h20). The body's
# encoder is moved into the task's contract as logs/contract/<task>_zs_<body>_<tag>.
set -euo pipefail

body="$1"
run="$2"
checkpoint="$3"
tag="$4"
shift 4
stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then stages=(l1 map encoder walk l3 l4); fi
python="${PYTHON:-C:/Users/USER/miniconda3/envs/env_isaaclab/python.exe}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
want() { [[ " ${stages[*]} " == *" $1 "* ]]; }

task="${ZS_TASK:-flat}"
# ZS_GENERATOR: another generator of the same task (after task-side DAgger, say).
task_generator="${ZS_GENERATOR:-logs/contract/${task}_h20/generator.pt}"
generator_name="$(basename "$(dirname "$task_generator")")"
# ZS_REFERENCE: what the deployed generator reads in L4 -- the body's history (physical), or
# its own plans, caught up with the body only when the body has left them (autonomous).
reference="${ZS_REFERENCE:-physical}"
mode=""
if [ "$reference" = autonomous ]; then mode="_auto"; fi
# ZS_ESTIMATOR: what the deployed path reads the state with -- the learned encoder, or the
# body's kinematics for every field but the loads (kinematic; its results carry "_kin").
estimator="${ZS_ESTIMATOR:-encoder}"
est=""
if [ "$estimator" = kinematic ]; then est="_kin"; fi
if [ "$task" = rough ]; then
  commands="forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0 0.3"
else
  commands="forward 0.25 0 0
fast 0.6 0 0
turning 0.3 0.2 0.3"
fi
if [ "$body" = g1 ]; then
  # ZS_TASK_ID: another tracking task of the same body (the physics tracker's, say).
  task_id="${ZS_TASK_ID:-G1Track-Play-v0}"
  walk="logs/contract/walk_dagger3e/generator.pt"
  rollouts=g1/rollouts
  mirror=g1/body/mirror.json
  scale="1 1"
else
  task_id="${ZS_TASK_ID:-NaoTrack-Play-v0}"
  walk="logs/contract/walk_dagger2e/generator.pt"
  rollouts=nao/rollouts
  mirror=nao/body/mirror.json
  export HUMANOID_POSTURE_CONTRACT="$root/$walk"
  scale="$("$python" -c "
import math
g1, nao = 0.6545728842979989, 0.24801
print(math.sqrt(nao / g1), math.sqrt(g1 / nao))")"
fi
read -r speed_scale turn_scale <<< "$scale"
walk_zs="logs/contract/walk_zs_${body}_${tag}/generator.pt"
task_zs="logs/contract/${generator_name}_zs_${body}_${tag}/generator.pt"
mkdir -p "$rollouts"

"$python" transfer/scripts/export_actor.py "$run/$checkpoint.pt" > /dev/null
policy="$run/exported/$checkpoint.pt"
# ZS_SYMMETRIZE=1: deploy the tracker averaged with its mirror image (symmetrize_actor.py);
# the tag should say so.
if [ "${ZS_SYMMETRIZE:-0}" = 1 ]; then
  "$python" transfer/scripts/symmetrize_actor.py "$policy" --mirror "$mirror" --contract "$walk" \
    --body "$body" | tail -1
  policy="$run/exported/${checkpoint}_sym.pt"
fi

# Isaac Sim fetches the robot's and the scene's USD from the Omniverse asset server at every
# launch; a capture that could not open one is no result (2026-10-22: two outages took down every
# capture of a ladder). It is retried, up to three times, once the server answers again.
assets_online() {
  local server=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 url
  for url in "$server/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd" \
    "$server/Isaac/Environments/Grid/default_environment.usd"; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$url")" = 200 ] || return 1
  done
}

capture() {
  local name="$1" attempt log
  shift
  log="logs/contract/capture_zs_${tag}_${name}.log"
  for attempt in 1 2 3; do
    until assets_online; do sleep 60; done
    "$python" -u transfer/scripts/capture_contract.py --headless --task-id "$task_id" \
      --driver-policy "$policy" --steps 900 --settle-steps 100 "$@" \
      --output "$rollouts/zs_${tag}_${name}.npz" > "$log" 2>&1
    grep -aqE "Unable to open the usd file at path: https|USD file not found at path at: 'https" "$log" || break
    echo "capture $name: an asset could not be opened (attempt $attempt); retrying"
  done
  tail -1 "$log"
}

export HUMANOID_CONTRACT="$root/$walk"
if want l1; then
  capture random --num-envs 128 --contract-from-truth --reference autonomous --procedural-fraction 1
  "$python" transfer/scripts/evaluate_competence.py --capture "$rollouts/zs_${tag}_random.npz" \
    --output "transfer/results/competence_${body}_${tag}_random.json" | tail -3
fi
if want map; then
  capture map --num-envs 213 --contract-from-truth --reference autonomous \
    --gait-probes transfer/probes/gait_probes.json
  "$python" transfer/scripts/evaluate_competence.py --capture "$rollouts/zs_${tag}_map.npz" \
    --output "transfer/results/competence_${body}_${tag}.json" | head -20
fi
if want corners; then
  # The family's corners, some beyond its ranges (make_gait_probes.py --corners), three envs each.
  capture corners --num-envs 108 --contract-from-truth --reference autonomous \
    --gait-probes transfer/probes/corner_probes.json
  "$python" transfer/scripts/evaluate_competence.py --capture "$rollouts/zs_${tag}_corners.npz" \
    --output "transfer/results/competence_${body}_${tag}_corners.json" | head -20
fi
if want encoder; then
  capture walk_truth --num-envs 32 --contract-from-truth --reference physical
  capture walk_auto --num-envs 32 --contract-from-truth --reference autonomous
  capture walk_starts --num-envs 32 --contract-from-truth --reference physical \
    --settle-steps 0 --standing-fraction 0.3 --resample-seconds 2 5
  capture procedural_starts --num-envs 64 --contract-from-truth --reference autonomous \
    --settle-steps 0 --procedural-fraction 1
  "$python" transfer/scripts/train_body_encoder.py "$walk" \
    --data "$rollouts/zs_${tag}_walk_truth.npz" --data "$rollouts/zs_${tag}_walk_auto.npz" \
    --data "$rollouts/zs_${tag}_walk_starts.npz" --data "$rollouts/zs_${tag}_procedural_starts.npz" \
    --output "$walk_zs" --results "transfer/results/encoder_zs_${body}_${tag}.json" | tail -2
fi
if want walk; then
  export HUMANOID_CONTRACT="$root/$walk_zs"
  capture "walk${est}_encoder" --num-envs 32 --reference physical --estimator "$estimator"
  "$python" transfer/scripts/evaluate_contract.py \
    --run teacher=g1/rollouts/closed_loop_teacher.npz \
    --run "encoder=$rollouts/zs_${tag}_walk${est}_encoder.npz" \
    --run "truth=$rollouts/zs_${tag}_walk_truth.npz" \
    --run "autonomous=$rollouts/zs_${tag}_walk_auto.npz" \
    --output "transfer/results/zero_shot_${body}_${tag}_walk${est}.json" | grep -v "^  "
fi
if want l3 || want l4; then
  if [ ! -f "$task_zs" ]; then
    "$python" transfer/scripts/body_encoder.py "$task_generator" "$walk_zs" --output "$task_zs"
  fi
fi
if want l3; then
  export HUMANOID_CONTRACT="$root/$task_zs"
  export HUMANOID_START_STATES="$root/g1/rollouts/${task}_dart.npz"
  export HUMANOID_START_WINDOWS="$root/g1/rollouts/${task}_dart.npz"
  capture "${generator_name}_chain" --num-envs 64 --contract-from-truth --reference autonomous \
    --start-fraction 1
  "$python" transfer/scripts/evaluate_competence.py --all \
    --capture "$rollouts/zs_${tag}_${generator_name}_chain.npz" \
    --output "transfer/results/zero_shot_${body}_${tag}_${generator_name}_l3.json" | grep -A7 '"checks"'
  unset HUMANOID_START_STATES HUMANOID_START_WINDOWS
fi
if want l4; then
  export HUMANOID_CONTRACT="$root/$task_zs"
  while read -r name vx vy wz; do
    vx=$("$python" -c "print($vx * $speed_scale)")
    vy=$("$python" -c "print($vy * $speed_scale)")
    wz=$("$python" -c "print($wz * $turn_scale)")
    capture "${generator_name}${mode}${est}_${name}_encoder" --num-envs 32 \
      --reference "$reference" --estimator "$estimator" --command "$vx" "$vy" "$wz"
    # The true-state path does not depend on the estimator: reuse it when it exists.
    truth="$rollouts/zs_${tag}_${generator_name}${mode}_${name}_truth.npz"
    if [ -z "$est" ] || [ ! -f "$truth" ]; then
      capture "${generator_name}${mode}_${name}_truth" --num-envs 32 --reference "$reference" \
        --contract-from-truth --command "$vx" "$vy" "$wz"
    fi
    "$python" transfer/scripts/evaluate_contract.py \
      --run "teacher=g1/rollouts/${task}_play_${name}.npz" \
      --run "encoder=$rollouts/zs_${tag}_${generator_name}${mode}${est}_${name}_encoder.npz" \
      --run "truth=$truth" \
      --output "transfer/results/zero_shot_${body}_${tag}_${generator_name}${mode}${est}_l4_${name}.json" \
      | grep -v "^  [^a]"
  done <<< "$commands"
fi
